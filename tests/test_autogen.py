"""Tests for the AutoGen bridge, against autogen-core / autogen-agentchat 0.7.5.

No network. The catalogue is served from ``live_manifest.json`` — 47 real capabilities with
their real schemas — through an ``httpx.MockTransport``, and the invoke path is fed a stub
agent injected via ``HubClient(agent=...)``.

Tests use ``asyncio.run`` inside synchronous test functions rather than pytest-asyncio: the
bridge's whole async surface is one coroutine, and not depending on a plugin's mode
configuration keeps this file runnable in any of the three framework venvs.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage
from autogen_core.tools import BaseTool, StaticWorkbench
from pydantic import BaseModel

from aimarket_bridges.autogen import AIMarketTool, CapabilityResult, aimarket_tools
from aimarket_bridges.catalog import Capability, CatalogError, fetch_catalog
from aimarket_bridges.client import HubClient, HubUnavailable
from aimarket_bridges.receipts import ReceiptCheck

MANIFEST = json.loads((Path(__file__).parent / "live_manifest.json").read_text())
HUB = "https://modelmarket.dev"


# ── fixtures ────────────────────────────────────────────────────────────────


def _mock_client() -> httpx.Client:
    """An httpx client that serves the captured manifest and nothing else.

    Any request other than the manifest fails loudly: a test that silently reached the network
    would pass for the wrong reason, which is the failure mode these rules exist to prevent.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ai-market/v2/manifest":
            return httpx.Response(200, json=MANIFEST)
        raise AssertionError(f"unexpected network call to {request.url}")

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(scope="module")
def catalog() -> list[Capability]:
    with _mock_client() as http:
        return fetch_catalog(HUB, client=http)


@pytest.fixture(scope="module")
def by_id(catalog: list[Capability]) -> dict[str, Capability]:
    return {c.capability_id: c for c in catalog}


class StubAgent:
    """Stands in for ``AIMarketAgent``; the core calls exactly one method on it."""

    def __init__(self, body: Any = None, *, delay: float = 0.0, raises: Exception | None = None):
        self._body = body if body is not None else {"ok": True, "output": {"beta": "ab12"}}
        self._delay = delay
        self._raises = raises
        self.calls: list[dict[str, Any]] = []
        self.threads: list[int] = []
        self._lock = threading.Lock()

    def invoke_single(self, **kw: Any) -> Any:
        if self._delay:
            time.sleep(self._delay)
        with self._lock:
            self.calls.append(kw)
            self.threads.append(threading.get_ident())
        if self._raises is not None:
            raise self._raises
        return self._body


def make_hub(agent: StubAgent, *, budget_usd: float = 1.0) -> HubClient:
    # verify_receipts=False by default so the receipt tests can opt in explicitly; the
    # resolver would otherwise want a well-known document over the network.
    return HubClient(HUB, budget_usd=budget_usd, verify_receipts=False, agent=agent)


def make_tool(cap: Capability, agent: StubAgent, **kw: Any) -> AIMarketTool:
    return AIMarketTool(cap, make_hub(agent, **kw))


def args_for(tool: AIMarketTool, **values: Any) -> BaseModel:
    return tool.args_type()(**values)


# ── the framework facts the design rests on ─────────────────────────────────


def test_basetool_run_is_async_and_the_only_abstract_method():
    """If either fact changes, the offload in `run` is either unnecessary or insufficient."""
    assert asyncio.iscoroutinefunction(BaseTool.run)
    assert BaseTool.__abstractmethods__ == frozenset({"run"})


def test_strict_mode_would_make_two_thirds_of_the_hub_unusable(by_id, catalog):
    """Why `strict=False`, measured rather than asserted in a comment.

    Under strict, `BaseTool.schema` refuses to build unless every property is required — and 31
    of the 47 live capabilities declare at least one optional one. The guarantee strict buys is
    worth nothing to a capability that already validates its own input; losing two thirds of the
    catalogue to get it is not a trade, so the number is pinned here.
    """
    optional = [
        c for c in catalog
        if set(c.input_schema.get("properties") or {}) - set(c.input_schema.get("required") or [])
    ]
    assert len(optional) == 31

    class Strict(BaseTool[BaseModel, CapabilityResult]):
        async def run(self, args: BaseModel, cancellation_token: Any = None) -> CapabilityResult:
            raise NotImplementedError

    for cid in ("sortes.draw@v1", "kantor.transport@v1", "fermat.route@v1"):
        tool = Strict(args_type=by_id[cid].args_model(), return_type=CapabilityResult,
                      name="probe", description="d", strict=True)
        with pytest.raises(ValueError, match="not all input arguments are marked as required"):
            tool.schema


def test_unregistering_a_cancel_callback_degrades_rather_than_raising():
    """`CancellationToken` has no public remove, so the unlink reaches into `_callbacks`.

    That is a deliberate bet on a private name, and the cost of losing it must be the old
    unbounded growth — never an exception thrown out of a paid call. Pinned for all three ways
    it can miss: renamed internals, no token, and a callback that already fired.
    """
    from aimarket_bridges.autogen import _forget_cancel_callback

    class Renamed:
        pass

    _forget_cancel_callback(None, None)
    _forget_cancel_callback(Renamed(), lambda: None)  # type: ignore[arg-type]

    token = CancellationToken()
    _forget_cancel_callback(token, lambda: None)  # never registered
    assert token._callbacks == []


def test_return_type_must_be_a_pydantic_model():
    """Why `CapabilityResult` exists rather than returning a plain dict.

    ``ReturnT`` is bound to ``BaseModel``, and ``return_value_as_string`` only produces JSON
    for a ``BaseModel`` — a dict falls through to ``str()``, i.e. Python repr.
    """
    import autogen_core.tools._base as base

    assert base.ReturnT.__bound__ is BaseModel
    assert issubclass(CapabilityResult, BaseModel)

    # The concrete consequence, measured rather than assumed.
    probe = AIMarketTool.__mro__[1]  # BaseTool
    assert probe.return_value_as_string(None, {"beta": "ab12"}) == "{'beta': 'ab12'}"
    with pytest.raises(json.JSONDecodeError):
        json.loads(probe.return_value_as_string(None, {"beta": "ab12"}))


# ── the schema a model sees ─────────────────────────────────────────────────


def test_schema_of_a_real_capability_is_exact(by_id):
    """The whole schema for ``sortes.draw@v1``, asserted field by field.

    This is what the model is shown, and it is the thing most likely to regress silently: a
    lost description, a required field going optional, or ``additionalProperties`` flipping
    would all still produce a tool that builds and runs.
    """
    tool = make_tool(by_id["sortes.draw@v1"], StubAgent())
    schema = tool.schema

    assert schema["name"] == "sortes_draw_v1"
    assert schema["strict"] is False
    params = schema["parameters"]
    assert params["type"] == "object"
    assert params["additionalProperties"] is False
    assert params["required"] == ["alpha"]
    assert set(params["properties"]) == {"alpha", "num_bytes"}

    # Required string: a plain type, and the manifest's description carried through — it is
    # the only hint the model gets about the 'hex:' prefix.
    alpha = params["properties"]["alpha"]
    assert alpha["type"] == "string"
    assert "prefix 'hex:' for raw bytes" in alpha["description"]

    # Optional integer with a schema default. It becomes `int | None` with the default
    # attached, so the model may omit it; `_payload` then drops nothing because pydantic
    # fills in 32.
    num_bytes = params["properties"]["num_bytes"]
    assert num_bytes["anyOf"] == [{"type": "integer"}, {"type": "null"}]
    assert num_bytes["default"] == 32
    assert "Length of the derived uniform" in num_bytes["description"]

    # Price is in the description, because choosing between two fitting capabilities on cost
    # is a decision only the model can make.
    assert "$0.0060 per call" in schema["description"]
    assert "oracles.modelmarket.dev/family" in schema["description"]


def test_enum_reaches_the_model_as_an_enum(by_id):
    """``skopos.briefing@v1``'s language enum must survive Literal -> JSON Schema."""
    tool = make_tool(by_id["skopos.briefing@v1"], StubAgent())
    language = tool.schema["parameters"]["properties"]["language"]
    branches = language["anyOf"]
    assert {"enum": ["en", "ru", "es"], "type": "string"} in branches
    assert {"type": "null"} in branches
    assert language["default"] == "en"


def test_capability_with_no_arguments_produces_an_empty_object_schema(by_id, catalog):
    """12 of the 47 take no arguments; `BaseTool.schema` indexes ``["properties"]`` directly,
    so an empty model that omitted the key would raise instead of building."""
    assert sum(1 for c in catalog if not (c.input_schema.get("properties") or {})) == 12
    tool = make_tool(by_id["platon.random@v1"], StubAgent())
    params = tool.schema["parameters"]
    assert params["properties"] == {}
    assert params["required"] == []
    assert json.loads(asyncio.run(_call(tool, {})).model_dump_json())["ok"] is True


def test_nested_objects_are_inlined_not_left_as_refs(by_id):
    """``fermat.route@v1`` has a nested object and ``oneOf`` in ``items``.

    Nested models make pydantic emit ``$defs``/``$ref``; `BaseTool.schema` resolves them with
    jsonref. A model handed an unresolved ``$ref`` cannot construct the argument at all, so
    this is a hard requirement, not a nicety.
    """
    tool = make_tool(by_id["fermat.route@v1"], StubAgent())
    blob = json.dumps(tool.schema)
    assert "$ref" not in blob
    assert "$defs" not in blob

    params = tool.schema["parameters"]
    assert set(params["required"]) == {"edges", "goal", "start"}
    # The nested 'blend' object is present with its own properties spelled out.
    blend = params["properties"]["blend"]
    nested = next(b for b in blend["anyOf"] if b.get("properties"))
    assert set(nested["properties"]) == {"cost", "latency", "latency_scale", "reputation"}


def test_kantor_polymorphic_points_reach_the_model_as_a_union(by_id):
    """``kantor.transport@v1``'s ``oneOf`` — a point is an array of numbers OR a bare number.

    The failure this guards is a union that silently collapses to ``{}`` or to one branch:
    either way the model is told a shape the capability does not accept, and finds out only
    after being billed.
    """
    tool = make_tool(by_id["kantor.transport@v1"], StubAgent())
    points = tool.schema["parameters"]["properties"]["source_points"]

    # The property itself is optional, hence the outer null branch; the inner list is the
    # interesting part.
    array = next(b for b in points["anyOf"] if b.get("type") == "array")
    assert array["items"]["anyOf"] == [
        {"items": {"type": "number"}, "type": "array"},
        {"type": "number"},
    ]
    # The 2-D `cost` matrix is not flattened either.
    cost = by_id["kantor.transport@v1"]
    matrix = next(b for b in tool.schema["parameters"]["properties"]["cost"]["anyOf"]
                  if b.get("type") == "array")
    assert matrix["items"] == {"items": {"type": "number"}, "type": "array"}
    assert set(tool.schema["parameters"]["required"]) == {"a", "b"}
    assert cost.capability_id == "kantor.transport@v1"


def test_a_property_pydantic_had_to_rename_is_still_advertised_and_sent_by_its_real_name(by_id):
    """``fourier.verify@v1`` requires an argument called ``lambda`` — a Python keyword.

    ``_safe_name`` must rename the pydantic field to ``lambda_``, but the rename has to be
    invisible on the wire in BOTH directions. Advertising ``lambda_`` tells the model to send a
    key the capability has never heard of, and dumping ``lambda_`` sends one, so every call to
    this capability would be a paid refusal. This is the only capability of the 47 where a
    naming rule alone breaks it.
    """
    agent = StubAgent()
    tool = make_tool(by_id["fourier.verify@v1"], agent)
    params = tool.schema["parameters"]

    assert "lambda" in params["properties"] and "lambda_" not in params["properties"]
    assert set(params["required"]) == {"edges", "lambda", "vector"}
    assert params["properties"]["lambda"]["description"] == "Claimed eigenvalue to certify."

    # run_json validates the raw arguments the model produced, so validation must accept the
    # advertised spelling.
    args = tool.args_type().model_validate({"edges": [["a", "b"]], "lambda": 0.5, "vector": [1.0, -1.0]})
    asyncio.run(tool.run(args, CancellationToken()))
    assert agent.calls[0]["input_payload"]["lambda"] == 0.5
    assert "lambda_" not in agent.calls[0]["input_payload"]

    # And the pydantic field name still validates, for callers written against the model
    # rather than against the schema (populate_by_name).
    assert tool.args_type().model_validate(
        {"edges": [["a", "b"]], "lambda_": 0.5, "vector": [1.0]}
    ).model_dump(by_alias=True)["lambda"] == 0.5


def test_the_rewritten_names_survive_run_json_the_way_the_framework_calls_it(by_id):
    """The same round-trip as above, but through the entry point AutoGen actually uses.

    The test above hands `run` an already-validated model, which is the one path a model never
    takes. Every real call arrives at ``run_json(raw_mapping, token, call_id)`` — the workbench
    passes ``json.loads(tool_call.arguments)`` straight in — and `run_json` validates with
    ``self._args_type.model_validate(args)``. So the raw key really is the capability's own
    spelling, and validation has to accept ``lambda`` / nested ``from`` as *input* keys, not
    only emit them on the way out. A model built without ``populate_by_name`` and an alias
    would pass the test above and fail every call in production.
    """
    agent = StubAgent()
    tool = make_tool(by_id["fourier.verify@v1"], agent)
    asyncio.run(tool.run_json(
        {"edges": [["a", "b"]], "lambda": 0.5, "vector": [1.0, -1.0]},
        CancellationToken(), call_id="call-1",
    ))
    assert agent.calls[0]["input_payload"]["lambda"] == 0.5
    assert "lambda_" not in agent.calls[0]["input_payload"]

    agent = StubAgent()
    tool = make_tool(by_id["fermat.route@v1"], agent)
    asyncio.run(tool.run_json(
        {"edges": [{"from": "a", "to": "b", "cost": 1}], "start": "a", "goal": "b"},
        CancellationToken(), call_id="call-2",
    ))
    assert agent.calls[0]["input_payload"]["edges"] == [{"from": "a", "to": "b", "cost": 1.0}]


def test_a_dict_shaped_edge_keeps_its_own_keys_and_travels_without_nulls(by_id):
    """``fermat.route@v1``'s edges accept ``[u, v, w]`` or ``{from, to, cost, ...}``.

    Two ways for the object form to be quietly corrupted, both of them paid for: ``from`` is a
    keyword, so the nested model calls it ``from_`` and an un-aliased dump drops the edge's
    source node; and the object branch declares eleven optional keys, so a dump that only
    strips top-level ``None`` sends nine nulls inside every edge — into a schema where no
    property is nullable.
    """
    agent = StubAgent()
    tool = make_tool(by_id["fermat.route@v1"], agent)
    args = tool.args_type().model_validate({
        "edges": [["a", "b", 1], {"from": "b", "to": "c", "cost": 2}],
        "start": "a", "goal": "c",
        "blend": {"cost": 1, "latency": 1},
    })

    asyncio.run(tool.run(args, CancellationToken()))
    sent = agent.calls[0]["input_payload"]

    assert sent["edges"][1] == {"from": "b", "to": "c", "cost": 2.0}
    # The nested object argument is pruned the same way, not just the top level.
    assert sent["blend"] == {"cost": 1.0, "latency": 1.0}
    assert "$ref" not in json.dumps(tool.schema)
    nested = next(b for b in tool.schema["parameters"]["properties"]["edges"]["items"]["anyOf"]
                  if b.get("properties"))
    assert "from" in nested["properties"] and "from_" not in nested["properties"]


def test_a_legitimately_nullable_map_value_is_not_pruned(by_id):
    """The one place ``null`` is legal in the catalogue, and the reason the prune is
    ``exclude_none`` rather than a recursive strip.

    ``fermat.verify@v1.potentials`` is ``additionalProperties: {"type": ["number", "null"]}``,
    so it is an opaque mapping whose values may be null. A blanket recursive strip would delete
    a node's potential and change the certificate being verified.
    """
    agent = StubAgent()
    tool = make_tool(by_id["fermat.verify@v1"], agent)
    args = tool.args_type().model_validate({
        "edges": [["a", "b", 1]], "start": "a", "goal": "b",
        "path": ["a", "b"], "potentials": {"a": 0.0, "b": None},
    })

    asyncio.run(tool.run(args, CancellationToken()))

    assert agent.calls[0]["input_payload"]["potentials"] == {"a": 0.0, "b": None}


def test_every_live_capability_builds_a_valid_tool(catalog):
    """All 47, with names a model provider will accept and unique across the set."""
    assert len(catalog) == 47
    agent = StubAgent()
    hub = make_hub(agent)
    tools = [AIMarketTool(cap, hub) for cap in catalog]

    names = [t.name for t in tools]
    assert len(set(names)) == len(names)
    for tool in tools:
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tool.name), tool.name
        schema = tool.schema
        assert schema["description"].strip()
        assert schema["parameters"]["type"] == "object"


# ── not blocking the event loop ─────────────────────────────────────────────


async def _call(tool: AIMarketTool, values: dict[str, Any]) -> CapabilityResult:
    return await tool.run(args_for(tool, **values), CancellationToken())


def test_invoke_runs_off_the_event_loop(by_id):
    """The loop must stay live while a blocking invoke is in flight.

    A ticker sharing the loop has to keep firing. If `run` awaited the blocking call directly
    the ticker would be frozen for the whole invoke — which in AutoGen means every other agent
    and every token stream frozen too.
    """
    agent = StubAgent(delay=0.30)
    tool = make_tool(by_id["sortes.draw@v1"], agent)

    async def main() -> tuple[int, int]:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.005)
                ticks += 1

        beat = asyncio.ensure_future(ticker())
        result = await _call(tool, {"alpha": "hi"})
        beat.cancel()
        assert result.ok
        return ticks, threading.get_ident()

    ticks, loop_thread = asyncio.run(main())
    assert ticks > 10, f"the event loop only advanced {ticks} times — it was blocked"
    # And the blocking work really happened somewhere else.
    assert agent.threads and agent.threads[0] != loop_thread


def test_concurrent_calls_overlap(by_id):
    """Three 0.25s invokes must take ~0.25s, not ~0.75s."""
    agent = StubAgent(delay=0.25)
    hub = make_hub(agent)
    tools = [AIMarketTool(by_id["sortes.draw@v1"], hub) for _ in range(3)]

    async def main() -> float:
        started = time.monotonic()
        results = await asyncio.gather(*(_call(t, {"alpha": "hi"}) for t in tools))
        assert all(r.ok for r in results)
        return time.monotonic() - started

    elapsed = asyncio.run(main())
    assert elapsed < 0.5, f"calls serialised: {elapsed:.2f}s for 3 x 0.25s"
    # The shared spend counter still saw all three, which is what makes the offload safe.
    assert hub.spent_usd == pytest.approx(0.018)


# ── cancellation, honestly ──────────────────────────────────────────────────


def test_cancelled_before_dispatch_costs_nothing(by_id):
    agent = StubAgent()
    tool = make_tool(by_id["sortes.draw@v1"], agent)
    token = CancellationToken()
    token.cancel()

    result = asyncio.run(tool.run(args_for(tool, alpha="hi"), token))

    assert result.ok is False
    assert result.cancelled is True
    assert agent.calls == [], "a cancelled call must not reach the hub"
    assert tool.hub.spent_usd == 0.0
    assert "nothing was spent" in tool.return_value_as_string(result)


def test_cancelled_mid_flight_stops_waiting_but_not_the_charge(by_id):
    """The documented limit of cancellation, pinned as a test.

    Cancelling unblocks the caller immediately, but the worker thread cannot be killed: the
    invoke completes and the operator is billed. Asserting this keeps the module docstring
    honest — if a future version could really abort the request, this test fails and the claim
    gets rewritten.
    """
    agent = StubAgent(delay=0.35)
    tool = make_tool(by_id["sortes.draw@v1"], agent)

    async def main() -> float:
        token = CancellationToken()
        task = asyncio.ensure_future(tool.run(args_for(tool, alpha="hi"), token))
        await asyncio.sleep(0.05)
        token.cancel()
        started = time.monotonic()
        with pytest.raises(asyncio.CancelledError):
            await task
        return time.monotonic() - started

    waited = asyncio.run(main())
    assert waited < 0.1, f"cancellation did not return promptly: {waited:.2f}s"

    # The worker has to be waited for EXPLICITLY. asyncio.run() joins the loop's DEFAULT
    # executor on the way out, and this adapter deliberately no longer uses it: a hub that
    # accepts a connection and then trickles bytes holds its worker for the full timeout, and
    # on the shared default pool enough concurrent tool calls would starve every unrelated
    # to_thread user in the process. The free join was a convenience of that shared pool, not
    # a property of cancellation — the call still runs and is still billed, which is the thing
    # being asserted.
    deadline = time.monotonic() + 5.0
    while not agent.calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(agent.calls) == 1, "the request was not aborted — it ran and was billed"
    assert tool.hub.spent_usd == pytest.approx(0.006)
    # No result exists for a call that was abandoned. `last_result` must not still be holding
    # the PREVIOUS call's receipt, or provenance gets attributed to the wrong invoke.
    assert tool.last_result is None
    assert tool.hub.last_receipt is None  # this stub returns no receipt at all


def test_a_long_lived_token_does_not_collect_a_callback_per_call(by_id):
    """One token, many paid calls — the token must not end up holding all of them.

    `CancellationToken._callbacks` has no remove, and a `link_future`/`add_callback` entry
    closes over the call's task, which holds its `CapabilityResult` and receipt. The token is
    not per call: `AssistantAgent` reuses one for every tool call in a run and a team threads a
    single one through every agent turn, so a registration left behind pins every paid result
    of the session — an unbounded leak, and precisely the provenance this module keeps out of
    the way everywhere else.
    """
    agent = StubAgent()
    tool = make_tool(by_id["sortes.draw@v1"], agent, budget_usd=100.0)
    token = CancellationToken()

    async def main() -> None:
        for _ in range(40):
            assert (await tool.run(args_for(tool, alpha="hi"), token)).ok

    asyncio.run(main())

    assert len(agent.calls) == 40
    assert token._callbacks == [], f"{len(token._callbacks)} callbacks retained after 40 calls"
    # Still cancellable afterwards: unregistering a finished call must not unhook the mechanism.
    token.cancel()
    result = asyncio.run(tool.run(args_for(tool, alpha="hi"), token))
    assert result.cancelled is True
    assert len(agent.calls) == 40, "a cancelled call reached the hub"


def test_a_cancelled_call_does_not_leave_an_earlier_receipt_behind(by_id):
    receipt = {"capability_id": "sortes.draw@v1", "signature": "first-call"}
    agent = StubAgent({"ok": True, "output": {"beta": "ab12"}, "receipt": receipt}, delay=0.0)
    tool = make_tool(by_id["sortes.draw@v1"], agent)
    assert asyncio.run(_call(tool, {"alpha": "hi"})).receipt == receipt

    agent._delay = 0.35

    async def main() -> None:
        token = CancellationToken()
        task = asyncio.ensure_future(tool.run(args_for(tool, alpha="hi"), token))
        await asyncio.sleep(0.05)
        token.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert tool.last_result is None, "the first call's receipt was left standing in for the second"


# ── refusals are results, failures are errors ───────────────────────────────


def test_a_refusal_reaches_the_model_as_readable_text(by_id):
    agent = StubAgent({"ok": False, "error": "'alpha' must be a non-empty string"})
    tool = make_tool(by_id["sortes.draw@v1"], agent)

    result = asyncio.run(_call(tool, {"alpha": "hi"}))

    assert result.ok is False
    assert result.budget_exceeded is False and result.cancelled is False
    text = tool.return_value_as_string(result)
    assert text == "sortes.draw@v1 refused this input: 'alpha' must be a non-empty string"
    # No receipt, so nothing was metered and the reservation came back.
    assert tool.hub.spent_usd == 0.0


def test_a_refusal_is_not_an_error_result_in_the_workbench(by_id):
    """A refusal must not set ``is_error``: the model should treat it as an answer to fix, and
    an error result invites a retry loop on the same arguments."""
    agent = StubAgent({"ok": False, "error": "num_bytes must be <= 64"})
    tool = make_tool(by_id["sortes.draw@v1"], agent)

    result = asyncio.run(StaticWorkbench([tool]).call_tool(tool.name, {"alpha": "hi"}))

    assert result.is_error is False
    assert "refused this input: num_bytes must be <= 64" in result.result[0].content


def test_budget_exhaustion_is_returned_not_raised(by_id):
    """A $0.005 ceiling cannot afford a $0.006 call."""
    agent = StubAgent()
    tool = make_tool(by_id["sortes.draw@v1"], agent, budget_usd=0.005)

    result = asyncio.run(_call(tool, {"alpha": "hi"}))

    assert result.ok is False
    assert result.budget_exceeded is True
    assert agent.calls == []
    text = tool.return_value_as_string(result)
    assert "was not called" in text and "budget" in text
    # Precise detection without matching on prose.
    assert tool.last_result is not None and tool.last_result.budget_exceeded


def test_a_concurrent_fan_out_cannot_spend_past_the_ceiling(by_id):
    """20 calls in flight at once against a ceiling that affords exactly 5.

    This is the shape AutoGen really produces — `AssistantAgent` gathers every tool call in a
    turn — and it is the shape a spend counter fails under. The invokes run in worker threads,
    so the reservation is what has to be atomic: an unlocked check would let all 20 read the
    same remaining balance and every one of them pass. Exactly 5 may reach the hub.
    """
    agent = StubAgent(delay=0.02)
    hub = make_hub(agent, budget_usd=0.030)  # 5 x $0.006
    tools = [AIMarketTool(by_id["sortes.draw@v1"], hub) for _ in range(20)]

    async def main() -> list[CapabilityResult]:
        return await asyncio.gather(*(_call(t, {"alpha": "hi"}) for t in tools))

    results = asyncio.run(main())

    assert sum(1 for r in results if r.ok) == 5
    assert sum(1 for r in results if r.budget_exceeded) == 15
    assert len(agent.calls) == 5, "more calls were billed than the ceiling allowed"
    assert hub.spent_usd == pytest.approx(0.030)


def test_concurrent_refusals_do_not_consume_the_budget(by_id):
    """A refusal with no receipt was never metered, so ten of them must leave the budget whole.

    Worth its own test at concurrency: the reservation is taken before the call and released
    after, so a lost release under interleaving would silently retire budget an operator never
    spent — an agent that stops working with money still on the ceiling.
    """
    agent = StubAgent({"ok": False, "error": "'alpha' must be a non-empty string"}, delay=0.01)
    hub = make_hub(agent, budget_usd=0.10)
    tools = [AIMarketTool(by_id["sortes.draw@v1"], hub) for _ in range(10)]

    async def main() -> list[CapabilityResult]:
        return await asyncio.gather(*(_call(t, {"alpha": "hi"}) for t in tools))

    assert all(r.ok is False for r in asyncio.run(main()))
    assert len(agent.calls) == 10
    # approx, not ==: reserving and releasing a float ten times leaves ~1e-18 of IEEE-754
    # residue behind. Irrelevant against a $0.006 price, but it is not exactly zero.
    assert hub.spent_usd == pytest.approx(0.0, abs=1e-12)
    assert hub.remaining_usd == pytest.approx(0.10)


def test_a_refusal_the_hub_metered_still_shows_its_price(by_id):
    """A refusal that came back with a receipt WAS billed, and must not read as free.

    The core keeps the reservation in that case; the bridge has to carry the price through to
    the result rather than reporting the refusal as a costless non-event, or a loop of billed
    refusals spends without anything showing it.
    """
    agent = StubAgent({"ok": False, "error": "out of range", "receipt": {"signature": "s"}})
    tool = make_tool(by_id["sortes.draw@v1"], agent)

    result = asyncio.run(_call(tool, {"alpha": "hi"}))

    assert result.ok is False
    assert result.price_usd == pytest.approx(0.006)
    assert tool.hub.spent_usd == pytest.approx(0.006)
    assert "refused this input: out of range" in tool.return_value_as_string(result)


def test_arguments_the_model_got_wrong_never_reach_the_hub(by_id):
    """A validation failure has to be free, and it is the commonest failure of all.

    ``run_json`` validates before `run` is entered, so a missing or mistyped argument raises
    out of the tool and the workbench reports ``is_error=True`` — nothing dispatched, nothing
    reserved. The message names the capability's own ``lambda``, which is what lets the model
    fix the call rather than repeat it.
    """
    agent = StubAgent()
    tool = make_tool(by_id["fourier.verify@v1"], agent)

    result = asyncio.run(StaticWorkbench([tool]).call_tool(
        tool.name, {"edges": [["a", "b"]], "vector": [1.0]}))

    assert result.is_error is True
    assert "lambda" in result.result[0].content
    assert agent.calls == []
    assert tool.hub.spent_usd == 0.0


def test_transport_failure_raises_and_surfaces_as_an_error_result(by_id):
    """The other half of the split: nothing was called, so the model must not be handed a
    result it can narrate over. AutoGen carries this on ``ToolResult.is_error``."""
    agent = StubAgent(raises=RuntimeError("connection refused"))
    tool = make_tool(by_id["sortes.draw@v1"], agent)

    with pytest.raises(HubUnavailable):
        asyncio.run(_call(tool, {"alpha": "hi"}))

    result = asyncio.run(StaticWorkbench([tool]).call_tool(tool.name, {"alpha": "hi"}))
    assert result.is_error is True
    assert "connection refused" in result.result[0].content
    # Refunded both times — a call that never happened is not billed.
    assert tool.hub.spent_usd == 0.0


# ── what the call actually sends and returns ────────────────────────────────


def test_absent_optional_arguments_are_dropped_not_sent_as_null(by_id):
    """No schema in the catalogue accepts null, so a `None` key would be a rejected argument
    on a call that was already paid for.

    ``turing.bluenoise@v1`` has all three cases at once: ``count`` required, ``candidates``
    optional with a schema default of 10, and ``seed`` optional with no default — which is the
    one that becomes ``None`` and must disappear rather than travel as null.
    """
    agent = StubAgent()
    tool = make_tool(by_id["turing.bluenoise@v1"], agent)
    assert set(tool.args_type().model_fields) == {"count", "candidates", "seed"}

    asyncio.run(_call(tool, {"count": 16}))

    sent = agent.calls[0]["input_payload"]
    assert None not in sent.values()
    assert "seed" not in sent, "a default-less optional argument was sent as null"
    # The schema default is still sent: the capability declared it, so it is a real value.
    assert sent == {"count": 16, "candidates": 10}


def test_the_invoke_carries_the_federation_route(by_id):
    agent = StubAgent()
    tool = make_tool(by_id["sortes.draw@v1"], agent)

    asyncio.run(_call(tool, {"alpha": "hi", "num_bytes": 8}))

    call = agent.calls[0]
    assert call["capability_id"] == "sortes.draw@v1"
    assert call["product_id"] == "prod-sortes"
    assert call["source_hub"] == "https://oracles.modelmarket.dev/family"
    assert call["input_payload"] == {"alpha": "hi", "num_bytes": 8}


def test_structured_output_is_json_and_plain_text_is_not_requoted(by_id):
    agent = StubAgent({"ok": True, "output": {"beta": "ab12", "n": 3}})
    tool = make_tool(by_id["sortes.draw@v1"], agent)
    text = tool.return_value_as_string(asyncio.run(_call(tool, {"alpha": "hi"})))
    assert json.loads(text) == {"beta": "ab12", "n": 3}

    agent = StubAgent({"ok": True, "output": "all clear"})
    tool = make_tool(by_id["skopos.briefing@v1"], agent)
    assert tool.return_value_as_string(asyncio.run(_call(tool, {}))) == "all clear"


@pytest.mark.parametrize(
    "output, expected",
    [
        ({"beta": "ab12"}, '{"beta": "ab12"}'),
        ([1, 2, 3], "[1, 2, 3]"),
        ([[0.0, 1.0], 5.0], "[[0.0, 1.0], 5.0]"),
        (42, "42"),
        (3.5, "3.5"),
        (True, "true"),
        (None, "null"),
        ("all clear", "all clear"),
    ],
)
def test_every_output_shape_a_capability_returns_serializes_for_the_model(by_id, output, expected):
    """`return_type` is a pydantic model, but `output` inside it is `Any` — so the shapes have
    to be checked, not assumed.

    Capabilities in this catalogue answer with objects, arrays, arrays-of-arrays and bare
    scalars. Anything falling through to ``str()`` would hand the model Python repr —
    ``True`` rather than ``true``, single quotes in place of JSON — which is the failure
    `CapabilityResult` exists to avoid. A string stays unquoted, since escaping a sentence only
    gives the model more to read past.
    """
    agent = StubAgent({"ok": True, "output": output})
    tool = make_tool(by_id["sortes.draw@v1"], agent)

    result = asyncio.run(_call(tool, {"alpha": "hi"}))

    assert tool.return_value_as_string(result) == expected
    # And the whole result still round-trips, which is what AutoGen state and tracing need.
    assert json.loads(result.model_dump_json())["output"] == output
    # Through the workbench too: a non-dict output is not an error result.
    wb = asyncio.run(StaticWorkbench([tool]).call_tool(tool.name, {"alpha": "hi"}))
    assert wb.is_error is False
    assert wb.result[0].content == expected


@pytest.mark.parametrize("cid", ["sortes.draw@v1", "platon.random@v1"])
def test_fresh_randomness_is_never_answered_from_a_cache(by_id, cid):
    """Identical arguments must produce a new paid draw every time.

    A memoised tool result would sell the same random number twice, which for a randomness
    oracle is not a performance optimisation but a broken product. Nothing in 0.7.5 caches tool
    results — ``StaticWorkbench.call_tool`` goes straight to ``run_json``, and ``BaseTool``
    holds no result state (``state_type()`` is None, ``save_state_json()`` is ``{}``) — and this
    pins that, plus the fact that neither `CapabilityResult` nor `last_result` is ever consulted
    as a cache. autogen_ext is not installed in this venv, so nothing is claimed here about
    ``ChatCompletionCache``; what IS pinned, in the AssistantAgent test below, is the path this
    bridge is actually driven through.
    """
    draws = iter(range(1, 99))

    class Draws(StubAgent):
        def invoke_single(self, **kw):
            super().invoke_single(**kw)
            return {"ok": True, "output": {"beta": f"draw-{next(draws)}"}}

    agent = Draws()
    tool = make_tool(by_id[cid], agent)
    workbench = StaticWorkbench([tool])
    arguments = _minimal_args(tool)

    async def main() -> list[str]:
        return [(await workbench.call_tool(tool.name, dict(arguments))).result[0].content
                for _ in range(3)]

    texts = asyncio.run(main())

    assert texts == ['{"beta": "draw-1"}', '{"beta": "draw-2"}', '{"beta": "draw-3"}']
    assert len(agent.calls) == 3, "a repeat call was answered without reaching the hub"
    assert tool.hub.spent_usd == pytest.approx(3 * by_id[cid].price_usd)
    assert tool.state_type() is None and asyncio.run(tool.save_state_json()) == {}


def test_the_receipt_stays_out_of_the_models_context(by_id):
    """Provenance must not cost tokens on every call, but must stay reachable."""
    receipt = {"capability_id": "sortes.draw@v1", "signature": "c2lnbmF0dXJl", "amount_usd": 0.006}
    agent = StubAgent({"ok": True, "output": {"beta": "ab12"}, "receipt": receipt})
    tool = make_tool(by_id["sortes.draw@v1"], agent)

    result = asyncio.run(_call(tool, {"alpha": "hi"}))
    text = tool.return_value_as_string(result)

    assert "c2lnbmF0dXJl" not in text and "signature" not in text
    assert text == '{"beta": "ab12"}'
    # Reachable on the result, on the tool, and on the shared client.
    assert result.receipt == receipt
    assert tool.last_result is result
    assert tool.hub.last_receipt == receipt


def test_receipt_verification_outcome_is_surfaced_not_reimplemented(by_id):
    """The core verifies against the ORIGIN key; the bridge only reports what it concluded,
    including the three-way distinction (ok / invalid / not checked)."""
    receipt = {"capability_id": "sortes.draw@v1", "signature": "x"}
    checks = {
        "ok": ReceiptCheck(True, "ok", key="YkAO...", origin="https://oracles.modelmarket.dev/family"),
        "bad": ReceiptCheck(False, "invalid-signature"),
        "unknown": ReceiptCheck(None, "no signing key published by origin"),
    }
    for label, check in checks.items():
        agent = StubAgent({"ok": True, "output": {"beta": "ab12"}, "receipt": receipt})
        hub = HubClient(HUB, budget_usd=1.0, verify_receipts=True, agent=agent)

        class Resolver:
            def check(self, _receipt: Any, *, source_hub: str = "", expect: Any = None) -> ReceiptCheck:
                assert source_hub == "https://oracles.modelmarket.dev/family"
                return check

            def close(self) -> None:
                return None

        hub._keys = Resolver()
        result = asyncio.run(_call(AIMarketTool(by_id["sortes.draw@v1"], hub), {"alpha": "hi"}))

        assert result.receipt_verified is check.verified, label
        assert result.receipt_verify_reason == check.reason, label


# ── the builder ─────────────────────────────────────────────────────────────


def test_aimarket_tools_builds_the_whole_catalogue():
    with _mock_client() as http:
        tools = aimarket_tools(HUB, catalog_client=http, agent=StubAgent(), budget_usd=2.0)

    assert len(tools) == 47
    assert all(isinstance(t, AIMarketTool) for t in tools)
    # One shared client, so the ceiling applies across the set rather than per tool.
    assert len({id(t.hub) for t in tools}) == 1
    assert tools[0].hub.budget_usd == 2.0


def test_aimarket_tools_filters_before_the_agent_ever_sees_a_tool():
    with _mock_client() as http:
        cheap = aimarket_tools(HUB, catalog_client=http, agent=StubAgent(), max_price_usd=0.01)
    assert cheap and all(t.capability.price_usd <= 0.01 for t in cheap)
    assert len(cheap) < 47

    with _mock_client() as http:
        limited = aimarket_tools(HUB, catalog_client=http, agent=StubAgent(), limit=5)
    assert len(limited) == 5


@pytest.mark.parametrize(
    "handler",
    [
        pytest.param(lambda request: httpx.Response(503, text="upstream down"), id="503"),
        pytest.param(lambda request: httpx.Response(200, json={"ok": True}), id="not-a-v2-hub"),
        pytest.param(lambda request: (_ for _ in ()).throw(httpx.ConnectError("no route")),
                     id="unreachable"),
    ],
)
def test_a_hub_that_is_down_at_build_time_raises_instead_of_yielding_zero_tools(handler):
    """The opposite failure from `free_only`, and the more dangerous one.

    An agent built with an empty tool list starts up believing it has no capabilities and then
    answers from its own weights instead of buying a verified answer — silently, and forever.
    `fetch_catalog` raises `CatalogError` rather than returning ``[]`` for exactly this reason,
    so the bridge must let it through with the URL attached, not soften it.
    """
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(CatalogError) as caught:
            aimarket_tools(HUB, catalog_client=http, agent=StubAgent())

    assert "/ai-market/v2/manifest" in str(caught.value)


def _dead_client() -> httpx.Client:
    """Fails on any request, so reaching an error at all proves nothing was fetched first."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"the catalogue was read before the budget check: {request.url}")

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("bad", [-1.0, -0.01])
def test_a_negative_budget_is_refused_before_the_catalogue_is_read(bad):
    """A negative ceiling means nothing, in either direction.

    0 is NOT in this list any more. It used to be, and for a real reason: `_reserve` tested
    ``if self.budget_usd and ...``, so a falsy budget skipped the check and every paid call went
    through while `remaining_usd` reported $0.00 — 0 buying unlimited spending, the exact inverse
    of what the value looks like it asks for. All three adapters carried that guard, which is how
    it became clear the defect was in the core. Fixed there, so 0 now means spend nothing and is
    passed straight down.

    Raised, not warned: a build-time constant is a configuration mistake, and the whole point of
    filtering at build time is that a tool the operator cannot afford is never handed over. The
    dead client proves the guard fires before any network work.
    """
    with _dead_client() as http:
        with pytest.raises(ValueError, match="must not be negative"):
            aimarket_tools(HUB, catalog_client=http, agent=StubAgent(), budget_usd=bad)


def test_the_core_now_enforces_a_zero_budget(by_id):
    """The measurement this rested on has been overturned — deliberately, and by the note it
    carried: "if the core ever starts enforcing it, this fails and the guard can be
    reconsidered rather than cargo-culted." It does now.

    Pre-fix, five paid calls against budget_usd=0 all went through ($0.030 spent) because
    `_reserve` tested `if self.budget_usd and ...`. Zero now means spend nothing, so the guard
    in aimarket_tools shrank to refusing only a negative value.
    """
    hub = make_hub(StubAgent(), budget_usd=0.0)
    tool = AIMarketTool(by_id["sortes.draw@v1"], hub)

    result = asyncio.run(_call(tool, {"alpha": "hi"}))
    assert result.ok is False, "a zero ceiling must refuse a paid call"
    assert hub.spent_usd == 0.0, "pre-fix five such calls spent $0.030"
    assert hub.remaining_usd == 0.0


def test_free_only_yields_nothing_and_says_why(caplog):
    """Nothing on the live hub is free. The empty build must be explained, because from the
    agent's side a zero-tool registry looks exactly like a broken hub."""
    with _mock_client() as http:
        tools = aimarket_tools(HUB, catalog_client=http, agent=StubAgent(), free_only=True)

    assert tools == []
    assert "no capability on this hub is free" in caplog.text


def test_shared_budget_is_enforced_across_different_tools():
    agent = StubAgent()
    with _mock_client() as http:
        tools = aimarket_tools(HUB, catalog_client=http, agent=agent, budget_usd=0.01)
    picked = [t for t in tools if t.capability.capability_id in
              ("sortes.draw@v1", "platon.random@v1", "chronos.eval@v1")]
    assert len(picked) == 3

    results = [asyncio.run(_call(t, _minimal_args(t))) for t in picked]
    spent = picked[0].hub.spent_usd
    assert spent <= 0.01
    assert any(r.budget_exceeded for r in results), "the ceiling was never reached"


def _minimal_args(tool: AIMarketTool) -> dict[str, Any]:
    """Just enough to satisfy the required fields of any capability in this test."""
    values: dict[str, Any] = {}
    model = tool.args_type()
    for name, field in model.model_fields.items():
        if field.is_required():
            values[name] = "x" if field.annotation is str else 1
    return values


# ── AssistantAgent integration, with no model ───────────────────────────────


class ScriptedModelClient(ChatCompletionClient):
    """A ChatCompletionClient that replays canned ``CreateResult``s.

    Introspected requirements for 0.7.5, not guessed: ``ChatCompletionClient`` declares nine
    abstract members — ``create``, ``create_stream``, ``close``, ``actual_usage``,
    ``total_usage``, ``count_tokens``, ``remaining_tokens``, ``capabilities`` and
    ``model_info`` — and all nine must exist or instantiation fails. ``AssistantAgent.__init__``
    reads only one thing off the client, ``model_info["function_calling"]``, and refuses tools
    when it is False; nothing else is touched until a turn runs.
    """

    def __init__(self, script: list[CreateResult]):
        self._script = list(script)
        self._calls = 0
        self.tool_schemas: list[list[dict[str, Any]]] = []

    async def create(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                     extra_create_args={}, cancellation_token=None, **kw):
        self.tool_schemas.append([t.schema if not isinstance(t, dict) else t for t in tools])
        result = self._script[min(self._calls, len(self._script) - 1)]
        self._calls += 1
        return result

    async def create_stream(self, messages, **kw):  # type: ignore[override]
        yield await self.create(messages, **kw)

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def total_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def count_tokens(self, messages, *, tools=[]) -> int:
        return 0

    def remaining_tokens(self, messages, *, tools=[]) -> int:
        return 100_000

    @property
    def capabilities(self) -> ModelInfo:
        return self.model_info

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(vision=False, function_calling=True, json_output=False,
                         family="unknown", structured_output=False)


def _tool_call_script(name: str, arguments: dict[str, Any]) -> list[CreateResult]:
    return [CreateResult(
        finish_reason="function_calls",
        content=[FunctionCall(id="call-1", name=name, arguments=json.dumps(arguments))],
        usage=RequestUsage(prompt_tokens=1, completion_tokens=1),
        cached=False,
    )]


def test_assistant_agent_accepts_the_tools_and_calls_one(by_id):
    """A full AssistantAgent turn with no model involved.

    Proves the tools are usable by the real agent class, and shows exactly what lands in the
    conversation: the capability's output, no receipt, ``is_error=False``.
    """
    from autogen_agentchat.agents import AssistantAgent

    agent = StubAgent({"ok": True, "output": {"beta": "ab12"}, "receipt": {"signature": "zzz"}})
    tool = make_tool(by_id["sortes.draw@v1"], agent)
    client = ScriptedModelClient(_tool_call_script("sortes_draw_v1", {"alpha": "hello"}))

    assistant = AssistantAgent("buyer", model_client=client, tools=[tool])
    result = asyncio.run(assistant.run(task="draw some randomness"))

    kinds = [type(m).__name__ for m in result.messages]
    assert "ToolCallExecutionEvent" in kinds
    execution = result.messages[kinds.index("ToolCallExecutionEvent")]
    (executed,) = execution.content
    assert executed.name == "sortes_draw_v1"
    assert executed.is_error is False
    assert executed.content == '{"beta": "ab12"}'
    assert "zzz" not in executed.content

    # The hub really was invoked, once, through the agent.
    assert len(agent.calls) == 1
    assert agent.calls[0]["input_payload"] == {"alpha": "hello", "num_bytes": 32}

    # And the schema handed to the model client is the one asserted above.
    (offered,) = client.tool_schemas
    assert [s["name"] for s in offered] == ["sortes_draw_v1"]
    assert offered[0]["parameters"]["required"] == ["alpha"]


def test_assistant_agent_rejects_a_client_without_function_calling(by_id):
    """The one client property that gates tool registration, pinned so the stub's shape is
    documented by a test rather than by a comment."""
    from autogen_agentchat.agents import AssistantAgent

    class NoTools(ScriptedModelClient):
        @property
        def model_info(self) -> ModelInfo:
            return ModelInfo(vision=False, function_calling=False, json_output=False,
                             family="unknown", structured_output=False)

    tool = make_tool(by_id["sortes.draw@v1"], StubAgent())
    with pytest.raises(ValueError, match="does not support function calling"):
        AssistantAgent("buyer", model_client=NoTools([]), tools=[tool])


def _executed(result: Any) -> list[Any]:
    """The FunctionExecutionResults from a finished `AssistantAgent.run`."""
    kinds = [type(m).__name__ for m in result.messages]
    assert "ToolCallExecutionEvent" in kinds, kinds
    return list(result.messages[kinds.index("ToolCallExecutionEvent")].content)


def test_the_keyword_named_arguments_survive_a_whole_agent_turn(by_id):
    """``lambda`` and a nested ``from``, end to end through the real agent.

    The last and only link that matters: the model emits a JSON *string*, `AssistantAgent`
    ``json.loads`` it, the workbench forwards the raw mapping, `run_json` validates it, and
    `_payload` dumps it. Five hand-offs, each of which could substitute the sanitised spelling.
    What the stub receives is what the capability would receive, and for these two that is the
    difference between an answer and a paid refusal.
    """
    from autogen_agentchat.agents import AssistantAgent

    fourier = StubAgent({"ok": True, "output": {"certified": True}})
    fermat = StubAgent({"ok": True, "output": {"path": ["a", "b"]}})
    tools = [make_tool(by_id["fourier.verify@v1"], fourier),
             make_tool(by_id["fermat.route@v1"], fermat)]
    client = ScriptedModelClient([CreateResult(
        finish_reason="function_calls",
        content=[
            FunctionCall(id="c1", name="fourier_verify_v1", arguments=json.dumps(
                {"edges": [["a", "b"]], "lambda": 0.5, "vector": [1.0, -1.0]})),
            FunctionCall(id="c2", name="fermat_route_v1", arguments=json.dumps(
                {"edges": [{"from": "a", "to": "b", "cost": 1}], "start": "a", "goal": "b"})),
        ],
        usage=RequestUsage(prompt_tokens=1, completion_tokens=1), cached=False)])

    assistant = AssistantAgent("buyer", model_client=client, tools=tools)
    executed = _executed(asyncio.run(assistant.run(task="verify and route")))

    assert [e.is_error for e in executed] == [False, False], [e.content for e in executed]
    assert fourier.calls[0]["input_payload"]["lambda"] == 0.5
    assert "lambda_" not in fourier.calls[0]["input_payload"]
    assert fermat.calls[0]["input_payload"]["edges"] == [{"from": "a", "to": "b", "cost": 1.0}]

    # And the schema the model was handed advertises those same names, with the polymorphic
    # branches intact — this is the only assertion covering what reaches a real model client.
    (offered,) = client.tool_schemas
    fourier_schema, fermat_schema = offered
    assert set(fourier_schema["parameters"]["required"]) == {"edges", "lambda", "vector"}
    edge = next(b for b in fermat_schema["parameters"]["properties"]["edges"]["items"]["anyOf"]
                if b.get("properties"))
    assert "from" in edge["properties"] and "from_" not in edge["properties"]
    assert "$ref" not in json.dumps(offered) and "$defs" not in json.dumps(offered)


def test_the_polymorphic_schema_reaches_the_model_client_intact(by_id):
    """``kantor.transport@v1``'s ``oneOf`` as a real model client receives it.

    `tool.schema` is asserted elsewhere; this pins the hop after it. `AssistantAgent` wraps
    tools in a `StaticStreamWorkbench` and offers `list_tools()` output to the client, so a
    union flattened there would be invisible to every other test in this file — and the model
    would be told a shape the capability refuses, after being billed.
    """
    from autogen_agentchat.agents import AssistantAgent

    agent = StubAgent({"ok": True, "output": {"plan": []}})
    tool = make_tool(by_id["kantor.transport@v1"], agent)
    client = ScriptedModelClient(_tool_call_script("kantor_transport_v1", {
        "a": [1.0, 2.0], "b": [1.0, 2.0], "source_points": [[0.0, 1.0], 5.0]}))

    assistant = AssistantAgent("buyer", model_client=client, tools=[tool])
    assert _executed(asyncio.run(assistant.run(task="transport")))[0].is_error is False

    (offered,) = client.tool_schemas
    array = next(b for b in offered[0]["parameters"]["properties"]["source_points"]["anyOf"]
                 if b.get("type") == "array")
    assert array["items"]["anyOf"] == [
        {"items": {"type": "number"}, "type": "array"},
        {"type": "number"},
    ]
    # Both branches in one argument, forwarded unchanged: the union is real, not decoration.
    assert agent.calls[0]["input_payload"]["source_points"] == [[0.0, 1.0], 5.0]


def test_a_turn_full_of_tool_calls_bills_every_one_and_still_overlaps(by_id):
    """The concurrency AutoGen really produces, against the shared spend counter.

    `AssistantAgent._execute_tool_calls` gathers every call in the turn, so twelve paid invokes
    are in flight at once on one `HubClient`. Two things have to hold at the same time: every
    increment is counted (a dropped one is money spent off the books), and they genuinely
    overlap (12 x 50ms serialised would be 0.6s, and a blocked loop is the failure the offload
    exists to prevent).
    """
    from autogen_agentchat.agents import AssistantAgent

    agent = StubAgent(delay=0.05)
    tool = make_tool(by_id["sortes.draw@v1"], agent, budget_usd=1.0)
    client = ScriptedModelClient([CreateResult(
        finish_reason="function_calls",
        content=[FunctionCall(id=f"c{i}", name="sortes_draw_v1",
                              arguments=json.dumps({"alpha": f"a{i}"})) for i in range(12)],
        usage=RequestUsage(prompt_tokens=1, completion_tokens=1), cached=False)])

    assistant = AssistantAgent("buyer", model_client=client, tools=[tool])
    started = time.monotonic()
    executed = _executed(asyncio.run(assistant.run(task="draw a lot")))
    elapsed = time.monotonic() - started

    assert len(executed) == 12 and all(e.is_error is False for e in executed)
    assert len(agent.calls) == 12
    assert tool.hub.spent_usd == pytest.approx(12 * 0.006)
    assert elapsed < 0.4, f"the turn serialised: {elapsed:.2f}s for 12 x 0.05s"


def test_identical_draws_in_one_turn_are_each_paid_for_separately(by_id):
    """Three identical `sortes.draw@v1` calls in a single turn must be three real draws.

    The argument-keyed cache a tool framework might reasonably add is, for a randomness oracle,
    a way to sell one draw three times and hand the second buyer a number the first already
    has. `test_fresh_randomness_is_never_answered_from_a_cache` pins the workbench in isolation;
    this pins the whole agent, with identical arguments in one gather, which is the case any
    memoisation would collapse.
    """
    from autogen_agentchat.agents import AssistantAgent

    draws = iter(range(1, 99))

    class Draws(StubAgent):
        def invoke_single(self, **kw: Any) -> Any:
            super().invoke_single(**kw)
            return {"ok": True, "output": {"beta": f"draw-{next(draws)}"}}

    agent = Draws()
    tool = make_tool(by_id["sortes.draw@v1"], agent)
    arguments = json.dumps({"alpha": "same", "num_bytes": 8})
    client = ScriptedModelClient([CreateResult(
        finish_reason="function_calls",
        content=[FunctionCall(id=f"c{i}", name="sortes_draw_v1", arguments=arguments)
                 for i in range(3)],
        usage=RequestUsage(prompt_tokens=1, completion_tokens=1), cached=False)])

    assistant = AssistantAgent("buyer", model_client=client, tools=[tool])
    executed = _executed(asyncio.run(assistant.run(task="draw three times")))

    assert sorted(e.content for e in executed) == [
        '{"beta": "draw-1"}', '{"beta": "draw-2"}', '{"beta": "draw-3"}']
    assert len(agent.calls) == 3, "a repeat call was answered without reaching the hub"
    assert tool.hub.spent_usd == pytest.approx(3 * 0.006)


@pytest.mark.parametrize(
    "body, budget, is_error, fragment",
    [
        pytest.param({"ok": False, "error": "'alpha' must be a non-empty string"}, 1.0, False,
                     "refused this input", id="capability-refusal"),
        pytest.param({"ok": True, "output": 1}, 0.001, False,
                     "was not called", id="budget-exhausted"),
        pytest.param({"safety_blocked": True, "reason": "prompt injection"}, 1.0, False,
                     "safety gate", id="safety-blocked"),
    ],
)
def test_a_refusal_never_aborts_the_agent_turn(by_id, body, budget, is_error, fragment):
    """Every "no" the model could act on has to arrive as text inside a completed turn.

    Driven through `AssistantAgent` rather than the workbench, because that is where an
    exception would do its damage: raising out of a tool call ends the run, and the model never
    gets the sentence that would have let it fix the call or stop. ``is_error`` stays False for
    all three — an error result invites a retry on the same arguments, which for the budget case
    is a retry that can never succeed.
    """
    from autogen_agentchat.agents import AssistantAgent

    agent = StubAgent(body)
    tool = make_tool(by_id["sortes.draw@v1"], agent, budget_usd=budget)
    client = ScriptedModelClient(_tool_call_script("sortes_draw_v1", {"alpha": "hi"}))

    assistant = AssistantAgent("buyer", model_client=client, tools=[tool])
    result = asyncio.run(assistant.run(task="draw"))
    (executed,) = _executed(result)

    assert executed.is_error is is_error
    assert fragment in executed.content
    # The turn finished normally, so the conversation is still usable.
    assert type(result.messages[-1]).__name__ in ("ToolCallSummaryMessage", "TextMessage")
    # Nothing was billed on any of the three: refused unmetered, never dispatched, or blocked
    # by the hub before the provider ran.
    assert tool.hub.spent_usd == pytest.approx(0.0, abs=1e-12)


def test_a_hub_that_dies_mid_run_is_an_error_result_not_a_narratable_answer(by_id):
    """The other half of the split, through the agent.

    A transport failure must not become a normal tool result: the model would then narrate an
    answer over a call that never happened. AutoGen carries it on ``is_error``, which is the
    channel the other two bridges do not have.
    """
    from autogen_agentchat.agents import AssistantAgent

    agent = StubAgent(raises=RuntimeError("connection refused"))
    tool = make_tool(by_id["sortes.draw@v1"], agent)
    client = ScriptedModelClient(_tool_call_script("sortes_draw_v1", {"alpha": "hi"}))

    assistant = AssistantAgent("buyer", model_client=client, tools=[tool])
    (executed,) = _executed(asyncio.run(assistant.run(task="draw")))

    assert executed.is_error is True
    assert "connection refused" in executed.content
    assert tool.hub.spent_usd == 0.0


def test_all_47_tools_can_be_registered_on_one_agent(catalog):
    """Name uniqueness is enforced by AssistantAgent itself, so this is the real check that
    `tool_name_for` produced a usable set for the whole live catalogue."""
    from autogen_agentchat.agents import AssistantAgent

    hub = make_hub(StubAgent())
    tools = [AIMarketTool(cap, hub) for cap in catalog]
    assistant = AssistantAgent("buyer", model_client=ScriptedModelClient([]), tools=tools)
    assert len(assistant._tools) == 47



def test_one_unbuildable_capability_does_not_cost_the_whole_registry():
    """A property named `model_dump` raises ValueError out of pydantic — about neither depth
    nor size, so the catalogue clamp cannot catch it. Before the per-capability try/except,
    that single entry aborted the whole build and the agent started with ZERO tools.
    """
    import json
    import logging
    import pathlib as _p

    import httpx

    live = json.loads((_p.Path(__file__).parent / "live_manifest.json").read_text())
    hostile = {
        "capability_id": "evil.name@v1", "product_id": "prod-evil", "name": "evil",
        "description": "d", "output_schema": {}, "price_per_call_usd": 0.001,
        "source_hub": "https://evil.test/x",
        "input_schema": {"type": "object", "properties": {"model_dump": {"type": "string"}}},
    }
    payload = {**live, "tools": live["tools"] + [hostile]}
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(
        200, json=payload if r.url.path.endswith("/manifest") else {"matches": []})))

    logging.disable(logging.CRITICAL)
    try:
        tools = aimarket_tools("https://hub.test", catalog_client=client, agent=object())
    finally:
        logging.disable(logging.NOTSET)

    assert len(tools) == len(live["tools"]), (
        f"expected the {len(live['tools'])} innocent capabilities, got {len(tools)}"
    )
    assert "evil" not in {getattr(t, "name", "") for t in tools}


def test_run_with_the_args_CLASS_says_what_to_do_instead(by_id):
    """`tool.args_type()` returns the CLASS — in autogen-core it is a method, not a
    constructor — so passing its result to run() is an easy mistake. It used to surface as
    `TypeError: BaseModel.model_dump() missing 1 required positional argument: 'self'`
    raised deep inside _payload, pointing at the wrong place entirely. Found by driving the
    adapter against the live hub by hand, which is exactly who hits it.
    """
    import asyncio

    import pytest as _pytest

    tool = AIMarketTool(by_id["platon.state@v1"], make_hub(StubAgent()))
    with _pytest.raises(TypeError) as caught:
        asyncio.run(tool.run(tool.args_type(), None))
    message = str(caught.value)
    assert "takes an instance" in message
    assert "run_json" in message, "the message must name the entry point autogen uses"
    assert tool.name in message
