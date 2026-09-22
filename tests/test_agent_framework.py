"""Tests for the Microsoft Agent Framework bridge, against agent-framework-core 1.15.0.

No network. The catalogue is served from ``live_manifest.json`` — 47 real capabilities with
their real schemas — through an ``httpx.MockTransport``, and the invoke path is fed a stub
agent injected via ``HubClient(agent=...)``.

Tests use ``asyncio.run`` inside synchronous test functions rather than pytest-asyncio, for the
same reason as the AutoGen suite: the bridge's async surface is one coroutine, and depending on
no plugin mode keeps this file runnable in any of the framework venvs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

# Skipped rather than failed where the framework is absent: agent-framework and crewai do not
# agree on a pydantic version, so — like the other three adapters — this one is developed and
# run in its own virtualenv, and the suite has to stay collectable in the others.
pytest.importorskip("agent_framework", reason="agent-framework-core is not installed")

from agent_framework import FunctionTool  # noqa: E402

from aimarket_bridges.agent_framework import (  # noqa: E402
    AIMarketTool,
    CapabilityResult,
    aimarket_tools,
)
from aimarket_bridges.catalog import Capability, fetch_catalog  # noqa: E402
from aimarket_bridges.client import HubClient, HubUnavailable  # noqa: E402
from aimarket_bridges.receipts import ReceiptCheck  # noqa: E402

MANIFEST = json.loads((Path(__file__).parent / "live_manifest.json").read_text())
HUB = "https://modelmarket.dev"

#: A complete, valid argument set for `fourier.verify@v1` — the capability whose `lambda`
#: property is the reason this adapter passes the hub's schema rather than a pydantic model.
FOURIER_ARGS: dict[str, Any] = {
    "edges": [["a", "b"]],
    "lambda": 1.0,
    "vector": [0.7071, -0.7071],
}


# ── fixtures ────────────────────────────────────────────────────────────────


def _mock_client() -> httpx.Client:
    """An httpx client that serves the captured manifest and nothing else."""

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
    return HubClient(HUB, budget_usd=budget_usd, verify_receipts=False, agent=agent)


def make_tool(cap: Capability, agent: StubAgent, **kw: Any) -> AIMarketTool:
    return AIMarketTool(cap, make_hub(agent, **kw))


def text_of(contents: Any) -> str:
    """The text the model would read from one tool result."""
    return "".join(getattr(c, "text", "") or "" for c in contents)


# ── the framework facts the design rests on ─────────────────────────────────


def test_a_pydantic_input_model_cannot_call_a_capability_named_lambda(by_id):
    """Why this adapter hands over the raw schema instead of `capability.args_model()`.

    With a pydantic `input_model`, `parameters()` advertises the by-alias schema while the
    arguments are dumped without aliases, so the framework's own validator then reports the
    advertised name as missing. `fourier.verify@v1` really does take a property called
    `lambda`, which pydantic cannot use as a field name.
    """
    cap = by_id["fourier.verify@v1"]
    assert "lambda" in cap.input_schema["properties"]

    broken = FunctionTool(
        name="fourier_verify_pydantic",
        description="d",
        func=lambda **kw: kw,
        input_model=cap.args_model(),
    )
    assert "lambda" in broken.parameters()["properties"]
    with pytest.raises(TypeError, match="Missing required argument"):
        asyncio.run(broken.invoke(arguments=FOURIER_ARGS))


def test_this_adapter_calls_the_same_capability_and_sends_the_real_name(by_id):
    agent = StubAgent()
    tool = make_tool(by_id["fourier.verify@v1"], agent)
    asyncio.run(tool.invoke(arguments=dict(FOURIER_ARGS)))
    (call,) = agent.calls
    assert call["input_payload"]["lambda"] == 1.0
    assert "lambda_" not in call["input_payload"]


def test_the_model_is_shown_the_hubs_own_schema(by_id):
    cap = by_id["fourier.verify@v1"]
    tool = make_tool(cap, StubAgent())
    advertised = tool.parameters()
    assert advertised["properties"] == cap.input_schema["properties"]
    assert advertised["required"] == cap.input_schema["required"]


def test_the_advertised_schema_is_a_copy(by_id):
    cap = by_id["fourier.verify@v1"]
    tool = make_tool(cap, StubAgent())
    cap.input_schema["properties"]["injected_later"] = {"type": "string"}
    assert "injected_later" not in tool.parameters()["properties"]
    del cap.input_schema["properties"]["injected_later"]


def test_a_capability_with_no_arguments_still_advertises_an_object(catalog):
    empty = [c for c in catalog if not (c.input_schema or {}).get("properties")]
    if not empty:
        pytest.skip("every live capability declares at least one property")
    tool = make_tool(empty[0], StubAgent())
    assert tool.parameters()["type"] == "object"
    assert tool.parameters()["properties"] == {}


# ── what never reaches the hub ──────────────────────────────────────────────


def test_a_missing_required_argument_is_refused_before_anything_is_billed(by_id):
    agent = StubAgent()
    tool = make_tool(by_id["fourier.verify@v1"], agent)
    with pytest.raises(TypeError, match="Missing required argument"):
        asyncio.run(tool.invoke(arguments={"edges": FOURIER_ARGS["edges"]}))
    assert agent.calls == []
    assert tool.hub.spent_usd == 0.0


def test_a_wrong_scalar_type_is_refused_before_anything_is_billed(by_id):
    agent = StubAgent()
    tool = make_tool(by_id["fourier.verify@v1"], agent)
    with pytest.raises(TypeError, match="Invalid type"):
        asyncio.run(tool.invoke(arguments={**FOURIER_ARGS, "lambda": "not a number"}))
    assert agent.calls == []


def test_an_invented_argument_is_dropped_rather_than_sent(by_id, caplog):
    """The framework forwards undeclared keys; the capability would bill and then refuse."""
    agent = StubAgent()
    tool = make_tool(by_id["fourier.verify@v1"], agent)
    with caplog.at_level(logging.WARNING, logger="aimarket_bridges.agent_framework"):
        asyncio.run(
            tool.invoke(arguments={**FOURIER_ARGS, "hallucinated": 7})
        )
    (call,) = agent.calls
    assert "hallucinated" not in call["input_payload"]
    assert call["input_payload"]["lambda"] == 1.0
    assert "hallucinated" in caplog.text


def test_the_framework_refuses_a_null_for_a_singly_typed_property(by_id):
    """Measured: it never reaches the function, so it can never reach the hub either."""
    agent = StubAgent()
    tool = make_tool(by_id["fourier.verify@v1"], agent)
    with pytest.raises(TypeError, match="expected number, got NoneType"):
        asyncio.run(tool.invoke(arguments={**FOURIER_ARGS, "tol": None}))
    assert agent.calls == []


def test_the_framework_also_refuses_a_null_for_a_union_typed_property(by_id):
    """`nonce` is typed `["string", "integer"]`; the checker walks the list too."""
    cap = by_id["percola.threshold@v1"]
    assert cap.input_schema["properties"]["nonce"]["type"] == ["string", "integer"]
    agent = StubAgent()
    tool = make_tool(cap, agent)
    with pytest.raises(TypeError, match="expected one of"):
        asyncio.run(tool.invoke(arguments={"edges": [["a", "b"]], "nonce": None}))
    assert agent.calls == []


def test_a_null_under_an_undeclared_type_is_dropped_rather_than_sent():
    """The one null the framework does not catch: a property with no `type` at all.

    No capability on this hub publishes one — the catalogue was checked — but a schema comes
    from whichever hub is federated in at runtime, so the case is reachable without any change
    to this bridge.
    """
    cap = Capability(
        tool_name="untyped_probe",
        capability_id="untyped.probe@v1",
        product_id="prod-untyped",
        description="a capability whose property declares no type",
        input_schema={"type": "object", "properties": {"anything": {}}, "required": []},
        output_schema={"type": "object"},
        price_usd=0.0,
    )
    agent = StubAgent()
    tool = AIMarketTool(cap, make_hub(agent))
    asyncio.run(tool.invoke(arguments={"anything": None}))
    (call,) = agent.calls
    assert call["input_payload"] == {}


def test_a_legitimately_nullable_map_value_is_not_pruned(by_id):
    """`fermat.verify@v1`'s `potentials` accepts null values — pruning is top-level only."""
    cap = by_id["fermat.verify@v1"]
    assert "potentials" in cap.input_schema["properties"]
    agent = StubAgent()
    tool = make_tool(cap, agent)
    potentials = {"a": 1.0, "b": None}
    asyncio.run(
        tool.invoke(
            arguments={
                "potentials": potentials,
                "edges": [["a", "b", 1.0]],
                "start": "a",
                "goal": "b",
                "path": ["a", "b"],
            }
        )
    )
    (call,) = agent.calls
    assert call["input_payload"]["potentials"] == potentials


# ── how the call is made ────────────────────────────────────────────────────


def test_the_wrapped_function_is_a_coroutine_so_the_framework_does_not_pick_the_executor(by_id):
    tool = make_tool(by_id["platon.random@v1"], StubAgent())
    assert asyncio.iscoroutinefunction(tool.func)


def test_invoke_runs_off_the_event_loop(by_id):
    """A blocking hub call must not stall the loop that is running the agent."""
    agent = StubAgent(delay=0.2)
    tool = make_tool(by_id["platon.random@v1"], agent)

    async def main() -> list[float]:
        ticks: list[float] = []

        async def heartbeat() -> None:
            start = time.perf_counter()
            while True:
                ticks.append(time.perf_counter() - start)
                await asyncio.sleep(0.01)

        beat = asyncio.create_task(heartbeat())
        await tool.invoke(arguments={})
        beat.cancel()
        return ticks

    ticks = asyncio.run(main())
    assert len(ticks) > 5, f"the loop was blocked: only {len(ticks)} ticks during a 0.2s call"
    assert agent.threads and agent.threads[0] != threading.get_ident()


def test_concurrent_calls_overlap(by_id):
    agent = StubAgent(delay=0.15)
    hub = make_hub(agent, budget_usd=100.0)
    tools = [AIMarketTool(by_id["platon.random@v1"], hub) for _ in range(4)]

    async def main() -> float:
        start = time.perf_counter()
        await asyncio.gather(*(t.invoke(arguments={}) for t in tools))
        return time.perf_counter() - start

    elapsed = asyncio.run(main())
    assert len(agent.calls) == 4
    assert elapsed < 0.45, f"four 0.15s calls took {elapsed:.2f}s — they were serialised"


def test_a_cancelled_call_does_not_leave_an_earlier_receipt_behind(by_id):
    agent = StubAgent(delay=0.2)
    tool = make_tool(by_id["platon.random@v1"], agent, budget_usd=100.0)

    async def main() -> None:
        await tool.invoke(arguments={})
        assert tool.last_result is not None
        call = asyncio.ensure_future(tool.invoke(arguments={}))
        await asyncio.sleep(0.02)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

    asyncio.run(main())
    assert tool.last_result is None


# ── what the model reads ────────────────────────────────────────────────────


def test_a_refusal_reaches_the_model_as_readable_text(by_id):
    agent = StubAgent({"ok": False, "error": "'count' must be an integer, got str"})
    tool = make_tool(by_id["platon.random@v1"], agent)
    contents = asyncio.run(tool.invoke(arguments={}))
    said = text_of(contents)
    assert "refused this input" in said
    assert "'count' must be an integer" in said
    assert '"ok"' not in said


def test_structured_output_is_json_and_plain_text_is_not_requoted(by_id):
    structured = StubAgent({"ok": True, "output": {"beta": "ab12", "n": 3}})
    tool = make_tool(by_id["platon.random@v1"], structured)
    assert json.loads(text_of(asyncio.run(tool.invoke(arguments={})))) == {"beta": "ab12", "n": 3}

    plain = StubAgent({"ok": True, "output": "just a sentence"})
    tool = make_tool(by_id["platon.random@v1"], plain)
    assert text_of(asyncio.run(tool.invoke(arguments={}))) == "just a sentence"


def test_the_receipt_stays_out_of_the_conversation_but_stays_reachable(by_id):
    receipt = {"receipt_id": "r-1", "signature": "sig"}
    agent = StubAgent({"ok": True, "output": {"beta": "ab12"}, "receipt": receipt})
    tool = make_tool(by_id["platon.random@v1"], agent)
    said = text_of(asyncio.run(tool.invoke(arguments={})))
    assert "r-1" not in said and "signature" not in said
    assert tool.last_result is not None and tool.last_result.receipt == receipt


def test_metadata_a_middleware_would_gate_on_is_on_the_tool(by_id):
    cap = by_id["platon.random@v1"]
    tool = make_tool(cap, StubAgent())
    assert tool.additional_properties == {
        "capability_id": cap.capability_id,
        "product_id": cap.product_id,
        "price_usd": cap.price_usd,
        "source_hub": cap.source_hub,
    }


# ── money and failure ───────────────────────────────────────────────────────


def test_budget_exhaustion_is_returned_not_raised(by_id):
    cap = by_id["platon.random@v1"]
    if cap.is_free:
        pytest.skip("a free capability cannot exhaust a budget")
    agent = StubAgent()
    tool = make_tool(cap, agent, budget_usd=0.0)
    contents = asyncio.run(tool.invoke(arguments={}))
    said = text_of(contents)
    assert "was not called" in said
    assert agent.calls == []
    assert tool.last_result is not None and tool.last_result.budget_exceeded is True


def test_a_concurrent_fan_out_cannot_spend_past_the_ceiling(by_id):
    cap = by_id["platon.random@v1"]
    if cap.is_free:
        pytest.skip("a free capability cannot exhaust a budget")
    ceiling = cap.price_usd * 3
    agent = StubAgent(delay=0.05)
    hub = make_hub(agent, budget_usd=ceiling)
    tools = [AIMarketTool(cap, hub) for _ in range(10)]

    async def main() -> None:
        await asyncio.gather(*(t.invoke(arguments={}) for t in tools))

    asyncio.run(main())
    assert len(agent.calls) <= 3
    assert hub.spent_usd <= ceiling + 1e-9


def test_a_transport_failure_propagates(by_id):
    agent = StubAgent(raises=HubUnavailable("connection refused"))
    tool = make_tool(by_id["platon.random@v1"], agent)
    with pytest.raises(HubUnavailable):
        asyncio.run(tool.invoke(arguments={}))


# ── the whole catalogue ─────────────────────────────────────────────────────


def test_every_live_capability_builds_a_valid_tool(catalog):
    hub = make_hub(StubAgent())
    for cap in catalog:
        tool = AIMarketTool(cap, hub)
        schema = tool.parameters()
        assert schema["type"] == "object"
        assert isinstance(schema.get("properties"), dict)
        assert tool.name and tool.description
    assert len(catalog) == 47


def test_the_price_is_the_first_thing_the_model_reads(by_id):
    tool = make_tool(by_id["platon.random@v1"], StubAgent())
    assert tool.description.startswith("[")
    plain = AIMarketTool(by_id["platon.random@v1"], make_hub(StubAgent()), include_price=False)
    assert not plain.description.startswith("[")


def test_aimarket_tools_builds_the_catalogue_with_one_shared_budget():
    with _mock_client() as http:
        tools = aimarket_tools(HUB, catalog_client=http, budget_usd=2.0, agent=StubAgent())
    assert len(tools) == 47
    assert len({id(t.hub) for t in tools}) == 1
    assert tools[0].hub.budget_usd == 2.0


def test_aimarket_tools_filters_before_the_agent_can_choose():
    with _mock_client() as http:
        cheap = aimarket_tools(HUB, catalog_client=http, max_price_usd=0.005, agent=StubAgent())
    assert cheap and all(t.capability.price_usd <= 0.005 for t in cheap)


def test_a_negative_budget_is_refused():
    with pytest.raises(ValueError, match="must not be negative"):
        aimarket_tools(HUB, budget_usd=-1.0)


def test_free_only_on_a_paid_hub_is_an_explained_empty_build(caplog):
    with _mock_client() as http, caplog.at_level(
        logging.WARNING, logger="aimarket_bridges.agent_framework"
    ):
        tools = aimarket_tools(HUB, catalog_client=http, free_only=True, agent=StubAgent())
    if tools:
        pytest.skip("this hub has free capabilities")
    assert "no capability on this hub is free" in caplog.text


# ── the paths that only fire when something is wrong ────────────────────────


def test_schema_keywords_nothing_validates_are_named_at_build_time(caplog):
    cap = Capability(
        tool_name="gappy",
        capability_id="gappy.probe@v1",
        product_id="prod-gappy",
        description="a capability whose schema uses keywords the pre-call checks ignore",
        input_schema={
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 3}},
            "required": [],
        },
        output_schema={"type": "object"},
        price_usd=0.0,
        schema_gaps=("minimum",),
    )
    with caplog.at_level(logging.INFO, logger="aimarket_bridges.agent_framework"):
        AIMarketTool(cap, make_hub(StubAgent()))
    assert "minimum" in caplog.text


def test_the_three_way_verification_outcome_is_surfaced_not_reimplemented(by_id, caplog):
    """The core verifies against the ORIGIN key; the bridge reports what it concluded.

    ok / invalid / not-checked are three different states, and only the middle one is a
    warning — "no key published" is not a failed signature.
    """
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
                return check

            def close(self) -> None:
                return None

        hub._keys = Resolver()
        tool = AIMarketTool(by_id["sortes.draw@v1"], hub)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="aimarket_bridges.agent_framework"):
            said = text_of(asyncio.run(tool.invoke(arguments={"alpha": "hi"})))

        assert tool.last_result is not None
        assert tool.last_result.receipt_verified is check.verified, label
        assert tool.last_result.receipt_verify_reason == check.reason, label
        assert ("did not verify" in caplog.text) is (label == "bad"), label
        # The answer is returned either way: an unverified receipt is a provenance problem for
        # the operator to see, not a reason to hide output the operator already paid for.
        assert "ab12" in said, label


def test_a_value_from_somewhere_else_falls_back_to_the_frameworks_own_rendering(by_id):
    tool = make_tool(by_id["platon.random@v1"], StubAgent())
    rendered = tool._render("a bare string from another code path")
    assert text_of(rendered) == "a bare string from another code path"


def test_an_output_json_cannot_encode_still_reaches_the_model(by_id):
    class Unserialisable:
        def __repr__(self) -> str:
            return "<opaque>"

        def __str__(self) -> str:
            raise TypeError("not even str() works")

    tool = make_tool(by_id["platon.random@v1"], StubAgent())
    result = CapabilityResult(ok=True, output={"weird": Unserialisable()}, capability_id="x")
    # `default=str` covers almost everything; this object defeats even that.
    assert "weird" in tool._render(result) or "opaque" in tool._render(result)


def test_one_unusable_capability_does_not_lose_the_catalogue(caplog, monkeypatch):
    """A hub can publish a schema this bridge cannot wrap; the other 46 must survive it."""
    import aimarket_bridges.agent_framework as module

    real_schema = module._tool_schema
    calls = {"n": 0}

    def explode_once(capability):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("schema is not something FunctionTool can take")
        return real_schema(capability)

    monkeypatch.setattr(module, "_tool_schema", explode_once)
    with _mock_client() as http, caplog.at_level(
        logging.WARNING, logger="aimarket_bridges.agent_framework"
    ):
        tools = aimarket_tools(HUB, catalog_client=http, agent=StubAgent())
    assert len(tools) == 46
    assert "skipping" in caplog.text
