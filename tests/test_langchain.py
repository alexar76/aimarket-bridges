"""The LangChain / LangGraph bridge, tested against langchain-core 1.5.2 + langgraph 1.2.10.

Two things make these tests worth more than a mock round trip.

The capabilities are the real ones. `live_manifest.json` is the captured manifest of
https://modelmarket.dev (47 entries, 42 of them federated, none free), so the schemas being
passed through are the schemas the hub actually publishes — including the four that use
`oneOf`, which is exactly where the "pass the raw dict through" claim earns or loses.

The graph is a real graph. `create_react_agent` is built and *run* here, with a scripted stub
in place of the model, because "the tool schemas survive binding" and "the artifact reaches
the ToolMessage" are claims about langgraph's plumbing and cannot be checked by inspecting the
tool object. No test touches the network and no test calls an LLM.

On the stub: `BaseChatModel.bind_tools` raises NotImplementedError in 1.5.2, and
`create_react_agent` calls it eagerly at construction — so langchain's own
`ParrotFakeChatModel` and `GenericFakeChatModel` both fail there (measured). The minimal stub
therefore needs three things: `_llm_type`, `_generate`, and a `bind_tools` override.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import pathlib
import re
import threading
from typing import Any

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool, ToolException
from langchain_core.utils.function_calling import convert_to_openai_tool

from aimarket_bridges.catalog import Capability, CatalogError, fetch_catalog
from aimarket_bridges.client import BudgetExceeded, HubClient, HubUnavailable
from aimarket_bridges.langchain import (
    AIMarketToolkit,
    aimarket_tools,
    tool_for_capability,
)

FIXTURE = pathlib.Path(__file__).parent / "live_manifest.json"
MANIFEST = json.loads(FIXTURE.read_text())
TOOLS = MANIFEST["tools"]
HUB = "https://modelmarket.dev"

# The four live capabilities whose input schema uses `oneOf` — the fidelity test cases.
ONEOF_IDS = [
    t["capability_id"] for t in TOOLS if "oneOf" in json.dumps(t.get("input_schema") or {})
]

LANGCHAIN_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Prices are NOT uniform — the live catalogue spreads $0.001 to $0.15 over 12 distinct price
# points. A first draft of these tests assumed a flat $0.01 and two of them passed for the
# wrong reason, so anything price-dependent is derived from the fixture rather than guessed.
PRICES = [t["price_per_call_usd"] for t in TOOLS]
CHEAPEST = min(PRICES)


# ── offline plumbing ─────────────────────────────────────────────────────────


def _http(manifest: dict[str, Any] | None = None, matches: list[dict] | None = None) -> httpx.Client:
    """A hub that serves the captured manifest and nothing else."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/manifest"):
            return httpx.Response(200, json=manifest if manifest is not None else MANIFEST)
        if request.url.path.endswith("/search"):
            return httpx.Response(200, json={"matches": matches or []})
        return httpx.Response(404, json={"error": "not found"})

    return httpx.Client(transport=httpx.MockTransport(handler))


def _http_status(code: int) -> httpx.Client:
    """A hub that answers every request with an error status."""
    return httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(code, json={"error": "no"}))
    )


def _dead_http() -> httpx.Client:
    """A hub that cannot be reached at all.

    Doubles as proof that a code path never went near the network: any request raises.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


class _FakeAgent:
    """Stands in for AIMarketAgent.invoke_single. Records calls, returns scripted bodies."""

    def __init__(self, *responses: Any):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def invoke_single(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.responses:
            return {"ok": True, "output": {"fine": True}}
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        pass


def _hub(*responses: Any, budget_usd: float = 1.0) -> HubClient:
    """A HubClient with the network replaced and receipt verification off.

    `verify_receipts=False` keeps OriginKeyResolver from reaching for a well-known document;
    the origin-key behaviour itself is the core's test, not this bridge's.

    The stub is also parked on `.fake` so a test can assert on what the hub was asked to do —
    "the guard prevented the call" is only provable by looking at the calls that were made.
    """
    agent = _FakeAgent(*responses)
    client = HubClient(HUB, budget_usd=budget_usd, agent=agent, verify_receipts=False)
    client.fake = agent  # type: ignore[attr-defined]
    return client


def _capability(**over: Any) -> Capability:
    base: dict[str, Any] = dict(
        tool_name="probe_v1",
        capability_id="probe.thing@v1",
        product_id="prod-probe",
        description="A probe capability.",
        input_schema={
            "type": "object",
            "properties": {"n": {"type": "integer"}, "note": {"type": "string"}},
            "required": ["n"],
        },
        output_schema={},
        price_usd=0.01,
        source_hub="local",
    )
    base.update(over)
    return Capability(**base)


def _live(capability_id: str) -> Capability:
    """One capability exactly as the hub publishes it.

    The synthetic `_capability()` is enough for money and plumbing, but nothing about argument
    NAMES can be tested on an invented schema: `lambda` and the `from` nested in a fermat edge
    are the whole risk, and no schema written here would contain them by accident.
    """
    record = next(t for t in TOOLS if t["capability_id"] == capability_id)
    return _capability(
        tool_name=re.sub(r"[^A-Za-z0-9_-]+", "_", capability_id).strip("_"),
        capability_id=capability_id,
        product_id=record["product_id"],
        description=record["description"] or capability_id,
        input_schema=record["input_schema"] or {},
        price_usd=record["price_per_call_usd"],
        source_hub=record["source_hub"],
    )


def _value_for(spec: Any) -> Any:
    """A value that satisfies one property spec, well enough to be sent.

    Only shape matters: the stub hub accepts anything. What is being proved is that whatever
    the model puts in arrives unaltered, so the values have to cover the schema shapes the
    catalogue actually uses — including a `oneOf` branch and a nested object.
    """
    if not isinstance(spec, dict):
        return "x"
    if isinstance(spec.get("enum"), list) and spec["enum"]:
        return spec["enum"][0]
    branches = spec.get("oneOf") or spec.get("anyOf")
    if isinstance(branches, list) and branches:
        return _value_for(branches[0])
    declared = spec.get("type")
    if isinstance(declared, list):
        declared = declared[0] if declared else None
    if declared == "array":
        return [_value_for(spec.get("items") or {})]
    if declared == "object":
        return {k: _value_for(v) for k, v in (spec.get("properties") or {}).items()}
    return {"string": "x", "integer": 1, "number": 1.5, "boolean": True, "null": None}.get(
        declared, "x"
    )


def _required_args(schema: dict[str, Any]) -> dict[str, Any]:
    """The smallest argument dict the local guard will let through."""
    properties = schema.get("properties") or {}
    return {name: _value_for(properties.get(name)) for name in (schema.get("required") or [])}


def _tool_call(tool: StructuredTool, args: dict[str, Any], call_id: str = "tc1") -> dict[str, Any]:
    """A ToolCall dict — the invoke shape that yields a ToolMessage with an artifact.

    Invoking with a bare argument dict returns the raw content instead and drops the
    artifact, so provenance tests must use this shape.
    """
    return {"name": tool.name, "args": args, "id": call_id, "type": "tool_call"}


class ScriptedModel(BaseChatModel):
    """The minimal stub create_react_agent accepts: _llm_type, _generate, bind_tools."""

    script: list[AIMessage] = []
    bound: list[Any] = []
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        message = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kw):
        self.bound = list(tools)
        return self


def _react_agent(tools: list[Any], script: list[AIMessage]):
    """A real compiled agent graph over `tools`, with the model scripted.

    Module level because three test classes need it: ToolNode.invoke() cannot be called
    standalone (it wants CONFIG_KEY_RUNTIME and raises `Missing required config key 'N/A'`),
    so running the compiled graph is the only way to exercise tool execution as langgraph
    actually performs it — including the parallel fan-out and the error handling.

    create_react_agent is deprecated in langgraph 1.2.10 in favour of
    langchain.agents.create_agent, but the `langchain` distribution is not installed here —
    langgraph.prebuilt is the path that exists in this environment.
    """
    from langgraph.prebuilt import create_react_agent

    model = ScriptedModel(script=script)
    return create_react_agent(model, tools), model


def _parallel_turn(
    tool: StructuredTool, count: int, args: dict[str, Any] | None = None
) -> AIMessage:
    """One assistant message asking for the same tool `count` times at once.

    This is what a model does when it decides it needs four draws, and it is the shape that
    puts several paid calls inside one shared budget simultaneously. `args=None` varies the
    arguments; passing `args` repeats the identical call, which is the interesting case for a
    randomness capability.
    """
    return AIMessage(
        content="",
        tool_calls=[
            {"name": tool.name, "args": dict(args) if args is not None else {"n": i},
             "id": f"c{i}", "type": "tool_call"}
            for i in range(count)
        ],
    )


# ── the framework advantage: the raw schema goes straight through ────────────


class TestRawSchemaPassthrough:
    def test_args_schema_is_the_hub_dict_not_a_pydantic_model(self):
        capability = _capability()
        tool = tool_for_capability(capability, _hub())
        assert isinstance(tool.args_schema, dict), (
            "langchain-core 1.5.2 takes a raw JSON Schema dict; building a pydantic model "
            "here would be work the framework does not ask for"
        )
        assert tool.args_schema == capability.input_schema

    def test_the_schema_is_copied_so_later_mutation_cannot_reshape_the_tool(self):
        """langchain stores a dict args_schema by reference, so the copy is load-bearing."""
        capability = _capability()
        tool = tool_for_capability(capability, _hub())

        assert tool.args_schema is not capability.input_schema
        capability.input_schema["properties"]["injected"] = {"type": "string"}
        assert "injected" not in tool.args_schema["properties"], (
            "an agent built against this tool would otherwise see its arguments change shape"
        )

    def test_no_pydantic_model_is_built_for_this_framework(self, monkeypatch):
        """The claim in the module docstring, enforced rather than asserted in prose."""

        def explode(*a, **kw):
            raise AssertionError("model_from_schema must not be called by the langchain bridge")

        monkeypatch.setattr("aimarket_bridges.catalog.model_from_schema", explode)
        tools = aimarket_tools(HUB, client=_hub(), http_client=_http())
        assert len(tools) == len(TOOLS)

    def test_a_schemaless_capability_still_answers_dot_args(self):
        """Why `properties` is normalised rather than passed through as absent.

        `BaseTool.args` indexes `args_schema["properties"]` directly, so a tool holding
        `{"type": "object"}` raises KeyError there — measured on langchain itself below. `.args`
        is what langchain's prompt renderers call, so one schemaless capability in the
        catalogue would take down any agent that renders its tool list.
        """
        tool = tool_for_capability(_capability(input_schema={}), _hub())
        assert tool.args == {}
        assert tool.args_schema["type"] == "object"

        plain = StructuredTool.from_function(
            func=lambda **kw: ("x", {}), name="bare", description="d",
            args_schema={"type": "object"}, infer_schema=False,
            response_format="content_and_artifact",
        )
        with pytest.raises(KeyError):
            _ = plain.args

    @pytest.mark.parametrize("record", TOOLS, ids=[t["capability_id"] for t in TOOLS])
    def test_every_live_capability_becomes_a_usable_tool(self, record):
        capability = _capability(
            capability_id=record["capability_id"],
            tool_name=re.sub(r"[^A-Za-z0-9_-]+", "_", record["capability_id"]).strip("_"),
            input_schema=record["input_schema"] or {},
            product_id=record["product_id"],
            source_hub=record["source_hub"],
            price_usd=record["price_per_call_usd"],
        )
        tool = tool_for_capability(capability, _hub())
        assert LANGCHAIN_TOOL_NAME.match(tool.name), (
            f"manifest name {record['name']!r} is not a legal tool name; the derived one must be"
        )
        # A schemaless capability still has to render as an object, or providers reject it.
        assert tool.args_schema["type"] == "object"
        assert isinstance(tool.args_schema["properties"], dict)

    @pytest.mark.parametrize("capability_id", ONEOF_IDS)
    def test_oneof_survives_verbatim_all_the_way_into_the_bound_tool(self, capability_id):
        """The reason passing the dict through beats converting it.

        `oneOf` is genuine polymorphism — a fermat edge is either `[from, to, cost]` or an
        object. The pydantic route has to flatten it into a Union; here the hub's own branches
        reach the model definition unchanged.
        """
        record = next(t for t in TOOLS if t["capability_id"] == capability_id)
        capability = _capability(input_schema=record["input_schema"], capability_id=capability_id)
        tool = tool_for_capability(capability, _hub())

        model = ScriptedModel(script=[AIMessage(content="done")])
        model.bind_tools([tool])
        definition = convert_to_openai_tool(model.bound[0])
        parameters = definition["function"]["parameters"]

        assert "oneOf" in json.dumps(parameters), (
            "the oneOf branches were lost between the manifest and the bound tool definition"
        )
        assert parameters == record["input_schema"]

    @pytest.mark.parametrize("capability_id", ["kantor.transport@v1", "fermat.route@v1"])
    def test_the_schema_the_model_reads_keeps_required_as_well_as_the_branches(
        self, capability_id
    ):
        """`required` is the half of the schema that costs money when it goes missing.

        A model shown `properties` but no `required` omits mandatory arguments, and every one
        of those calls is either refused after being billed or caught by the local guard. Both
        of langchain's model-facing views are checked, because `.args` is what a prompt-based
        agent renders and `tool_call_schema` is what a tool-calling model is bound to.
        """
        record = next(t for t in TOOLS if t["capability_id"] == capability_id)
        tool = tool_for_capability(_live(capability_id), _hub())

        assert tool.args == record["input_schema"]["properties"]
        schema = tool.tool_call_schema
        assert isinstance(schema, dict), "a dict args_schema must stay a dict for the model"
        assert schema["required"] == record["input_schema"]["required"]
        assert "oneOf" in json.dumps(schema["properties"])
        assert convert_to_openai_tool(tool)["function"]["parameters"]["required"] == (
            record["input_schema"]["required"]
        )

    @pytest.mark.parametrize("capability_id", ["kantor.transport@v1", "fermat.route@v1"])
    def test_a_strict_conversion_does_not_rewrite_the_tool_s_own_schema(self, capability_id):
        """`tool_call_schema` is a SHALLOW copy of the dict, so nested specs are shared.

        Provider conversion is where a schema would get edited in place — strict mode walks the
        tree adding `additionalProperties: false`. Measured on 1.5.2: it copies first, so the
        tool still advertises what the hub published. Pinned because a regression here would
        change the contract shown to the model with nothing in the diff to show it.
        """
        tool = tool_for_capability(_live(capability_id), _hub())
        before = copy.deepcopy(tool.args_schema)
        convert_to_openai_tool(tool, strict=True)
        assert tool.args_schema == before


# ── the name the model sends is the name the capability reads ────────────────


class TestArgumentNamesReachTheHubUnchanged:
    """The single most expensive way to be wrong: a billed call the capability must refuse.

    `fourier.verify@v1` requires a property literally named `lambda`, and a `fermat.route@v1`
    edge object carries `from` — both Python keywords. The pydantic bridges have to rewrite
    them and carry an alias back; this one is supposed to need no rewrite at all, because the
    dict schema means the model reads and sends the capability's own names. "Supposed to" is
    not evidence, so the assertion is on the payload the stub hub RECEIVED.
    """

    def test_lambda_is_advertised_and_arrives_spelled_lambda(self):
        capability = _live("fourier.verify@v1")
        assert "lambda" in capability.input_schema["required"], (
            "the fixture no longer has the required keyword property this test exists for"
        )
        hub = _hub({"ok": True, "output": {"certified": True}})
        tool = tool_for_capability(capability, hub)

        assert "lambda" in tool.args
        assert "lambda" in tool.tool_call_schema["required"]

        args = {"edges": [["a", "b"]], "lambda": 0.5, "vector": [1.0, -1.0]}
        tool.invoke(_tool_call(tool, args))

        payload = hub.fake.calls[0]["input_payload"]
        assert payload == args
        assert "lambda_" not in payload, (
            "the sanitised spelling went out: fourier.verify@v1 would refuse a call already paid"
        )

    def test_lambda_survives_the_way_langgraph_actually_calls_the_tool(self):
        """Through the graph, because that is the path a model's arguments travel.

        A direct `tool.invoke` proves the function signature accepts the keyword; only a real
        run proves nothing between the AIMessage and the hub rewrites the key.
        """
        hub = _hub({"ok": True, "output": {"certified": True}})
        tool = tool_for_capability(_live("fourier.verify@v1"), hub)
        args = {"edges": [["a", "b"]], "lambda": 0.25, "vector": [1.0, -1.0]}
        agent, _ = _react_agent(
            [tool],
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": tool.name, "args": args, "id": "f1",
                                 "type": "tool_call"}],
                ),
                AIMessage(content="certified"),
            ],
        )

        agent.invoke({"messages": [HumanMessage(content="verify this eigenvalue")]})
        assert hub.fake.calls[0]["input_payload"] == args
        assert hub.spent_usd == pytest.approx(0.001), "fourier.verify@v1 costs $0.001 a call"

    def test_the_from_nested_in_a_fermat_edge_is_not_rewritten(self):
        """The nested case. It is only data here, and the test is what makes that a fact."""
        hub = _hub({"ok": True, "output": {"path": ["a", "b"]}})
        tool = tool_for_capability(_live("fermat.route@v1"), hub)
        args = {
            "edges": [{"from": "a", "to": "b", "cost": 1.0}],
            "start": "a",
            "goal": "b",
        }
        tool.invoke(_tool_call(tool, args))

        edge = hub.fake.calls[0]["input_payload"]["edges"][0]
        assert edge == {"from": "a", "to": "b", "cost": 1.0}
        assert "from_" not in edge

    @pytest.mark.parametrize("record", TOOLS, ids=[t["capability_id"] for t in TOOLS])
    def test_every_live_capability_is_sent_exactly_what_the_model_supplied(self, record):
        """All 47, one paid call each: same keys, same values, billed exactly once.

        The sweep matters because the two keyword properties are not the only names a hub can
        publish — this is the assertion that would catch the next one, whatever it is called.
        """
        capability = _live(record["capability_id"])
        hub = _hub()
        tool = tool_for_capability(capability, hub)
        args = _required_args(capability.input_schema)

        message = tool.invoke(_tool_call(tool, args))

        assert message.status == "success"
        assert len(hub.fake.calls) == 1
        assert hub.fake.calls[0]["input_payload"] == args
        assert hub.spent_usd == pytest.approx(record["price_per_call_usd"])


class TestLangchainReservesTwoArgumentNames:
    """langchain injects `run_manager` and its `config` into the call by signature.

    `BaseTool.run` reads them off `_run` and merges them over the model's arguments, so a
    capability property with either name was replaced by langchain's own object before the
    invoker saw it. Nothing raised — a dict `args_schema` validates nothing — so the hub was
    billed for a call missing an argument the model had supplied.
    """

    RESERVED = {
        "type": "object",
        "properties": {
            "config": {"type": "string"},
            "run_manager": {"type": "string"},
            "callbacks": {"type": "string"},
            "func": {"type": "string"},
            "self": {"type": "string"},
        },
        "required": ["config", "run_manager", "callbacks", "func", "self"],
    }
    ARGS = {"config": "mine", "run_manager": "mine", "callbacks": "mine",
            "func": "mine", "self": "mine"}

    def test_a_plain_structured_tool_really_does_swallow_them(self):
        """The framework behaviour this bridge works around, pinned on langchain itself.

        If a later version stops reserving the names, this is the test that says so — and the
        subclass in the bridge can then go away rather than being kept out of superstition.
        """
        seen: dict[str, Any] = {}

        def echo(**kwargs: Any) -> tuple[str, dict]:
            seen.update(kwargs)
            return "ok", {}

        plain = StructuredTool.from_function(
            func=echo, name="echo", description="echo", args_schema=self.RESERVED,
            infer_schema=False, response_format="content_and_artifact",
        )
        plain.invoke(_tool_call(plain, {"config": "mine", "run_manager": "mine"}))
        assert "config" not in seen and "run_manager" not in seen

    def test_the_bridge_forwards_every_reserved_name_verbatim(self):
        hub = _hub()
        tool = tool_for_capability(_capability(input_schema=self.RESERVED), hub)
        tool.invoke(_tool_call(tool, dict(self.ARGS)))
        assert hub.fake.calls[0]["input_payload"] == self.ARGS

    def test_the_async_paths_forward_them_too(self):
        """Two async entry points, and they take different routes through langchain.

        `ainvoke` on a tool with no coroutine runs the sync path in a thread, while `arun` goes
        through `_arun` — whose signature `arun` inspects for exactly the same two names. Only
        testing one of them would have left the other clobbering arguments.
        """
        for call in (
            lambda tool: tool.ainvoke(_tool_call(tool, dict(self.ARGS))),
            lambda tool: tool.arun(dict(self.ARGS), tool_call_id="a1"),
        ):
            hub = _hub()
            tool = tool_for_capability(_capability(input_schema=self.RESERVED), hub)
            asyncio.run(call(tool))
            assert hub.fake.calls[0]["input_payload"] == self.ARGS

    def test_a_property_named_self_does_not_raise(self):
        """`self._run(**{"self": ...})` is a TypeError unless `self` is positional-only."""
        hub = _hub()
        tool = tool_for_capability(
            _capability(input_schema={"type": "object", "properties": {"self": {}},
                                      "required": ["self"]}),
            hub,
        )
        assert tool.invoke(_tool_call(tool, {"self": "mine"})).status == "success"
        assert hub.fake.calls[0]["input_payload"] == {"self": "mine"}

    def test_the_tool_is_still_a_structured_tool_and_still_binds(self):
        """The subclass must not cost the framework integration it exists inside."""
        tool = tool_for_capability(_capability(input_schema=self.RESERVED), _hub())
        assert isinstance(tool, StructuredTool)
        agent, model = _react_agent([tool], [AIMessage(content="done")])
        assert {"agent", "tools"} <= set(agent.get_graph().nodes)
        assert convert_to_openai_tool(model.bound[0])["function"]["parameters"] == (
            tool.args_schema
        )


# ── provenance on the artifact channel ───────────────────────────────────────


class TestArtifactCarriesProvenance:
    def test_output_is_the_content_and_the_receipt_is_the_artifact(self):
        receipt = {
            "capability_id": "probe.thing@v1",
            "signature": "s" * 86,
            "signer_public_key": "k" * 43,
        }
        tool = tool_for_capability(
            _capability(), _hub({"ok": True, "output": {"drawn": [4, 8]}, "receipt": receipt})
        )
        message = tool.invoke(_tool_call(tool, {"n": 2}))

        assert isinstance(message, ToolMessage)
        assert json.loads(message.content) == {"drawn": [4, 8]}
        assert message.artifact["receipt"] == receipt
        assert message.artifact["capability_id"] == "probe.thing@v1"
        assert message.artifact["price_usd"] == 0.01
        assert "receipt_verified" in message.artifact

    def test_the_receipt_never_reaches_the_content_the_model_reads(self):
        """The point of using the artifact channel: provenance at zero token cost."""
        receipt = {"signature": "deadbeef" * 10, "signer_public_key": "k" * 43}
        tool = tool_for_capability(
            _capability(), _hub({"ok": True, "output": {"drawn": [1]}, "receipt": receipt})
        )
        message = tool.invoke(_tool_call(tool, {"n": 1}))

        assert "deadbeef" not in message.content
        assert "signature" not in message.content
        assert message.artifact["receipt"]["signature"].startswith("deadbeef")

    def test_response_format_is_declared_so_langgraph_keeps_the_artifact(self):
        tool = tool_for_capability(_capability(), _hub())
        assert tool.response_format == "content_and_artifact"

    def test_a_bare_dict_invoke_still_returns_the_output_itself(self):
        """`response_format="content_and_artifact"` must not make the simple call unusable.

        With no tool_call_id there is no ToolMessage to hang an artifact on, so langchain
        returns the content alone — measured. What matters is that it is the capability's
        output and not the `(content, artifact)` tuple, and that the receipt is still reachable
        on the client rather than lost with the artifact.
        """
        hub = _hub({"ok": True, "output": {"drawn": [5]}, "receipt": {"signature": "s"}})
        tool = tool_for_capability(_capability(), hub)

        assert tool.invoke({"n": 1}) == {"drawn": [5]}
        assert hub.last_receipt == {"signature": "s"}

        refusing = tool_for_capability(_capability(), _hub({"ok": False, "error": "too small"}))
        assert "refused this input" in refusing.invoke({"n": 0})

    def test_verification_outcome_is_surfaced_rather_than_recomputed(self):
        """`receipt_verified is None` means "not checked" and must not read as False."""
        tool = tool_for_capability(
            _capability(), _hub({"ok": True, "output": 1, "receipt": {"signature": "x"}})
        )
        artifact = tool.invoke(_tool_call(tool, {"n": 1})).artifact
        assert artifact["receipt_verified"] is None
        assert artifact["ok"] is True


# ── metadata, for routing rather than for the model ──────────────────────────


class TestMetadataForRouting:
    def test_metadata_carries_what_a_graph_routes_on(self):
        capability = _capability(
            capability_id="fermat.route@v1",
            price_usd=0.05,
            source_hub="https://oracles.modelmarket.dev/family",
        )
        tool = tool_for_capability(capability, _hub())
        assert tool.metadata == {
            "capability_id": "fermat.route@v1",
            "price_usd": 0.05,
            "source_hub": "https://oracles.modelmarket.dev/family",
            "product_id": "prod-probe",
        }

    def test_a_graph_can_filter_the_catalogue_on_metadata_alone(self):
        tools = aimarket_tools(HUB, client=_hub(), http_client=_http())
        federated = [t for t in tools if t.metadata["source_hub"] != "local"]
        local = [t for t in tools if t.metadata["source_hub"] == "local"]
        # The measured split of the live catalogue: 42 federated, 5 local.
        assert (len(federated), len(local)) == (42, 5)

    def test_tags_mark_paid_calls_for_callback_filtering(self):
        assert "paid" in tool_for_capability(_capability(price_usd=0.01), _hub()).tags
        assert "free" in tool_for_capability(_capability(price_usd=0.0), _hub()).tags

    def test_the_price_is_in_the_description_the_model_reads(self):
        tool = tool_for_capability(_capability(price_usd=0.01), _hub())
        assert "$0.0100" in tool.description
        bare = tool_for_capability(_capability(), _hub(), include_price=False)
        assert "$" not in bare.description


# ── refusals are text; only what the model cannot fix raises ─────────────────


class TestRefusalsAndErrors:
    def test_a_capability_refusal_arrives_as_readable_text_not_an_exception(self):
        hub = _hub({"ok": False, "error": "'n' must be >= 1, got 0"})
        tool = tool_for_capability(_capability(), hub)
        message = tool.invoke(_tool_call(tool, {"n": 0}))

        assert isinstance(message, ToolMessage)
        assert message.status == "success", "a refusal is a result the model can act on"
        assert "'n' must be >= 1" in message.content
        assert message.artifact["ok"] is False
        # No receipt came back, so nothing was metered and the reservation was released.
        assert hub.spent_usd == 0.0
        assert message.artifact["price_usd"] == 0.0

    def test_a_refusal_that_was_billed_says_so_instead_of_hiding_it(self):
        """A receipt means the hub metered the call even though it refused.

        The dangerous version of this is a refusal that quietly consumes budget: a loop of
        them would drain the ceiling with nothing to show. So the spend moves AND the artifact
        carries the price, which is what makes it auditable after the run.
        """
        hub = _hub({"ok": False, "error": "out of range", "receipt": {"nonce": "n"}})
        tool = tool_for_capability(_capability(price_usd=0.02), hub)
        message = tool.invoke(_tool_call(tool, {"n": 99}))

        assert message.status == "success" and "out of range" in message.content
        assert hub.spent_usd == 0.02
        assert message.artifact["price_usd"] == 0.02
        assert message.artifact["ok"] is False

    def test_a_safety_block_reaches_the_model_as_text_and_costs_nothing(self):
        """The hub blocks before the provider runs, so this must not consume the ceiling."""
        hub = _hub({"safety_blocked": True, "reason": "policy"})
        tool = tool_for_capability(_capability(price_usd=0.02), hub)
        message = tool.invoke(_tool_call(tool, {"n": 1}))

        assert message.status == "success" and "safety gate" in message.content
        assert hub.spent_usd == 0.0

    def test_budget_exceeded_becomes_a_readable_tool_message(self):
        """ToolException is the only exception handle_tool_error intercepts — hence the wrap."""
        hub = _hub(budget_usd=1.0)
        tool = tool_for_capability(_capability(price_usd=0.60), hub)
        assert tool.invoke(_tool_call(tool, {"n": 1})).status == "success"

        message = tool.invoke(_tool_call(tool, {"n": 2}, call_id="tc2"))
        assert message.status == "error", "the graph must survive hitting the spend ceiling"
        assert "budget" in message.content.lower()
        assert "0.60" in message.content
        # The reservation is claimed before the call and raises before it is applied, so the
        # refused call leaves the counter exactly where the served one left it.
        assert hub.spent_usd == 0.60
        assert len(hub.fake.calls) == 1

    def test_the_wrapped_exception_is_a_tool_exception(self):
        tool = tool_for_capability(
            _capability(price_usd=5.0), _hub(budget_usd=0.10), handle_budget_errors=False
        )
        with pytest.raises(ToolException) as caught:
            tool.invoke(_tool_call(tool, {"n": 1}))
        assert isinstance(caught.value.__cause__, BudgetExceeded)

    def test_a_dead_hub_still_propagates_and_is_not_disguised_as_an_answer(self):
        """Deliberate asymmetry: the model cannot fix transport, so it must not see prose.

        Measured on 1.5.2: handle_tool_error=True catches ToolException only, so a
        HubUnavailable escapes without any extra wiring here.
        """
        tool = tool_for_capability(_capability(), _hub(ConnectionError("connection refused")))
        with pytest.raises(HubUnavailable):
            tool.invoke(_tool_call(tool, {"n": 1}))

    def test_spend_is_released_when_the_call_never_happened(self):
        hub = _hub(ConnectionError("down"))
        tool = tool_for_capability(_capability(price_usd=0.25), hub)
        with pytest.raises(HubUnavailable):
            tool.invoke(_tool_call(tool, {"n": 1}))
        assert hub.spent_usd == 0.0


# ── the one local guard: do not pay for a call that must be refused ──────────


class TestMissingRequiredArgumentsGuard:
    def test_a_missing_required_argument_costs_nothing_and_explains_itself(self):
        """A dict args_schema does no validation, so nothing else would stop this call."""
        hub = _hub()
        tool = tool_for_capability(_capability(), hub)
        message = tool.invoke(_tool_call(tool, {"note": "no n supplied"}))

        assert "n" in message.content
        assert "nothing was billed" in message.content
        assert hub.spent_usd == 0.0
        assert hub.fake.calls == [], "the hub must not have been called at all"

    def test_the_guard_does_not_fire_when_the_argument_is_present(self):
        hub = _hub()
        tool = tool_for_capability(_capability(), hub)
        tool.invoke(_tool_call(tool, {"n": 1}))
        assert len(hub.fake.calls) == 1
        assert hub.spent_usd == 0.01

    def test_a_required_property_with_a_default_is_left_to_the_hub(self):
        """`required` plus `default` is legal, and the server may fill it in."""
        hub = _hub()
        capability = _capability(
            input_schema={
                "type": "object",
                "properties": {"n": {"type": "integer", "default": 3}},
                "required": ["n"],
            }
        )
        tool = tool_for_capability(capability, hub)
        tool.invoke(_tool_call(tool, {}))
        assert len(hub.fake.calls) == 1, "refusing locally would block a call the hub serves"

    def test_a_boolean_property_spec_does_not_crash_the_guard(self):
        """`{"properties": {"n": true}}` is legal JSON Schema, and the guard read it as a dict.

        `"default" in True` raises TypeError, which `handle_tool_error` does not intercept and
        langgraph's default handler re-raises — so a hub publishing a boolean property spec
        would abort the graph on every call to that tool, before any of it reached the hub.
        """
        hub = _hub()
        capability = _capability(
            input_schema={"type": "object", "properties": {"n": True}, "required": ["n"]}
        )
        tool = tool_for_capability(capability, hub)

        assert "n" in tool.invoke(_tool_call(tool, {})).content
        assert hub.fake.calls == []
        tool.invoke(_tool_call(tool, {"n": 1}, call_id="tc2"))
        assert hub.fake.calls[0]["input_payload"] == {"n": 1}

    def test_nothing_else_is_validated_locally(self):
        """Wrong types reach the hub on purpose: it is the authority on its own contract."""
        hub = _hub({"ok": False, "error": "'n' must be an integer, got str"})
        tool = tool_for_capability(_capability(), hub)
        message = tool.invoke(_tool_call(tool, {"n": "twelve"}))
        assert hub.fake.calls[0]["input_payload"] == {"n": "twelve"}
        assert "must be an integer" in message.content


# ── langgraph: construction, binding, and a real graph run ───────────────────


class TestLangGraphIntegration:
    def _agent(self, tools, script):
        return _react_agent(tools, script)

    def test_a_react_agent_can_be_built_over_the_whole_live_catalogue(self):
        tools = aimarket_tools(HUB, client=_hub(), http_client=_http())
        agent, model = self._agent(tools, [AIMessage(content="done")])

        assert {"agent", "tools"} <= set(agent.get_graph().nodes)
        assert len(model.bound) == len(TOOLS), "every tool must reach bind_tools"

    def test_tool_schemas_survive_binding_for_every_live_capability(self):
        tools = aimarket_tools(HUB, client=_hub(), http_client=_http())
        _, model = self._agent(tools, [AIMessage(content="done")])

        by_name = {t["capability_id"]: t for t in TOOLS}
        for bound in model.bound:
            definition = convert_to_openai_tool(bound)["function"]
            expected = by_name[bound.metadata["capability_id"]]["input_schema"] or {}
            if expected.get("properties"):
                assert definition["parameters"]["properties"] == expected["properties"], (
                    f"{bound.metadata['capability_id']} lost properties during binding"
                )
            assert LANGCHAIN_TOOL_NAME.match(definition["name"])

    def test_a_full_graph_run_delivers_the_artifact_to_the_tool_message(self):
        """The claim that cannot be checked on the tool object alone."""
        receipt = {"signature": "a" * 86, "signer_public_key": "k" * 43}
        hub = _hub({"ok": True, "output": {"drawn": [7]}, "receipt": receipt})
        tool = tool_for_capability(_capability(), hub)
        agent, _ = self._agent(
            [tool],
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": tool.name, "args": {"n": 1}, "id": "tc1", "type": "tool_call"}
                    ],
                ),
                AIMessage(content="one winner drawn"),
            ],
        )

        state = agent.invoke({"messages": [HumanMessage(content="draw one")]})
        tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]

        assert len(tool_messages) == 1
        assert json.loads(tool_messages[0].content) == {"drawn": [7]}
        assert tool_messages[0].artifact["receipt"] == receipt
        assert tool_messages[0].artifact["spent_usd"] == 0.01
        assert state["messages"][-1].content == "one winner drawn"

    def test_a_budget_wall_inside_a_graph_run_does_not_kill_the_graph(self):
        hub = _hub(budget_usd=0.005)  # cheaper than the capability's $0.01
        tool = tool_for_capability(_capability(), hub)
        agent, _ = self._agent(
            [tool],
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": tool.name, "args": {"n": 1}, "id": "tc1", "type": "tool_call"}
                    ],
                ),
                AIMessage(content="I could not afford that call"),
            ],
        )

        state = agent.invoke({"messages": [HumanMessage(content="draw one")]})
        failed = [m for m in state["messages"] if isinstance(m, ToolMessage)][0]
        assert failed.status == "error"
        assert "budget" in failed.content.lower()
        assert state["messages"][-1].content == "I could not afford that call"

    def test_the_tool_works_when_the_graph_runs_async(self):
        """LangGraph runs tools in a worker thread on the async path.

        Only `func` is supplied, and langchain falls back to a thread pool for `ainvoke` —
        which is exactly why HubClient's spend counter is lock-guarded.
        """
        hub = _hub({"ok": True, "output": {"drawn": [3]}, "receipt": {"signature": "s"}})
        tool = tool_for_capability(_capability(), hub)

        message = asyncio.run(tool.ainvoke(_tool_call(tool, {"n": 1})))
        assert json.loads(message.content) == {"drawn": [3]}
        assert message.artifact["receipt"] == {"signature": "s"}


# ── build-time filtering and the toolkit ─────────────────────────────────────


class TestBuildTimeFilters:
    def test_max_price_keeps_a_tool_the_operator_cannot_afford_out_of_the_registry(self):
        """Filtering happens at BUILD time, across the catalogue's real price spread."""
        ceiling = 0.004
        expected = sum(1 for p in PRICES if p <= ceiling)
        assert 0 < expected < len(TOOLS), "a ceiling that filters nothing tests nothing"

        tools = aimarket_tools(HUB, client=_hub(), http_client=_http(), max_price_usd=ceiling)
        assert len(tools) == expected
        assert all(t.metadata["price_usd"] <= ceiling for t in tools)

    def test_a_ceiling_below_the_cheapest_capability_yields_no_tools(self):
        tools = aimarket_tools(
            HUB, client=_hub(), http_client=_http(), max_price_usd=CHEAPEST / 2
        )
        assert tools == []

    def test_free_only_yields_no_tools_and_says_so(self, caplog):
        """The trap this catalogue sets: not one of the 47 capabilities is free."""
        with caplog.at_level("WARNING", logger="aimarket_bridges.langchain"):
            tools = aimarket_tools(HUB, client=_hub(), http_client=_http(), free_only=True)
        assert tools == []
        assert "no tools" in caplog.text

    def test_intent_narrows_the_registry_to_what_search_ranked(self):
        ranked = [{"capability_id": t["capability_id"]} for t in TOOLS[:3]]
        tools = aimarket_tools(
            HUB, client=_hub(), http_client=_http(matches=ranked), intent="randomness"
        )
        assert [t.metadata["capability_id"] for t in tools] == [
            r["capability_id"] for r in ranked
        ]

    def test_limit_caps_the_registry(self):
        tools = aimarket_tools(HUB, client=_hub(), http_client=_http(), limit=4)
        assert len(tools) == 4

    def test_a_timeout_covers_the_catalogue_read_as_well_as_the_calls(self, monkeypatch):
        """`timeout` is HubClient configuration, but the catalogue read is the request most
        likely to hang at build time — it used to keep fetch_catalog's own 30s default however
        short a timeout the caller asked for, so no tool existed yet and the stall was silent.
        """
        seen: dict[str, Any] = {}

        def spy(base_url, **kw):
            seen.update(kw)
            return fetch_catalog(base_url, **kw)

        monkeypatch.setattr("aimarket_bridges.langchain.fetch_catalog", spy)
        tools = aimarket_tools(
            HUB, http_client=_http(), limit=1, timeout=2.0,
            agent=_FakeAgent(), verify_receipts=False,
        )
        assert seen["timeout"] == 2.0
        assert len(tools) == 1


class TestToolkit:
    def test_the_toolkit_exposes_the_spend_a_bare_list_cannot(self):
        hub = _hub({"ok": True, "output": 1}, {"ok": True, "output": 2})
        kit = AIMarketToolkit.from_hub(HUB, client=hub, http_client=_http(), limit=2)

        tools = kit.get_tools()
        assert len(tools) == 2
        assert kit.spent_usd == 0.0
        assert kit.remaining_usd == 1.0

        for tool in tools:
            tool.invoke(_tool_call(tool, {"ecosystem": "py", "n": 1}))

        billed = sum(PRICES[:2])
        assert kit.spent_usd == pytest.approx(billed)
        assert kit.remaining_usd == pytest.approx(1.0 - billed)

    def test_every_tool_in_the_toolkit_shares_one_budget(self):
        """Two tools spending from separate counters would make the ceiling meaningless."""
        # Enough for the first capability alone, never for both.
        budget = PRICES[0] + PRICES[1] / 2
        kit = AIMarketToolkit.from_hub(
            HUB, client=_hub(budget_usd=budget), http_client=_http(), limit=2
        )
        first, second = kit.get_tools()
        assert first.invoke(_tool_call(first, {"ecosystem": "py"})).status == "success"
        assert second.invoke(_tool_call(second, {"ecosystem": "py"})).status == "error"

    def test_the_toolkit_closes_its_client(self):
        closed: list[bool] = []
        hub = _hub()
        hub.close = lambda: closed.append(True)  # type: ignore[method-assign]
        with AIMarketToolkit.from_hub(HUB, client=hub, http_client=_http(), limit=1) as kit:
            assert kit.get_tools()
        assert closed == [True]

    def test_the_toolkit_is_a_real_langchain_toolkit(self):
        from langchain_core.tools import BaseToolkit

        kit = AIMarketToolkit.from_hub(HUB, client=_hub(), http_client=_http(), limit=1)
        assert isinstance(kit, BaseToolkit)
        assert all(isinstance(t, StructuredTool) for t in kit.get_tools())


# ── whose budget it is ───────────────────────────────────────────────────────


class TestBudgetOwnership:
    """The two ways an operator could ask for one ceiling and be given another."""

    def test_a_budget_passed_alongside_a_client_is_refused_not_dropped(self):
        """`budget_usd` lives on the HubClient, so passing both used to be a silent no-op.

        Asking for $0.10 and being handed the client's $1.00 default is a tenfold overspend
        introduced by the bridge, and nothing in the run would ever show it.
        """
        hub = _hub(budget_usd=1.0)
        with pytest.raises(ValueError) as caught:
            aimarket_tools(HUB, client=hub, budget_usd=0.10, http_client=_dead_http())
        assert "budget_usd" in str(caught.value)
        assert "1.00" in str(caught.value), "the message must name the ceiling actually in force"

        with pytest.raises(ValueError):
            AIMarketToolkit.from_hub(HUB, client=hub, budget_usd=0.10, http_client=_dead_http())

    def test_other_hub_client_keywords_alongside_a_client_are_refused_too(self):
        """`verify_receipts`, `timeout`, `affiliate_id`, `agent` configure a client we do not
        build when one is handed in, so accepting them would silently discard them."""
        with pytest.raises(ValueError) as caught:
            aimarket_tools(
                HUB, client=_hub(), verify_receipts=False, timeout=5.0, http_client=_dead_http()
            )
        assert "timeout" in str(caught.value) and "verify_receipts" in str(caught.value)

    def test_the_budget_is_checked_before_the_hub_is_touched(self):
        """A bad budget must not cost a network round trip to discover.

        `_dead_http` raises on any request, so reaching ValueError proves the catalogue was
        never read. The invalid value is -1 now: 0 became legitimate when the core learned to
        honour it as "spend nothing".
        """
        with pytest.raises(ValueError):
            aimarket_tools(HUB, budget_usd=-1.0, http_client=_dead_http())

    def test_a_zero_budget_is_honoured_as_spend_nothing(self):
        """The core enforces it now, so the bridge no longer has to refuse it.

        This test used to assert the opposite, and was written that way on purpose: it pinned
        the measurement the guard rested on — `HubClient._reserve` tested
        `if self.budget_usd and ...`, so 0 skipped the check and five $0.50 calls all went
        through while `remaining_usd` said $0.00. Three adapters each carried a guard against
        that, which is what made it clear the defect was in the core. Fixed there; 0 now means
        spend nothing.
        """
        hub = _hub(budget_usd=0.0)
        for _ in range(5):
            with pytest.raises(BudgetExceeded):
                hub.invoke(_capability(price_usd=0.50), {"n": 1})
        assert hub.spent_usd == 0.0, "pre-fix this was 2.50"
        assert hub.remaining_usd == 0.0

        # And the bridge passes it through rather than vetoing a coherent request.
        tools = aimarket_tools(HUB, budget_usd=0.0, http_client=_http(), limit=1)
        assert len(tools) == 1

    def test_a_negative_budget_is_still_refused(self):
        """It means nothing, in either direction."""
        with pytest.raises(ValueError) as caught:
            aimarket_tools(HUB, budget_usd=-1.0, http_client=_dead_http())
        assert "must not be negative" in str(caught.value)

    def test_a_client_that_already_has_no_ceiling_is_warned_about_not_vetoed(self, caplog):
        """The same $0 trap, arriving on the one path the guard above cannot refuse.

        `budget_usd=None` is the spelling for "no ceiling" since the core was fixed, and a
        client arriving with it would otherwise pass in silence. The client is the caller's, so
        it is not vetoed; it is said out loud. (Before the fix the dangerous value was 0, which
        an operator would write meaning the exact opposite.)
        """
        with caplog.at_level("WARNING", logger="aimarket_bridges.langchain"):
            tools = aimarket_tools(HUB, client=_hub(budget_usd=None), http_client=_http(), limit=1)
        assert len(tools) == 1
        assert "without limit" in caplog.text

        with caplog.at_level("WARNING", logger="aimarket_bridges.langchain"):
            kit = AIMarketToolkit.from_hub(
                HUB, client=_hub(budget_usd=0.0), http_client=_http(), limit=1
            )
        assert len(kit.get_tools()) == 1 and "without limit" in caplog.text

    def test_a_client_alone_is_still_the_supported_way_to_own_the_budget(self):
        hub = _hub(budget_usd=0.25)
        tools = aimarket_tools(HUB, client=hub, http_client=_http(), limit=1)
        assert len(tools) == 1
        kit = AIMarketToolkit.from_hub(HUB, client=hub, http_client=_http(), limit=1)
        assert kit.remaining_usd == 0.25


# ── an unreachable hub at build time ─────────────────────────────────────────


class TestHubFailureAtBuildTime:
    """A hub that cannot be read must raise, never yield an agent with zero tools."""

    def test_a_hub_error_response_raises_and_names_what_could_not_be_read(self):
        with pytest.raises(CatalogError) as caught:
            aimarket_tools(HUB, http_client=_http_status(500))
        assert "/ai-market/v2/manifest" in str(caught.value)

    def test_an_unreachable_host_raises_rather_than_returning_no_tools(self):
        with pytest.raises(CatalogError) as caught:
            aimarket_tools(HUB, http_client=_dead_http())
        assert HUB in str(caught.value)

    def test_a_hub_that_is_not_an_aimarket_v2_hub_says_that(self):
        """A 200 with the wrong body is the failure that most looks like an empty catalogue."""
        with pytest.raises(CatalogError) as caught:
            aimarket_tools(HUB, http_client=_http(manifest={"data": []}))
        assert "tools" in str(caught.value)

    def test_the_toolkit_builds_no_hub_client_when_the_catalogue_cannot_be_read(
        self, monkeypatch
    ):
        """Ordering, and the reason from_hub reads the catalogue first.

        A HubClient built before the failing read would be lost holding two open httpx pools:
        the exception never returns the toolkit whose `close()` would have released them.
        """
        built: list[Any] = []
        monkeypatch.setattr(
            "aimarket_bridges.langchain.HubClient",
            lambda *a, **kw: built.append(kw) or object(),
        )
        for build in (
            lambda: AIMarketToolkit.from_hub(HUB, http_client=_http_status(503)),
            lambda: aimarket_tools(HUB, http_client=_http_status(503)),
        ):
            with pytest.raises(CatalogError):
                build()
        assert built == [], "a client was constructed for a hub that could not be read"

    def test_an_empty_registry_really_does_produce_a_graph_with_no_tools_node(self):
        """Why the empty-filter case is warned about rather than shrugged at.

        Measured: create_react_agent accepts an empty list and compiles a graph that has no
        tools node at all, so a filter that matches nothing yields a silently toolless agent.
        Only a filter can do this — an unreachable hub raises above.
        """
        agent, _ = _react_agent([], [AIMessage(content="nothing to call")])
        assert "tools" not in set(agent.get_graph().nodes)


# ── fresh randomness is never served twice ───────────────────────────────────


class TestFreshnessIsNeverCached:
    """Selling the same random number twice is the worst failure available here.

    CrewAI caches tool results by default, so this was checked rather than assumed.
    """

    def test_langchain_core_has_no_tool_result_cache_to_opt_out_of(self):
        """Pinned by introspection so a version that adds one cannot land unnoticed."""
        from langchain_core.tools import BaseTool

        fields = set(BaseTool.model_fields) | set(StructuredTool.model_fields)
        assert not [f for f in fields if "cach" in f.lower()], (
            "langchain-core grew a tool cache field; a randomness capability must opt out"
        )

        from langgraph.prebuilt import create_react_agent

        parameters = inspect.signature(create_react_agent).parameters
        assert not [p for p in parameters if "cach" in p.lower()], (
            "create_react_agent grew a cache argument; check what it does to tool results"
        )

    def test_langgraph_has_a_node_cache_and_the_prebuilt_agent_uses_none_of_it(self):
        """langgraph 1.2.10 CAN cache — the earlier note here read as if it could not.

        `StateGraph.compile` takes a `cache` and each node takes a `cache_policy`, so a
        hand-built graph can absolutely serve the same paid draw twice. What is true is that
        `create_react_agent` reaches neither, which is worth pinning separately from "no cache
        exists": the two statements fail in different releases.
        """
        from langgraph.graph import StateGraph

        assert "cache" in inspect.signature(StateGraph.compile).parameters

        agent, _ = _react_agent(
            [tool_for_capability(_capability(), _hub())], [AIMessage(content="done")]
        )
        assert agent.cache is None
        assert all(node.cache_policy is None for node in agent.nodes.values())

    def test_two_identical_draws_in_one_turn_both_reach_the_hub(self):
        """The behaviour that matters, measured through a graph rather than inferred.

        Same tool, same arguments, one assistant turn: both calls are executed, both are
        billed, and the model sees two different draws.
        """
        hub = _hub(
            {"ok": True, "output": {"draw": [11]}, "receipt": {"signature": "a"}},
            {"ok": True, "output": {"draw": [42]}, "receipt": {"signature": "b"}},
        )
        tool = tool_for_capability(
            _capability(capability_id="sortes.draw@v1", price_usd=0.001), hub
        )
        agent, _ = _react_agent(
            [tool], [_parallel_turn(tool, 2, {"n": 1}), AIMessage(content="two draws")]
        )

        state = agent.invoke({"messages": [HumanMessage(content="draw twice")]})
        drawn = sorted(
            json.loads(m.content)["draw"][0]
            for m in state["messages"]
            if isinstance(m, ToolMessage)
        )
        assert drawn == [11, 42], "the second identical call was served from somewhere else"
        assert len(hub.fake.calls) == 2
        assert hub.spent_usd == pytest.approx(0.002), "two draws, two receipts, two charges"


# ── money under the parallel tool calls langgraph really makes ───────────────


class _BarrierAgent:
    """A stub that answers nothing until `size` calls are inside it at once.

    Proves simultaneity instead of assuming it: were langgraph running the calls one after
    another, the barrier would time out and the test would fail loudly rather than pass for
    the wrong reason. Measured on langgraph 1.2.10 — four sync calls arrive on four distinct
    ThreadPoolExecutor threads and four async ones on four asyncio worker threads, so the
    lock inside HubClient's spend counter is load-bearing, not defensive.
    """

    def __init__(self, size: int, timeout: float = 10.0):
        self.barrier = threading.Barrier(size, timeout=timeout)
        self.calls: list[dict[str, Any]] = []
        self.threads: set[str] = set()
        self.ran_sequentially = False
        self._lock = threading.Lock()

    def invoke_single(self, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            self.calls.append(kwargs)
            self.threads.add(threading.current_thread().name)
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            self.ran_sequentially = True
        return {"ok": True, "output": kwargs["input_payload"], "receipt": {"signature": "s"}}

    def close(self) -> None:
        pass


class TestConcurrentToolCalls:
    @pytest.mark.parametrize("mode", ["sync", "async"])
    def test_langgraph_really_runs_paid_calls_simultaneously(self, mode):
        agent_stub = _BarrierAgent(4)
        hub = HubClient(HUB, budget_usd=1.0, agent=agent_stub, verify_receipts=False)
        tool = tool_for_capability(_capability(price_usd=0.01), hub)
        agent, _ = _react_agent(
            [tool], [_parallel_turn(tool, 4), AIMessage(content="four results")]
        )

        payload = {"messages": [HumanMessage(content="four at once")]}
        state = agent.invoke(payload) if mode == "sync" else asyncio.run(agent.ainvoke(payload))

        assert not agent_stub.ran_sequentially, (
            "the barrier timed out: langgraph no longer overlaps tool calls, so the "
            "concurrency this bridge is built for should be re-measured"
        )
        assert agent_stub.threads and "MainThread" not in agent_stub.threads
        assert len([m for m in state["messages"] if isinstance(m, ToolMessage)]) == 4
        assert hub.spent_usd == pytest.approx(0.04)

    @pytest.mark.parametrize("mode", ["sync", "async"])
    def test_parallel_calls_cannot_outspend_the_shared_budget(self, mode):
        """The money assertion this bridge's own suite was missing.

        Twelve simultaneous calls against a budget that fits four: because the reservation is
        claimed under a lock before the call, exactly four reach the hub however the threads
        interleave. The ceiling is set half a call above the four so the answer cannot depend
        on float accumulation ($0.01 is not representable, and four of them land a hair above
        $0.04). Both drive modes are checked because they take different routes into the tool —
        the async one hands the sync call to a worker thread, which is the interleaving the
        lock exists for.
        """
        price, affordable = 0.01, 4
        hub = _hub(budget_usd=price * (affordable + 0.5))
        tool = tool_for_capability(_capability(price_usd=price), hub)
        agent, _ = _react_agent(
            [tool], [_parallel_turn(tool, 12), AIMessage(content="I ran out of budget")]
        )

        payload = {"messages": [HumanMessage(content="twelve at once")]}
        state = agent.invoke(payload) if mode == "sync" else asyncio.run(agent.ainvoke(payload))
        messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        refused = [m for m in messages if m.status == "error"]

        assert len(hub.fake.calls) == affordable, "the hub was called more times than paid for"
        assert hub.spent_usd == pytest.approx(price * affordable)
        assert len(messages) == 12 and len(refused) == 12 - affordable
        assert all("budget" in m.content.lower() for m in refused)
        # The graph must still finish: a spend ceiling is the operator's decision, not an error
        # the model caused, so it degrades to answering from what it has.
        assert state["messages"][-1].content == "I ran out of budget"


# ── a refusal inside a real graph run ────────────────────────────────────────


class TestRefusalInsideAGraph:
    def test_a_refusing_capability_leaves_the_graph_running(self):
        """Tested through the graph because that is where the claim matters.

        langgraph 1.2.10's default tool-error handler re-raises anything that is not a
        `ToolInvocationError` (read from `_default_handle_tool_errors`), so a refusal that
        escaped as an exception really would abort the run — a refusal has to be a *result*.
        """
        hub = _hub({"ok": False, "error": "'n' must be >= 1, got 0"})
        tool = tool_for_capability(_capability(), hub)
        agent, _ = _react_agent(
            [tool],
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": tool.name, "args": {"n": 0}, "id": "tc1", "type": "tool_call"}
                    ],
                ),
                AIMessage(content="I will pass 1 instead"),
            ],
        )

        state = agent.invoke({"messages": [HumanMessage(content="draw zero")]})
        refusal = [m for m in state["messages"] if isinstance(m, ToolMessage)][0]

        assert refusal.status == "success", "a refusal must not read as a tool failure"
        assert "must be >= 1" in refusal.content
        assert state["messages"][-1].content == "I will pass 1 instead"
        assert hub.spent_usd == 0.0

    def test_a_dead_hub_stops_the_graph_instead_of_being_answered_around(self):
        """The asymmetry, confirmed at graph level and not just on the tool object.

        Measured: langgraph's default handler re-raises non-ToolInvocationError exceptions, so
        HubUnavailable ends the run. A graph that looped here would burn turns to reach the
        same wall.
        """
        hub = _hub(ConnectionError("connection refused"))
        tool = tool_for_capability(_capability(price_usd=0.25), hub)
        agent, _ = _react_agent(
            [tool],
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": tool.name, "args": {"n": 1}, "id": "tc1", "type": "tool_call"}
                    ],
                ),
                AIMessage(content="unreachable"),
            ],
        )

        with pytest.raises(HubUnavailable):
            agent.invoke({"messages": [HumanMessage(content="go")]})
        assert hub.spent_usd == 0.0, "the reservation for a call that never happened is released"


# ── the module name is a collision, not a shadow ─────────────────────────────


def test_a_bare_import_langchain_inside_the_package_is_the_real_distribution():
    """The module docstring claims `aimarket_bridges.langchain` shadows nothing. Measured.

    Python 3 has no implicit relative imports, so `import langchain` executed with
    `__package__="aimarket_bridges"` resolves to the top-level distribution — which is not
    installed in this environment, hence ModuleNotFoundError rather than a handle on this
    bridge. If it ever silently succeeded, the shadowing warning would need to come back.
    """
    namespace = {"__name__": "aimarket_bridges._probe", "__package__": "aimarket_bridges"}
    with pytest.raises(ModuleNotFoundError):
        exec("import langchain", namespace)  # noqa: S102

    from aimarket_bridges import langchain as bridge

    assert bridge.aimarket_tools is aimarket_tools, (
        "the collision is only inside the aimarket_bridges namespace, where it is deliberate"
    )
