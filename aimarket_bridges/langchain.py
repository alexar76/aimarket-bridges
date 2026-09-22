"""AIMarket capabilities as LangChain tools, ready to drop into a LangGraph node.

    from aimarket_bridges.langchain import aimarket_tools
    from langgraph.prebuilt import create_react_agent

    tools = aimarket_tools("https://modelmarket.dev", intent="verifiable randomness", limit=5)
    agent = create_react_agent(model, tools)

Built and measured against **langchain-core 1.5.2 / langgraph 1.2.10** on 2026-07-29. Both
had moved past what the documentation implies, in the ways numbered below.

**1. `args_schema` takes a raw JSON Schema dict, so no pydantic model is built here.**
`BaseTool.args_schema` is typed `type[BaseModel] | type[v1.BaseModel] | dict | None`, and a
dict is stored *by reference and untouched* — measured: `tool.args_schema is the_dict`. Of the
three bridges this is the only one that can skip `schema.model_from_schema` entirely, and
that is not merely less code, it is **higher fidelity**. A capability's schema reaches the
model verbatim: `fermat.route@v1` and `kantor.transport@v1` declare `oneOf` branches, which
survive into the bound tool definition exactly as the hub published them, whereas the pydantic
round trip has to collapse them to a `Union` and loses the branch structure. Verified through
`convert_to_openai_tool` on a tool taken back off a real bound model.

Schema fidelity is not the same as ARGUMENT-NAME fidelity, and the second one needed a fix of
its own — see `_CapabilityTool`. A raw dict schema means the model is shown, and sends, the
capability's own property names, so `fourier.verify@v1`'s `lambda` and the `from` nested in a
`fermat.route@v1` edge need none of the aliasing the pydantic bridges do (`lambda` is only a
problem for a Python *parameter*; `call(**{"lambda": 0.5})` is legal and is what happens here).
But `langchain` reserves two names of its own inside the call path, and a capability property
that collides with either was silently replaced on a billed call.

The price of that fidelity: **a dict `args_schema` disables client-side validation.**
Measured — with `{"required": ["n"]}`, calls with `n` missing, with `n="not-an-int"`, and with
an undeclared extra key were all handed to the function unaltered. A pydantic `args_schema`
would have rejected them locally. That is mostly correct behaviour for this bridge, because
the hub is the authority on its own contract and it answers a bad argument with a readable
refusal the model can retry against. But it is *not* correct for money: none of the 47 live
capabilities is free, so a call that is certain to be refused must not be paid for. Hence the
one narrow local guard in `_invoker` — missing required arguments, and nothing else.

**2. `response_format="content_and_artifact"` is the provenance channel, and it works.**
Verified end to end through a real `create_react_agent` graph run: the tool returns a
`(content, artifact)` pair, the `ToolMessage` carries `content` for the model and `artifact`
alongside it. So the signed receipt rides on the message the model never reads — provenance
that costs zero context tokens yet stays reachable at `message.artifact["receipt"]`. This is
the cleanest of the three bridges on this point; the other two have no such channel and have
to fall back on out-of-band metadata.

**Errors.** `handle_tool_error=True` catches `ToolException` *only* — measured: a plain
`RuntimeError` still propagates and kills the graph. That gives exactly the split this bridge
wants, for free:

    BudgetExceeded  -> re-raised as ToolException -> ToolMessage(status="error", content=msg)
                       The operator's ceiling is not the model's fault; the graph should
                       degrade to answering from what it already has, not crash.
    HubUnavailable  -> propagates untouched. The model cannot fix a dead hub, and a graph
                       that loops on transport errors burns turns to reach the same wall.
    a refusal       -> not an exception at all, just text (see client.InvokeResult.for_model)

**3. Nothing caches a tool result by default, so fresh randomness is never sold twice.**
Checked rather than assumed, because CrewAI does cache tool results by default and the same
default here would resell a `sortes.draw@v1` draw the buyer paid for. langchain-core 1.5.2 has
no cache layer at all — `BaseTool.model_fields` declares no cache field — and while langgraph
1.2.10 *does* have one (`StateGraph.compile(cache=…)` plus a per-node `cache_policy`),
`create_react_agent` reaches neither: the compiled agent's `cache` is None and every node's
`cache_policy` is None. Measured through a graph run rather than inferred — two *identical*
tool calls in one assistant turn reach the hub twice, come back with different draws, and are
billed twice. All of it is pinned by tests, because the one way to resell a draw here is a
hand-built graph that puts a `cache_policy` on its tools node, and that is a decision the
graph's author has to make deliberately.

**Whose budget it is.** `budget_usd` belongs to a `HubClient`. Passing `client=` *and*
`budget_usd=` is therefore refused rather than silently ignored: an operator who asks for a
$0.10 ceiling and gets the $1.00 default has been overcharged by this API, not by the hub. A
NEGATIVE `budget_usd` is refused because it means nothing. Zero is allowed and honoured:
`HubClient` treats it as "spend nothing", which is a legitimate dry-run — build the tools,
let the model see them, refuse every paid call.

The module is named `aimarket_bridges.langchain`, which is a name collision but *not* a
shadowing one: Python 3 has no implicit relative imports, so a bare `import langchain`
anywhere inside this package resolves to the installed distribution, never to this file.
Measured, because the earlier note here claimed the opposite: executed with
`__package__="aimarket_bridges"`, `import langchain` still raises ModuleNotFoundError in this
environment, where only `langchain-core` is installed. Only `langchain_core` is imported below
in any case.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from langchain_core.runnables import run_in_executor
from langchain_core.tools import BaseTool, BaseToolkit, StructuredTool, ToolException

from aimarket_bridges.catalog import Capability, fetch_catalog
from aimarket_bridges.client import BudgetExceeded, HubClient

logger = logging.getLogger(__name__)

__all__ = ["aimarket_tools", "tool_for_capability", "AIMarketToolkit"]


class _CapabilityTool(StructuredTool):
    """A `StructuredTool` whose `_run` reserves no argument name of its own.

    `BaseTool.run` injects into the call whatever it finds on `_run`'s signature: a
    `run_manager` parameter, and any parameter annotated `RunnableConfig`.
    `StructuredTool._run` declares both, and `run` merges them over the arguments with `|=` —
    so a capability property named `config` or `run_manager` had the model's value *replaced by
    langchain's own object* before the invoker was reached. Measured on 1.5.2, both ways it can
    land, and a dict `args_schema` validates nothing so neither raised:

        optional property   the call goes out silently missing an argument the model supplied,
                            and is billed for whatever default the capability then used
        required property   the local guard sees the name absent and refuses every call, so
                            the capability is simply uncallable through this bridge

    Overriding `_run` is the whole fix: `run` finds neither name on it and injects nothing, so
    every key the model sent arrives verbatim. `_arun` needs the same treatment because `arun`
    inspects `_arun`'s signature instead of `_run`'s unless the class inherits `BaseTool._arun`
    — StructuredTool defines one, so the async path clobbered the same two names.

    None of the 47 live capabilities uses either name today. `config` is one capability away
    from it, and the failure would be silent.
    """

    # `self` is positional-only for the same reason: these are called as bound methods with
    # the model's arguments as keywords, and a capability property named `self` would
    # otherwise raise TypeError("got multiple values for argument 'self'") — measured.
    def _run(self, /, *args: Any, **kwargs: Any) -> Any:
        if self.func is None:  # pragma: no cover - every tool here is built with func=
            raise NotImplementedError(f"{self.name} has no sync implementation")
        return self.func(*args, **kwargs)

    async def _arun(self, /, *args: Any, **kwargs: Any) -> Any:
        # The hub call is blocking, so it goes to a worker thread — which is what the lock
        # inside HubClient's spend counter is for. A closure rather than
        # `run_in_executor(None, self._run, **kwargs)`: that helper has its own `func`
        # parameter, so a property named `func` would collide there instead.
        return await run_in_executor(None, lambda: self._run(*args, **kwargs))


def _args_schema_for(capability: Capability) -> dict[str, Any]:
    """The capability's own input schema, in the shape `args_schema` wants.

    Deep-copied because langchain stores the dict by reference: without the copy the tool an
    agent was built against would silently change shape if anything mutated
    `Capability.input_schema` afterwards. `Capability` is a frozen dataclass, which protects
    the field but not the dict it points at.

    Normalising `type` and `properties` is not cosmetic: `BaseTool.args` reads
    `args_schema["properties"]` straight out of the dict, so a schema without that key raises
    KeyError — measured — and `.args` is what langchain's own prompt renderers and several
    agent constructors call. A missing `type` leaves the bound definition without one, which
    is a contract the model has to guess at.
    """
    schema = copy.deepcopy(capability.input_schema) if capability.input_schema else {}
    if not isinstance(schema, dict):
        schema = {}
    schema.setdefault("type", "object")
    if not isinstance(schema.get("properties"), dict):
        schema["properties"] = {}
    return schema


def _missing_required(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """Required arguments the caller did not supply.

    The only validation done locally, and only because it is the one failure that is both
    free to detect and certain to cost money otherwise. A property that declares a `default`
    is excluded: JSON Schema permits `required` alongside `default`, and the server is then
    entitled to fill it in, so refusing locally would block a call the hub would have served.
    """
    required = schema.get("required")
    if not isinstance(required, list):
        return []
    properties = schema.get("properties") or {}

    def has_default(name: str) -> bool:
        # A property spec may legally be a boolean (`{"properties": {"n": true}}`), and then
        # it is not subscriptable — `"default" in True` raises TypeError, which is not a
        # ToolException and so would kill the graph on every call to that tool.
        spec = properties.get(name)
        return isinstance(spec, dict) and "default" in spec

    return [
        name for name in required
        if isinstance(name, str) and name not in arguments and not has_default(name)
    ]


def _budget_for(
    client: HubClient | None, budget_usd: float | None, kw: dict[str, Any]
) -> float:
    """The ceiling the tools will share, or a clear error about who owns it.

    Checked before the catalogue is read, so a caller who asks for a budget that cannot be
    honoured learns it without a network round trip. Two ways to end up with a ceiling nobody
    asked for, both of them silent until this existed:

    `client=` together with `budget_usd=`. The budget lives on the `HubClient`, so the second
    argument was dropped and the client's own ceiling — $1.00 by default — applied instead.
    Asking for $0.10 and being handed ten times that is this bridge overcharging, not the hub.
    Everything else in `**kw` (`timeout`, `verify_receipts`, `affiliate_id`, `agent`) is
    HubClient configuration too, and was dropped just as quietly.

    `budget_usd=0` is allowed. It used to be refused here, and the reason given was that
    `HubClient._reserve` read a falsy budget as "no ceiling" and spent without limit — which
    was true, and was a footgun each of the three adapters had independently grown a guard
    against. That is a defect in the core, not something three bridges should each police, so
    it was fixed there: 0 now means "spend nothing" and `None` means "no ceiling". A zero
    ceiling is a coherent request — build the tools, refuse every paid call — so it is passed
    through. Negative is still refused; it means nothing at all.
    """
    if client is not None:
        if client.budget_usd is None:
            logger.warning(
                "the HubClient passed as client= has budget_usd=None, so these tools may "
                "spend without limit (no live capability is free)",
            )
        ignored = sorted(kw)
        if budget_usd is not None:
            ignored.insert(0, "budget_usd")
        if ignored:
            raise ValueError(
                f"{', '.join(ignored)} configure the HubClient this would build, and you "
                f"passed client= so none is built — set them on that HubClient instead "
                f"(the one you passed allows ${client.budget_usd:.2f})"
            )
        return client.budget_usd

    budget = 1.0 if budget_usd is None else float(budget_usd)
    if budget < 0:
        raise ValueError(
            f"budget_usd must not be negative, got {budget!r}. Use 0 to forbid every paid "
            "call, or a positive ceiling; HubClient honours both."
        )
    return budget


def _capabilities_for(
    base_url: str,
    *,
    intent: str,
    limit: int,
    max_price_usd: float | None,
    free_only: bool,
    http_client: Any,
    kw: dict[str, Any],
) -> list[Capability]:
    """The filtered catalogue, read before any `HubClient` exists.

    Ordering, not style: `fetch_catalog` raises when the hub is unreachable, and a `HubClient`
    built first would then be lost holding an open httpx pool that nothing can ever close.
    (One, not two: the SDK session is opened in `HubClient.__init__`, while the receipt
    resolver's client is lazy and only appears once a receipt is actually verified.)
    """
    catalog_kw: dict[str, Any] = {}
    if "timeout" in kw:
        # The catalogue read is the request most likely to hang at build time, so a caller who
        # sets a timeout for the invokes means it here too. Left in `kw` as well: it is
        # HubClient configuration and this is only a second use of the same number.
        catalog_kw["timeout"] = float(kw["timeout"])

    capabilities = fetch_catalog(
        base_url,
        intent=intent,
        limit=limit,
        max_price_usd=max_price_usd,
        free_only=free_only,
        client=http_client,
        **catalog_kw,
    )

    if not capabilities:
        # fetch_catalog raises rather than returning [] when the hub is unreachable, so an
        # empty list here means the filters matched nothing. Said out loud because it is the
        # trap this catalogue sets: not one of the 47 live capabilities is free, so
        # free_only=True yields an agent with no tools and no error — measured, langgraph
        # builds a graph with no tools node at all from an empty list.
        logger.warning(
            "no capability at %s passed the filters (intent=%r limit=%s max_price_usd=%s "
            "free_only=%s) — the returned agent will have no tools",
            base_url, intent, limit, max_price_usd, free_only,
        )
    return capabilities


def _invoker(capability: Capability, client: HubClient, schema: dict[str, Any]):
    """The function behind one tool.

    Always returns a 2-tuple. `response_format="content_and_artifact"` raises ValueError on
    anything else — measured — so the refusal and guard paths have to carry an artifact too,
    even though there is no receipt to put in it.
    """

    def call(**arguments: Any) -> tuple[Any, dict[str, Any]]:
        provenance: dict[str, Any] = {
            "capability_id": capability.capability_id,
            "price_usd": 0.0,
            "receipt": None,
            "receipt_verified": None,
            "receipt_verify_reason": "",
            "ok": False,
            # The shared running total, not this call's cost: langgraph fans parallel tool
            # calls out over real worker threads (measured — a Barrier across four of them
            # completes), so a sibling call may already have been billed by the time this one
            # reads the counter. `price_usd` is the per-call figure.
            "spent_usd": 0.0,
        }

        missing = _missing_required(schema, arguments)
        if missing:
            provenance["receipt_verify_reason"] = "not called: missing required arguments"
            provenance["spent_usd"] = client.spent_usd
            return (
                f"{capability.capability_id} needs the argument(s) "
                f"{', '.join(sorted(missing))}, which were not supplied. Nothing was called "
                f"and nothing was billed — supply them and call again.",
                provenance,
            )

        try:
            result = client.invoke(capability, arguments)
        except BudgetExceeded as exc:
            # ToolException is the one exception type `handle_tool_error` intercepts, so this
            # reaches the model as ToolMessage(status="error") text instead of a traceback.
            raise ToolException(str(exc)) from exc

        provenance.update(
            ok=result.ok,
            price_usd=result.price_usd,
            receipt=result.receipt,
            receipt_verified=result.receipt_verified,
            receipt_verify_reason=result.receipt_verify_reason,
            spent_usd=client.spent_usd,
        )
        # for_model() is the capability's output on success and a readable sentence on a
        # refusal. Structured output needs no serialising here: langchain JSON-encodes
        # anything that is not already valid message content, and the one exception —
        # a list whose every element is a string or a typed content block, which it keeps as
        # a list of blocks — still reaches the model as the same text.
        return result.for_model(), provenance

    call.__name__ = capability.tool_name
    return call


def tool_for_capability(
    capability: Capability,
    client: HubClient,
    *,
    include_price: bool = True,
    handle_budget_errors: bool = True,
) -> StructuredTool:
    """One capability as a `StructuredTool`.

    Exposed separately from `aimarket_tools` because a graph that already holds `Capability`
    records — from its own filtered `fetch_catalog` call, or a saved catalogue — should not
    have to re-read the manifest to wrap one of them.
    """
    schema = _args_schema_for(capability)

    if capability.schema_gaps:
        # Only reachable if a future capability uses something `unsupported_keywords` flags.
        # Worth saying even here, where the schema passes through untouched: langchain will
        # forward the keyword to the model, but the provider is free to ignore what it does
        # not understand, so the model may still be shown a looser contract than the real one.
        logger.warning(
            "%s: input schema uses %s; it is forwarded verbatim, but a model provider that "
            "does not implement the keyword will treat the arguments as looser than they are",
            capability.capability_id, ", ".join(capability.schema_gaps),
        )

    return _CapabilityTool.from_function(
        func=_invoker(capability, client, schema),
        name=capability.tool_name,
        description=capability.tool_description(include_price=include_price),
        # The raw hub schema, straight through. No pydantic model is built for this framework.
        args_schema=schema,
        # infer_schema would otherwise try to read the signature of a **kwargs function.
        infer_schema=False,
        response_format="content_and_artifact",
        handle_tool_error=handle_budget_errors,
        # Routing data for the graph, not for the model: `metadata` never enters the prompt,
        # so a supervisor node can filter on price or origin without spending a token on it.
        metadata={
            "capability_id": capability.capability_id,
            "price_usd": capability.price_usd,
            "source_hub": capability.source_hub,
            "product_id": capability.product_id,
        },
        # Tags are what langchain's own callback filtering matches on, which makes
        # "trace every paid call" a one-line filter rather than a metadata walk.
        tags=["aimarket", "free" if capability.is_free else "paid"],
    )


def aimarket_tools(
    base_url: str,
    *,
    intent: str = "",
    limit: int = 0,
    max_price_usd: float | None = None,
    free_only: bool = False,
    budget_usd: float | None = None,
    client: HubClient | None = None,
    http_client: Any = None,
    include_price: bool = True,
    **kw: Any,
) -> list[StructuredTool]:
    """Every capability the hub sells that passes the filters, as `StructuredTool`s.

    `intent` ranks by the hub's own search and keeps only what it returns; `limit`,
    `max_price_usd` and `free_only` filter at BUILD time, which is the only honest place —
    once a tool is in an agent's registry the agent decides when to call it.

    Pass `client` to own the `HubClient` yourself, which is how you read `spent_usd` after a
    run and close the connection pool; `budget_usd` then belongs to that client and passing
    both is an error rather than a silent no-op. Extra keyword arguments go to the `HubClient`
    this builds when you do not (`timeout`, `verify_receipts`, `affiliate_id`, `agent`), and
    `timeout` covers the catalogue read as well. `http_client` is used only for the catalogue.

    Raises `CatalogError` if the hub cannot be read: an agent that boots believing it has no
    tools is a worse failure than one that refuses to boot.
    """
    budget = _budget_for(client, budget_usd, kw)
    capabilities = _capabilities_for(
        base_url, intent=intent, limit=limit, max_price_usd=max_price_usd,
        free_only=free_only, http_client=http_client, kw=kw,
    )
    if not capabilities:
        # Nothing to wrap, and no HubClient built — building one here would leave an open
        # httpx pool behind for a caller who only ever receives an empty list.
        return []

    hub = client or HubClient(base_url, budget_usd=budget, **kw)
    built = []
    for capability in capabilities:
        try:
            built.append(tool_for_capability(capability, hub, include_price=include_price))
        except Exception as exc:  # noqa: BLE001 - see the note below
            logger.warning(
                "skipping %s: its argument schema cannot be turned into a StructuredTool "
                "(%s: %s). The other %d capabilities are unaffected",
                capability.capability_id, type(exc).__name__, exc, len(capabilities) - 1,
            )
    return built


class AIMarketToolkit(BaseToolkit):
    """Tools plus the `HubClient` they share.

    This exists for one thing a `list[StructuredTool]` structurally cannot do: give the caller
    the spend counter and the connection pool. `aimarket_tools(...)` with no `client=` builds a
    `HubClient` internally, and it is then unreachable — you cannot ask what a graph run cost,
    and nothing closes the httpx pool inside it. The toolkit hands that back:

        with AIMarketToolkit.from_hub("https://modelmarket.dev", budget_usd=0.25) as kit:
            agent = create_react_agent(model, kit.get_tools())
            agent.invoke({"messages": [("user", "draw three winners")]})
            print(kit.spent_usd, kit.last_receipt)

    Nothing else is added, and `aimarket_tools` remains the short answer for the common case.
    """

    # BaseToolkit is a pydantic BaseModel with no fields, so both of these are declared and
    # HubClient needs the arbitrary-types escape hatch.
    model_config = {"arbitrary_types_allowed": True}

    client: HubClient
    tools: list[StructuredTool] = []

    @classmethod
    def from_hub(
        cls,
        base_url: str,
        *,
        intent: str = "",
        limit: int = 0,
        max_price_usd: float | None = None,
        free_only: bool = False,
        budget_usd: float | None = None,
        client: HubClient | None = None,
        http_client: Any = None,
        include_price: bool = True,
        **kw: Any,
    ) -> AIMarketToolkit:
        """The same build as `aimarket_tools`, keeping the client instead of hiding it.

        The catalogue is read BEFORE the `HubClient` is constructed, which is the whole reason
        this does not just call `aimarket_tools(client=...)`: constructing first and reading
        second means an unreachable hub raises with a live `HubClient` — and its open httpx
        pool — already built and unreferenced, since the exception never returns the toolkit
        that would have closed it.
        """
        budget = _budget_for(client, budget_usd, kw)
        capabilities = _capabilities_for(
            base_url, intent=intent, limit=limit, max_price_usd=max_price_usd,
            free_only=free_only, http_client=http_client, kw=kw,
        )
        hub = client or HubClient(base_url, budget_usd=budget, **kw)
        return cls(
            client=hub,
            tools=[
                tool_for_capability(c, hub, include_price=include_price) for c in capabilities
            ],
        )

    def get_tools(self) -> list[BaseTool]:  # type: ignore[override]
        """The abstract method BaseToolkit requires."""
        return list(self.tools)

    @property
    def spent_usd(self) -> float:
        return self.client.spent_usd

    @property
    def remaining_usd(self) -> float:
        return self.client.remaining_usd

    @property
    def last_receipt(self) -> dict[str, Any] | None:
        return self.client.last_receipt

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> AIMarketToolkit:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
