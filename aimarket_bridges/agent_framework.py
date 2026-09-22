"""A live AIMarket hub's catalogue, as Microsoft Agent Framework tools.

Built and verified against **agent-framework-core 1.15.0** and **pydantic 2.13.4**
(introspected on 2026-08-24 — every claim below was measured against the installed package).

Four framework facts shaped this module.

**The capability's own JSON Schema goes in, not a pydantic model.** `FunctionTool.__init__`
accepts `input_model: type[BaseModel] | Mapping[str, Any] | None`, and the pydantic branch is
broken for a large part of this hub. With a model, `invoke` validates the arguments against it
and then dumps them with `model_dump(exclude_unset=True)` — *without* `by_alias` — while
`parameters()` advertises pydantic's by-alias schema. So for any capability with a property
that is not a Python identifier, the two halves disagree: sending the advertised `lambda`
produces `TypeError: Missing required argument(s) for 't': lambda` and the tool can never be
called at all. Handing over `capability.input_schema` skips that round-trip entirely
(`_schema_supplied`), and has the better property anyway — the model reads exactly the schema
the hub publishes, with no renaming, no inlining and no synthesised defaults in between.

**The framework checks required arguments and top-level types, and forwards everything else.**
Measured: a missing required property raises `TypeError: Missing required argument(s)`, a
wrong scalar type raises `TypeError: Invalid type for 'x' ... expected number, got str` — both
before the function runs, so neither costs money. An *undeclared* property does not raise; it
is passed to the function verbatim. That would be a billed call carrying an argument the
capability then rejects, so this module drops unknown properties itself. See `_payload`.

**Sync tools are already offloaded, onto the executor this module must not use.**
`FunctionTool._invoke_function` runs a non-coroutine tool through `asyncio.to_thread` — the
loop's DEFAULT executor, shared with every other library in the process. `HubClient.invoke`
does blocking HTTP with a 120s default timeout, so a hub that accepts a connection and then
trickles bytes would hold a shared worker for the full timeout, and a fan-out of paid calls
would starve unrelated `to_thread` users: a file read, a DNS lookup, somebody else's SDK. This
module hands the framework a *coroutine* instead and offloads onto a dedicated bounded pool,
which is the same choice `aimarket_bridges.autogen` makes for the same reason.

**`result_parser` decides what a paid call costs in context.** Without one, the framework
JSON-encodes whatever the function returned — here that would push the receipt, its
verification state and the price into the conversation on every single call. Only
`CapabilityResult.for_model()` goes out; the receipt stays reachable on the returned object
and on `tool.last_result`, which is where a caller checking provenance should look for it.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import threading
from concurrent import futures
from typing import Any

from agent_framework import FunctionTool
from pydantic import BaseModel

from aimarket_bridges.catalog import Capability, fetch_catalog
from aimarket_bridges.client import BudgetExceeded, HubClient, InvokeResult

logger = logging.getLogger(__name__)

__all__ = ["CapabilityResult", "AIMarketTool", "aimarket_tools"]

#: Threads shared by every AIMarket tool in this process, dedicated rather than the loop's
#: default executor — see the module docstring. Bounded for the mirror reason: an agent that
#: fans out is not entitled to one thread per call.
_MAX_WORKERS = 8
_POOL: "futures.ThreadPoolExecutor | None" = None
_POOL_LOCK = threading.Lock()


def _executor() -> "futures.ThreadPoolExecutor":
    """The shared pool, built on first use so importing this module starts no threads."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = futures.ThreadPoolExecutor(
                max_workers=_MAX_WORKERS, thread_name_prefix="aimarket-invoke"
            )
        return _POOL


class CapabilityResult(BaseModel):
    """One capability call, as the value the wrapped function returns.

    A pydantic model rather than a dict so a caller reading `tool.last_result` gets named
    fields, and so the provenance travels as one object instead of being flattened into the
    text the model reads.
    """

    ok: bool
    output: Any = None
    error: str = ""
    capability_id: str = ""
    price_usd: float = 0.0
    receipt: dict[str, Any] | None = None
    #: None means "not checked" — no key published, or no verifier installed. Distinct from
    #: False, which means the signature did not verify. The core keeps these apart on purpose.
    receipt_verified: bool | None = None
    receipt_verify_reason: str = ""
    #: Set instead of raising, so an orchestrator can detect an exhausted ceiling without
    #: matching on message text.
    budget_exceeded: bool = False

    def for_model(self) -> Any:
        """What the calling model should read. Same contract as `InvokeResult.for_model`."""
        if self.ok:
            return self.output
        if self.budget_exceeded:
            # The call never reached the capability, so "refused this input" would be a lie
            # that sends the model off rewriting arguments that were never the problem.
            return f"{self.capability_id} was not called: {self.error}"
        return f"{self.capability_id} refused this input: {self.error}"

    @classmethod
    def from_invoke(cls, result: InvokeResult) -> "CapabilityResult":
        return cls(
            ok=result.ok,
            output=result.output,
            error=result.error,
            capability_id=result.capability_id,
            price_usd=result.price_usd,
            receipt=result.receipt,
            receipt_verified=result.receipt_verified,
            receipt_verify_reason=result.receipt_verify_reason,
        )


def _tool_schema(capability: Capability) -> dict[str, Any]:
    """The capability's argument schema, as the framework wants to receive it.

    A DEEP copy, because `FunctionTool` caches whatever mapping it is handed and a caller
    holding a reference to `capability.input_schema` must not be able to change a live tool's
    advertised interface. A shallow copy is not enough — `properties` would still be the same
    dict, so adding a property to the capability after the tool was built would silently add
    it to what the model is offered. `type` and `properties` are filled in when absent: a
    capability that takes no arguments publishes `{}` on some hubs, and the framework's
    validator reads `properties` directly.
    """
    schema = copy.deepcopy(dict(capability.input_schema or {}))
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


class AIMarketTool(FunctionTool):
    """One hub capability as an Agent Framework tool."""

    def __init__(
        self,
        capability: Capability,
        hub: HubClient,
        *,
        include_price: bool = True,
    ):
        self.capability = capability
        self.hub = hub
        # The most recent result, for a caller that wants the receipt after the agent has run.
        # Per-tool rather than per-client so it is not overwritten by a different capability.
        self.last_result: CapabilityResult | None = None
        self._schema = _tool_schema(capability)

        if capability.schema_gaps:
            # Unlike the pydantic-backed adapters this one advertises the hub's schema
            # verbatim, so the *model* sees the full contract. The gaps still matter, because
            # the framework's own pre-call validation is shallow: the keywords listed here are
            # ones nothing between the model and the capability enforces.
            logger.info(
                "%s: schema keywords the framework does not validate before the call (%s) — "
                "the capability enforces them itself, on a call that is already billed",
                capability.tool_name, ", ".join(capability.schema_gaps),
            )

        super().__init__(
            name=capability.tool_name,
            description=capability.tool_description(include_price=include_price),
            func=self._call,
            input_model=self._schema,
            result_parser=self._render,
            # Reachable as `tool.additional_properties` on a tool the framework hands back,
            # e.g. from a middleware deciding whether a call is worth approving.
            additional_properties={
                "capability_id": capability.capability_id,
                "product_id": capability.product_id,
                "price_usd": capability.price_usd,
                "source_hub": capability.source_hub,
            },
        )

    # ── invocation ───────────────────────────────────────────────────────────

    async def _call(self, **arguments: Any) -> CapabilityResult:
        """Invoke the capability without blocking the event loop.

        A coroutine on purpose: a plain function would be run by the framework on the loop's
        default executor. See the module docstring.
        """
        cid = self.capability.capability_id
        payload = self._payload(arguments)

        loop = asyncio.get_running_loop()
        try:
            invoked = await loop.run_in_executor(
                _executor(), self.hub.invoke, self.capability, payload
            )
        except BudgetExceeded as exc:
            # Returned as text, not raised. Hitting a spend ceiling is the guard rail working
            # as designed, not an exceptional condition — an agent handed a $1 budget and 47
            # paid tools will reach it on a normal run. The model can read this and stop,
            # which is the correct response; an exception out of `invoke` gives it nothing to
            # read and unwinds the agent instead.
            result = CapabilityResult(
                ok=False, capability_id=cid, budget_exceeded=True, error=str(exc)
            )
            self.last_result = result
            return result
        except asyncio.CancelledError:
            # The await returned, but the worker thread is inside a blocking read and will run
            # to completion: the capability answers and the operator is billed. There is no
            # result for THIS call, and leaving the previous call's result on `last_result`
            # would hand a caller a stale receipt for a different call.
            self.last_result = None
            raise

        # HubUnavailable and anything else unexpected propagate, as in every other adapter
        # here: a transport or configuration failure is not something the model can fix by
        # retrying with different arguments, and swallowing it into a normal result would let
        # the model narrate an answer over a call that never happened.
        result = CapabilityResult.from_invoke(invoked)
        if result.receipt_verified is False:
            logger.warning(
                "%s: receipt did not verify against its origin key (%s)",
                cid, result.receipt_verify_reason or "no reason given",
            )
        self.last_result = result
        return result

    def _payload(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """The invoke body for one call's arguments.

        Two corrections, both measured — against agent-framework 1.15.0 for the first and the
        47 live schemas for the second.

        **Undeclared properties are dropped.** The framework validates required arguments and
        top-level types but forwards anything else untouched, so a model that invents an
        argument would otherwise have it sent to the capability — which refuses the call after
        it has been billed. Dropping is limited to schemas that do not opt into
        `additionalProperties`: where a capability does accept open keys, they are its
        contract, not noise.

        **A top-level `None` is dropped.** Not one of the 47 live capabilities declares a
        nullable property, so a null argument is a rejection on a billed call. The framework
        catches almost all of these itself — measured, it raises `Invalid type for 'tol' ...
        expected number, got NoneType` for a `type` string and `expected one of ['string',
        'integer']` for a union — so what is left here is the one case its checker skips: a
        property that declares no `type` at all. No capability on this hub publishes one, but
        schemas arrive from whichever hub is federated in at runtime. Only the top level is
        pruned: `fermat.verify@v1`'s `potentials` accepts
        `additionalProperties: {"type": ["number", "null"]}`, and a null *inside* an opaque
        mapping is a legal value the caller meant to send.
        """
        properties = self._schema.get("properties") or {}
        open_keys = bool(self._schema.get("additionalProperties"))
        payload: dict[str, Any] = {}
        unknown: list[str] = []
        for key, value in arguments.items():
            if value is None:
                continue
            if not open_keys and properties and key not in properties:
                unknown.append(key)
                continue
            payload[key] = value
        if unknown:
            logger.warning(
                "%s: dropped %d argument(s) the capability does not declare (%s) — the model "
                "invented them, and sending them would fail the call after it was billed",
                self.name, len(unknown), ", ".join(sorted(unknown)),
            )
        return payload

    # ── what the model reads ─────────────────────────────────────────────────

    def _render(self, value: Any) -> Any:
        """Render the result for the model.

        The framework wraps this string in a text `Content` and that is what enters the
        conversation, so it is the one place that decides what a paid call costs in context.
        The default parser would JSON-encode the whole `CapabilityResult`, receipt included.
        """
        if not isinstance(value, CapabilityResult):
            # Not ours: fall back to the framework's own rendering rather than inventing one.
            return FunctionTool.parse_result(value)

        payload = value.for_model()
        if isinstance(payload, str):
            # Already a sentence (a refusal) or a plain string output — quoting it would only
            # add escaping for the model to read past.
            return payload
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(payload)


def aimarket_tools(
    base_url: str,
    *,
    intent: str = "",
    limit: int = 0,
    max_price_usd: float | None = None,
    free_only: bool = False,
    budget_usd: float = 1.0,
    include_price: bool = True,
    catalog_client: Any = None,
    **kw: Any,
) -> list[AIMarketTool]:
    """Every capability the hub offers, ready to hand to a `ChatAgent(tools=...)`.

    `intent`, `limit`, `max_price_usd` and `free_only` filter at build time, which is the only
    honest place: once a tool is in an agent's registry the agent decides when to call it, so a
    capability the operator cannot afford must never be handed over.

    `budget_usd` is the ceiling for the whole returned set — they share one `HubClient`, so
    spend is counted across every tool and every concurrent call, not per tool. 0 is honoured
    as "spend nothing" and `None` as "no ceiling"; only a NEGATIVE budget is refused, because
    it means nothing. The framework's own `max_invocations` counts calls, not dollars, and the
    two are not interchangeable on a catalogue whose prices differ by two orders of magnitude.

    Remaining keyword arguments go to `HubClient` (`timeout`, `verify_receipts`,
    `affiliate_id`, `agent`). `catalog_client` is an optional `httpx.Client` for the catalogue
    fetch only.
    """
    if budget_usd is not None and budget_usd < 0:
        raise ValueError(
            f"budget_usd must not be negative, got {budget_usd!r}. Use 0 to forbid every paid "
            "call, a positive number for a ceiling, or None for no ceiling."
        )

    caps = fetch_catalog(
        base_url,
        intent=intent,
        limit=limit,
        max_price_usd=max_price_usd,
        free_only=free_only,
        client=catalog_client,
        **({"timeout": kw["timeout"]} if "timeout" in kw else {}),
    )

    if not caps:
        # fetch_catalog raises CatalogError rather than returning [] when the hub is
        # unreachable, so an empty list here means the filters excluded everything.
        reason = "free_only=True, and no capability on this hub is free" if free_only else (
            f"max_price_usd={max_price_usd} excluded every capability"
            if max_price_usd is not None else f"the hub offers nothing matching intent={intent!r}"
            if intent else "the hub's manifest is empty"
        )
        logger.warning("no tools built from %s: %s", base_url, reason)
        # Returning before constructing a HubClient: with nothing to invoke there is no reason
        # to require the agent SDK to be installed.
        return []

    hub = HubClient(base_url, budget_usd=budget_usd, **kw)
    tools = []
    for cap in caps:
        try:
            tools.append(AIMarketTool(cap, hub, include_price=include_price))
        except Exception as exc:  # noqa: BLE001 - one bad schema must not lose the catalogue
            logger.warning(
                "skipping %s: its argument schema cannot be turned into an Agent Framework "
                "tool (%s: %s). The other %d capabilities are unaffected",
                cap.capability_id, type(exc).__name__, exc, len(caps) - 1,
            )
    logger.info(
        "built %d Agent Framework tools from %s, sharing a $%.2f budget",
        len(tools), base_url, budget_usd,
    )
    return tools
