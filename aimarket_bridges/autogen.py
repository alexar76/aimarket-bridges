"""A live AIMarket hub's catalogue, as AutoGen tools.

Built and verified against **autogen-core 0.7.5**, **autogen-agentchat 0.7.5** and
**pydantic 2.13.4** (introspected on 2026-07-29 — every claim below was measured against the
installed packages, not read from documentation, which is behind the code in several places).

Four framework facts shaped this module.

**`BaseTool`, not `FunctionTool`.** `FunctionTool(func, description, name=None, ...)` derives
its argument schema from the *type annotations* of a Python function. A federated capability's
interface arrives as JSON Schema at runtime, so using `FunctionTool` would mean synthesising a
function with annotations built from that schema — a second code-generation step whose only
purpose is to be read back off `__annotations__`. `BaseTool.__init__(self, args_type,
return_type, name, description, strict=False)` takes the model directly, and
`Capability.args_model()` already produces one. Its sole abstract method is `run`.

**`run` is async, `HubClient.invoke` is not.** `BaseTool.run` is a coroutine
(`inspect.iscoroutinefunction(BaseTool.run) is True`) and `HubClient.invoke` does blocking
HTTP with a 120s default timeout. Calling it directly inside `run` would hold the event loop
for the whole invoke, which in AutoGen means the entire agent runtime: one paid call would
freeze every other agent, every concurrent tool call and the token stream. So the invoke is
handed to a worker thread with `asyncio.to_thread`. `HubClient` is documented thread-safe and
locks its spend counter, which is exactly the property this offload requires.

**Cancellation is honoured as far as it can be, and no further.** `CancellationToken` in 0.7.5
offers `is_cancelled()`, `cancel()`, `add_callback()` and `link_future(future)`, and
`link_future` does one thing: `future.cancel()`. Measured behaviour of cancelling a linked
`to_thread` task — the `await` returns in 0.000s, and the worker thread then runs to
completion regardless. Python cannot kill a thread, and `HubClient.invoke` is already inside a
blocking socket read by then. So:

  * cancelled *before* dispatch — the call never happens and nothing is spent. This is the
    case worth checking, because it is the one where cancellation saves money.
  * cancelled *mid-flight* — the caller stops waiting promptly, but the HTTP request completes
    at the hub, the capability runs, and the operator is billed. Pretending otherwise would be
    a lie about money. Spend accounting stays correct anyway: the reservation and the receipt
    are handled inside `HubClient.invoke`, which finishes in the worker thread.

Neither `add_callback` nor `link_future` can be undone — `_callbacks` only grows — and a token
is not per call: one is reused for every tool call in an agent run and threaded through every
turn of a team. So the registration this module makes is one it can withdraw itself, and it
does, as soon as the call is over. See `_forget_cancel_callback`.

**The return type must be a pydantic model.** Not a preference —
`ReturnT = TypeVar("ReturnT", bound=BaseModel, covariant=True)`, so a bare `dict` violates the
declared bound. It is also better behaved: `BaseTool.return_value_as_string` special-cases
`BaseModel` to `json.dumps(value.model_dump())` and falls through to `str(value)` for
everything else, so returning a dict would show the model `{'ok': True, ...}` — Python repr,
single-quoted, not JSON. `StaticWorkbench.call_tool` calls `return_value_as_string` to build
the text the model reads, which makes it the right place to keep the receipt *out* of the
model's context while leaving it reachable on the returned object.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent import futures
import json
import logging
from typing import Any

from autogen_core import CancellationToken
from autogen_core.tools import BaseTool
from pydantic import BaseModel

from aimarket_bridges.catalog import Capability, fetch_catalog
from aimarket_bridges.client import BudgetExceeded, HubClient, InvokeResult

logger = logging.getLogger(__name__)

#: Threads shared by every AIMarket tool in this process. Deliberately a DEDICATED pool.
#: ``asyncio.to_thread`` uses the loop's DEFAULT executor, which every other library shares:
#: a hub that accepts a connection and then trickles bytes holds its worker for the full
#: timeout (120s by default), and enough concurrent tool calls would starve every unrelated
#: ``to_thread`` user in the process — a file read, a DNS lookup, somebody else's SDK. Bounded
#: rather than unbounded for the mirror reason: an agent that fans out is not entitled to one
#: thread per call.
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



__all__ = ["CapabilityResult", "AIMarketTool", "aimarket_tools"]


def _forget_cancel_callback(token: CancellationToken | None, callback: Any) -> None:
    """Unregister a finished call's cancel callback from the token.

    `add_callback` has no counterpart in 0.7.5: `_callbacks` only grows, and every entry closes
    over the task it would cancel — which holds that call's `CapabilityResult`, receipt
    included. A token outlives one call by a long way: `AssistantAgent` reuses one for every
    tool call in a run, and a team threads a single one through every agent turn, so entries
    left behind retain every paid result of the session. That is both an unbounded leak and the
    opposite of this module's one rule about provenance — keep it reachable, do not carry it
    around.

    Reaching into the private list is the price of the missing API, so it is guarded: if a
    future version renames either attribute this degrades to the old growth instead of raising
    in the middle of a paid call.
    """
    if token is None or callback is None:
        return
    try:
        with token._lock:
            token._callbacks.remove(callback)
    except (AttributeError, ValueError):
        # Renamed internals, or `add_callback` fired it immediately instead of storing it.
        pass


class CapabilityResult(BaseModel):
    """One capability call, as an AutoGen `ReturnT`.

    Mirrors `InvokeResult` rather than wrapping it: `BaseTool` requires a `BaseModel` return
    and a dataclass is not one. The provenance fields are carried here so a caller can reach
    them off the value `run` returns, but they are deliberately kept out of
    `return_value_as_string` — see `AIMarketTool.return_value_as_string`.
    """

    ok: bool
    output: Any = None
    error: str = ""
    capability_id: str = ""
    price_usd: float = 0.0
    receipt: dict[str, Any] | None = None
    # None means "not checked" — no key published, or no verifier installed. Distinct from
    # False, which means the signature did not verify. The core keeps these apart on purpose.
    receipt_verified: bool | None = None
    receipt_verify_reason: str = ""
    # Set instead of raising BudgetExceeded, so an orchestrator can detect an exhausted
    # ceiling without matching on the message text.
    budget_exceeded: bool = False
    cancelled: bool = False

    def for_model(self) -> Any:
        """What the calling model should read. Same contract as `InvokeResult.for_model`."""
        if self.ok:
            return self.output
        if self.cancelled or self.budget_exceeded:
            # Neither reached the capability, so "refused this input" would be a lie that
            # sends the model off rewriting arguments that were never the problem.
            return f"{self.capability_id} was not called: {self.error}"
        return f"{self.capability_id} refused this input: {self.error}"

    @classmethod
    def from_invoke(cls, result: InvokeResult) -> CapabilityResult:
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


class AIMarketTool(BaseTool[BaseModel, CapabilityResult]):
    """One hub capability as an AutoGen tool."""

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

        if capability.schema_gaps:
            # The declared arguments are looser than the capability's real contract, so the
            # model may be told an input is valid that the capability will refuse. Said once,
            # at build time, where it is actionable.
            logger.warning(
                "%s: argument schema keywords not modelled (%s) — the tool advertises a "
                "looser interface than the capability enforces",
                capability.tool_name, ", ".join(capability.schema_gaps),
            )

        super().__init__(
            args_type=capability.args_model(),
            return_type=CapabilityResult,
            name=capability.tool_name,
            description=capability.tool_description(include_price=include_price),
            # strict stays off. Under strict, `BaseTool.schema` raises unless every property
            # is required — measured across the live catalogue, 31 of 47 capabilities have at
            # least one optional property, so strict would make two thirds of the hub
            # unusable in exchange for a guarantee no capability here needs.
            strict=False,
        )

    # ── invocation ───────────────────────────────────────────────────────────

    async def run(
        self,
        args: BaseModel,
        cancellation_token: CancellationToken | None = None,
    ) -> CapabilityResult:
        """Invoke the capability without blocking the event loop.

        `BaseTool.run` declares `cancellation_token` as required and positional; it is
        optional here purely so the tool can be called directly in tests and scripts.
        Widening a parameter is substitutable, so `run_json` — which always passes a token —
        is unaffected.
        """
        cid = self.capability.capability_id

        # Cheap and load-bearing. `StaticWorkbench.call_tool` only mints a fresh token when its
        # caller passes none; `AssistantAgent` passes ONE token into an `asyncio.gather` over
        # every tool call in the turn (measured in `_execute_tool_calls`), so cancelling the
        # agent hands an already-cancelled token to each queued call. Returning here means the
        # rest of that fan-out costs nothing — the only point in the lifecycle where
        # cancellation can actually prevent a charge.
        if cancellation_token is not None and cancellation_token.is_cancelled():
            result = CapabilityResult(
                ok=False, capability_id=cid, cancelled=True,
                error="cancelled before the call was dispatched, so nothing was spent",
            )
            self.last_result = result
            return result

        payload = self._payload(args)

        # The offload. `HubClient.invoke` blocks on HTTP; running it on the loop would stall
        # the whole AutoGen runtime for up to the client timeout.
        loop = asyncio.get_running_loop()
        call = asyncio.ensure_future(
            loop.run_in_executor(_executor(), self.hub.invoke, self.capability, payload)
        )

        cancel_call: Any = None
        if cancellation_token is not None:
            # Lets a cancel stop the *waiting*. It cannot stop the request: the thread is
            # inside a blocking read and will finish, and the hub will bill for it. Linking is
            # still right — an agent shutting down should not hang for the full timeout — but
            # it buys responsiveness, not a refund.
            #
            # `add_callback(call.cancel)` rather than `link_future(call)` only so the
            # registration is a handle this module still holds and can withdraw in the
            # `finally` — see `_forget_cancel_callback`. Behaviour is otherwise identical:
            # `link_future` stores a closure around this same call, and both fire immediately
            # if the token is already cancelled, which matters because `cancel()` can arrive
            # from another thread after the check above.
            cancel_call = call.cancel
            cancellation_token.add_callback(cancel_call)

        try:
            invoked = await call
        except asyncio.CancelledError:
            # The await returned, but the worker thread is inside a blocking read and will run
            # to completion: the capability answers and the operator is billed. So there is no
            # result for THIS call — and leaving the previous call's result on `last_result`
            # would hand a caller a stale receipt for a call that produced a different one.
            # Spend and the receipt itself stay correct on the shared HubClient.
            self.last_result = None
            raise
        except BudgetExceeded as exc:
            # Returned as text, not raised. Hitting a spend ceiling is the guard rail working
            # as designed, not an exceptional condition — an agent handed a $1 budget and 47
            # paid tools will reach it on a normal run. Raising buys nothing here either:
            # StaticWorkbench.call_tool catches Exception and shows the model
            # `_format_errors(exc)`, which is the same sentence with a worse frame around it.
            # The model can read this and stop, which is the correct response.
            result = CapabilityResult(
                ok=False, capability_id=cid, budget_exceeded=True, error=str(exc),
            )
            self.last_result = result
            return result
        finally:
            _forget_cancel_callback(cancellation_token, cancel_call)

        # HubUnavailable and anything else unexpected propagate. Transport and configuration
        # failures are not things the model can fix by retrying with different arguments, and
        # AutoGen has a channel for exactly this that the other two frameworks lack:
        # call_tool turns a raise into `ToolResult(is_error=True)`, so "the hub is down" stays
        # distinguishable from "the capability answered". Swallowing it into a normal result
        # would let the model narrate a successful answer over a call that never happened.
        #
        # asyncio.CancelledError also propagates, deliberately: it is BaseException, so
        # call_tool does not catch it, and cancellation should unwind the task rather than be
        # reported as a tool result.

        result = CapabilityResult.from_invoke(invoked)
        if result.receipt_verified is False:
            logger.warning(
                "%s: receipt did not verify against its origin key (%s)",
                cid, result.receipt_verify_reason or "no reason given",
            )
        self.last_result = result
        return result

    def _payload(self, args: BaseModel) -> dict[str, Any]:
        """The invoke body for a validated args model.

        Two corrections, both measured against the 47 live schemas.

        **`None` values are dropped, at every depth.** `model_from_schema` gives every optional
        property a default, and a property with no schema default defaults to `None` — so a
        model that omits an optional argument still produces a key holding `None`. Sending that
        is not the same as omitting it: not one of the 47 declares a nullable PROPERTY, so every
        such key is an argument the capability rejects on a call already billed. `exclude_none`
        rather than a top-level filter, because the nested models matter more than the top
        level: a dict-shaped `fermat.route@v1` edge has eleven optional keys, so a two-key edge
        would otherwise travel with nine nulls in it. `exclude_none` is also the right
        granularity rather than a recursive strip — it prunes model FIELDS, and leaves values
        inside an opaque mapping alone, which `fermat.verify@v1`'s `potentials`
        (`additionalProperties: {"type": ["number", "null"]}`, the one place null is legal)
        needs.

        **`by_alias`, so a rewritten name goes back as the capability's own.** `_safe_name` has
        to produce legal identifiers; `model_from_schema` records the original as the field's
        alias. Dumping without it sends `lambda_` to a capability that requires `lambda`, and
        turns a dict edge's `from` into a `from_` the router silently reads as a missing node.
        """
        if isinstance(args, type) or not hasattr(args, "model_dump"):
            # autogen builds the instance itself in `run_json`, so this only fires for a
            # hand-written integration. Without it the failure is
            # `TypeError: BaseModel.model_dump() missing 1 required positional argument: 'self'`
            # raised deep inside this method — which is what passing `tool.args_type()`
            # produces, since in autogen-core `args_type` is a METHOD returning the CLASS, not
            # a constructor. Easy to write, and the traceback points at the wrong place.
            raise TypeError(
                f"{self.name}: run() takes an instance of {self.args_type().__name__}, got "
                f"{args if isinstance(args, type) else type(args).__name__}. Build it with "
                f"tool.args_type()(**kwargs), or call tool.run_json(dict, token) — which is "
                f"the entry point autogen itself uses."
            )
        return args.model_dump(mode="json", by_alias=True, exclude_none=True)

    # ── what the model reads ─────────────────────────────────────────────────

    def return_value_as_string(self, value: Any) -> str:
        """Render the result for the model.

        `StaticWorkbench.call_tool` uses this for the text it puts in the conversation, so it
        is the one place that decides what a paid call costs in context. The default would
        `json.dumps` the entire `CapabilityResult`, pushing the receipt, the verification
        reason and the price into every tool message — a provenance blob no model reads, on
        every call. Only `for_model()` goes out; the rest stays on the returned object and on
        `last_result`.
        """
        if not isinstance(value, CapabilityResult):
            return super().return_value_as_string(value)

        payload = value.for_model()
        if isinstance(payload, str):
            # Already a sentence (a refusal) or a plain string output — quoting it would just
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
    """Every capability the hub offers, ready to hand to an `AssistantAgent`.

    `intent`, `limit`, `max_price_usd` and `free_only` filter at build time, which is the only
    honest place: once a tool is in an agent's registry the agent decides when to call it, so a
    capability the operator cannot afford must never be handed over.

    `budget_usd` is the ceiling for the whole returned set — they share one `HubClient`, so
    spend is counted across every tool and every concurrent call, not per tool. 0 is
    honoured as "spend nothing" and `None` as "no ceiling"; only a NEGATIVE budget is refused,
    because it means nothing.

    This used to refuse 0 as well, and the reason was real: `HubClient._reserve` tested
    `if self.budget_usd and ...`, so a falsy budget skipped the check and every paid call went
    through while `remaining_usd` reported $0.00 (measured: 200 calls, $1.20 spent, ceiling
    "0"). All three adapters had independently grown the same guard, which is what finally made
    it obvious the defect was in the core rather than in each bridge. Fixed there. What is left
    here is the one value that cannot be honoured either way.

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
        # unreachable, so an empty list here means the filters excluded everything. Worth
        # naming: no capability on the live hub is free, so free_only=True is a silent
        # zero-tool build that looks identical to a broken hub from the agent's side.
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
        except Exception as exc:  # noqa: BLE001 - see the note below
            logger.warning(
                "skipping %s: its argument schema cannot be turned into an autogen tool "
                "(%s: %s). The other %d capabilities are unaffected",
                cap.capability_id, type(exc).__name__, exc, len(caps) - 1,
            )
    logger.info(
        "built %d AutoGen tools from %s, sharing a $%.2f budget",
        len(tools), base_url, budget_usd,
    )
    return tools
