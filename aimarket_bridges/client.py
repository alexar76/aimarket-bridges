"""One invoke path, shared by all three bridges.

The frameworks differ in how they declare a tool; they do not differ in what a tool call
needs to do, so the money, the refusals and the receipt live here once.

Two decisions shape everything below.

**A refusal is a RESULT, not an exception.** When a capability rejects its input — a missing
field, a value out of range — the right outcome is a message the calling model can read and
retry against. Raising instead aborts the surrounding graph or crew over something the model
could have fixed itself in one more turn. Transport and configuration failures do raise:
those the model cannot fix, and swallowing them produces an agent that reports success while
having called nothing.

**Provenance must not cost the model tokens it did not ask for.** Every invoke returns a
signed 7-field receipt, which is the point of the protocol — but pushing it into the tool's
text result would spend context on a blob no model reads. So the result the model sees is
the capability's own output, and the receipt is kept on the side, reachable through the
framework's own metadata channel and through `last_receipt`.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from aimarket_bridges.catalog import Capability
from aimarket_bridges.receipts import Expected, OriginKeyResolver

logger = logging.getLogger(__name__)

__all__ = ["InvokeResult", "HubClient", "BudgetExceeded", "HubUnavailable"]


def _visitor_id() -> str:
    """Trial identity for this installation, for the hub's free-trial allowance.

    The hub grants a few invokes per visitor keyed on ``X-AIMarket-Sandbox-Visitor`` (8-64
    characters of ``[A-Za-z0-9_-]``), which is how a bot gets to make its first calls before
    anyone has funded a payment channel. Without the header the allowance is not counted at
    all, so a bridge that omits it either gets whatever the hub happens to give an anonymous
    caller or a 402 on its very first call, depending on how the hub is configured.

    Random per PROCESS, and that is a deliberate limitation rather than an oversight: a stable
    id would have to be persisted, and this package writes nothing to disk on import. So a
    restart draws a fresh allowance, and the hub's own per-network rate limit is the real
    backstop — the trial is an on-ramp, not a security boundary. Set
    ``AIMARKET_SANDBOX_VISITOR`` to keep one allowance across restarts, which is also what an
    operator wants when several agents share a budget.
    """
    raw = "".join(
        c for c in (os.environ.get("AIMARKET_SANDBOX_VISITOR") or "") if c.isalnum() or c in "_-"
    )
    if len(raw) >= 8:
        return raw[:64]
    # Pad rather than refuse: a short override is rejected by the hub as an INVALID id, and
    # that refusal reads like a missing header to whoever set it.
    return (f"bridge-{raw}-{uuid.uuid4().hex[:12]}" if raw else f"bridge-{uuid.uuid4().hex[:12]}")[:64]


class HubUnavailable(RuntimeError):
    """The hub could not be reached, or answered something unusable."""


class BudgetExceeded(RuntimeError):
    """The configured spend ceiling for this bridge has been reached."""


@dataclass
class InvokeResult:
    """What one capability call produced.

    ``ok=False`` with ``error`` set is a refusal the model should read and retry against;
    it is not a failure of the bridge.
    """

    ok: bool
    output: Any = None
    error: str = ""
    #: True when the refusal is NOT about the arguments — payment, a spent trial, a safety
    #: block. The distinction changes what the model should do next, so it must reach the text.
    not_about_input: bool = False
    price_usd: float = 0.0
    receipt: dict[str, Any] | None = None
    receipt_verified: bool | None = None
    receipt_verify_reason: str = ""
    capability_id: str = ""

    def for_model(self) -> Any:
        """The value a framework should hand back to the calling model.

        A refusal becomes a plain sentence rather than a dict, because that is what a model
        acts on: told ``'count' must be an integer, got str`` it corrects the argument and
        calls again, whereas ``{"ok": false, ...}`` invites it to treat the envelope as data.
        """
        if self.ok:
            return self.output
        if self.not_about_input:
            # "refused this input" would send a model off rewriting arguments that were fine.
            # A payment state, a spent trial and a safety block are all things it cannot fix
            # that way, and saying so is the difference between one wasted turn and a loop.
            return f"{self.capability_id} cannot be called right now: {self.error}"
        return f"{self.capability_id} refused this input: {self.error}"


class HubClient:
    """Thin, thread-safe wrapper over the reference agent SDK.

    Thread-safety matters here and not in the SDK: LangGraph and CrewAI both run tool calls
    from worker threads, so the spend counter would otherwise lose increments under exactly
    the conditions that make a spend counter worth having.
    """

    def __init__(
        self,
        base_url: str,
        *,
        budget_usd: float | None = 1.0,
        timeout: float = 120.0,
        verify_receipts: bool = True,
        affiliate_id: str = "",
        agent: Any = None,
    ):
        self.base_url = (base_url or "").rstrip("/")
        if not self.base_url:
            raise HubUnavailable("base_url is required, e.g. https://modelmarket.dev")
        # None = no ceiling. 0 = spend nothing. Not coerced with float(), which
        # would turn None into a TypeError and 0 into the old footgun.
        self.budget_usd = None if budget_usd is None else float(budget_usd)
        self._spent = 0.0
        self._lock = threading.Lock()
        self.last_receipt: dict[str, Any] | None = None
        self._verify = bool(verify_receipts)
        self.visitor_id = _visitor_id()
        self._keys = OriginKeyResolver(self.base_url, timeout=timeout)

        if agent is not None:
            self._agent = agent
            return
        try:
            from aimarket_agent import AIMarketAgent
        except ImportError as exc:  # pragma: no cover - packaging guard
            raise HubUnavailable(
                "aimarket-agent is required: pip install aimarket-agent"
            ) from exc
        self._agent = AIMarketAgent(
            base_url=self.base_url,
            # The SDK's own budget only sizes its channel deposit; the ceiling that
            # actually refuses a call is ours, above.
            budget=self.budget_usd if self.budget_usd is not None else 0.0,
            timeout=timeout,
            affiliate_id=affiliate_id,
            # The SDK verifies every receipt against the HUB's key. For a federated
            # capability the signer is the origin, so that check answers
            # `invalid-signature` for 42 of the 47 live capabilities — on valid receipts.
            # Switched off here so there is exactly one answer, produced by
            # OriginKeyResolver against the key that actually signed.
            verify_receipts=False,
        )
        # The trial header rides on every request this agent makes. Set once on the session
        # rather than per call: the SDK owns the request, and a header it does not know about
        # is the one thing a caller can still contribute.
        session = getattr(self._agent, "session", None)
        if session is not None and hasattr(session, "headers"):
            session.headers["X-AIMarket-Sandbox-Visitor"] = self.visitor_id

    # ── spend ────────────────────────────────────────────────────────────────

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent

    @property
    def remaining_usd(self) -> float:
        """What is left. ``inf`` when no ceiling was set, so callers can compare numerically."""
        with self._lock:
            if self.budget_usd is None:
                return float("inf")
            return max(0.0, self.budget_usd - self._spent)

    def _reserve(self, price: float, capability_id: str) -> None:
        """Claim budget BEFORE the call, so a concurrent pair cannot both pass the check.

        ``budget_usd=0`` means SPEND NOTHING, and ``budget_usd=None`` means no ceiling.
        This used to read ``if self.budget_usd and …``, so a falsy budget skipped the check
        altogether: an operator writing 0 to mean "spend nothing" got unlimited spend, while
        ``remaining_usd`` reported $0.00 for the whole run. All three framework adapters had
        grown their own guard against it, which is the clearest possible sign the default
        belonged here rather than in each of them.
        """
        with self._lock:
            if self.budget_usd is None:
                self._spent += price
                return
            if self._spent + price > self.budget_usd:
                raise BudgetExceeded(
                    f"{capability_id} costs ${price:.4f} but only "
                    f"${max(0.0, self.budget_usd - self._spent):.4f} of the "
                    f"${self.budget_usd:.2f} budget is left"
                    + (" (budget_usd=0 forbids any paid call; pass None for no ceiling)"
                       if self.budget_usd == 0 else "")
                )
            self._spent += price

    def _refund(self, price: float) -> None:
        """Return the reservation when the call did not happen or was refused unbilled."""
        with self._lock:
            self._spent = max(0.0, self._spent - price)

    # ── invoke ───────────────────────────────────────────────────────────────

    def invoke(self, capability: Capability, arguments: dict[str, Any]) -> InvokeResult:
        """Call one capability. Raises only for what the model cannot fix."""
        price = 0.0 if capability.is_free else capability.price_usd
        if price:
            self._reserve(price, capability.capability_id)

        try:
            body = self._agent.invoke_single(
                product_id=capability.product_id,
                capability_id=capability.capability_id,
                input_payload=dict(arguments or {}),
                source_hub=capability.source_hub or "local",
            )
        except Exception as exc:
            if price:
                self._refund(price)
            raise HubUnavailable(
                f"invoking {capability.capability_id} at {self.base_url} failed: {exc}"
            ) from exc

        if not isinstance(body, dict):
            if price:
                self._refund(price)
            raise HubUnavailable(
                f"{capability.capability_id} returned {type(body).__name__}, expected an object"
            )

        # Payment required: the trial allowance is spent, or this capability was always paid
        # and no channel is funded. NOT an exception — an agent holding a mixed toolbox should
        # be able to keep working with what it can still call, and killing the graph over a
        # billing state the model cannot influence is the wrong trade. It IS logged at warning,
        # because the person who has to act on it is the operator, not the model.
        # `trial_quota_exhausted` is the SAME event from the buyer's side: the free allowance is
        # spent and from here on the capability costs money. The hub answers it 429 with its own
        # error code rather than 402, and handling only 402 left a bot reading the bare string
        # `trial_quota_exhausted` at precisely the moment it needed to be told what to do next.
        if (
            body.get("error") in ("payment_required", "trial_quota_exhausted")
            or body.get("payment_required")
        ):
            if price:
                self._refund(price)

            # A free-tier CEILING refusal wears the same `payment_required` code but is a
            # different event, and telling the two apart matters more than it looks. The
            # capabilities that sell computation cap what an unpaid caller may ask for, and
            # answer 402 carrying `free_tier: {field, requested, max}`. That refusal IS about
            # the input: lowering the field fixes it, and the model can do that itself. Falling
            # through to the branch below would tell it the opposite — "not something to retry
            # with different arguments, the operator has to fund a channel" — and a model that
            # believes it gives up on a call it could have made.
            free_tier = body.get("free_tier")
            if isinstance(free_tier, dict) and free_tier.get("max") is not None:
                field, ceiling = free_tier.get("field", "?"), free_tier.get("max")
                logger.info(
                    "%s refused an unpaid call above its free-tier ceiling: %s=%s, max %s. "
                    "This is a bounded free tier, not a billing failure — the same call fits "
                    "with %s<=%s.",
                    capability.capability_id, field, free_tier.get("requested", "?"),
                    ceiling, field, ceiling,
                )
                return InvokeResult(
                    ok=False, capability_id=capability.capability_id,
                    # Deliberately NOT not_about_input: the argument is the problem.
                    error=(
                        f"'{field}' is {free_tier.get('requested', '?')}, above the free-tier "
                        f"ceiling of {ceiling} for {capability.capability_id}. This capability "
                        f"sells computation, so unpaid calls are bounded. Retry with "
                        f"'{field}' at most {ceiling} — that works right now — or fund a "
                        f"payment channel for the full range."
                    ),
                )
            needed = body.get("needed", capability.price_usd)
            trial = body.get("sandbox") if isinstance(body.get("sandbox"), dict) else {}
            exhausted = body.get("error") == "trial_quota_exhausted"
            logger.warning(
                "%s now requires payment: %s. The hub wants $%.4f per call and no funded "
                "payment channel is available%s. Fund one at %s/ai-market/v2/channel/open with "
                "a verified on-chain deposit, or set AIMARKET_SANDBOX_VISITOR to an identity "
                "that still has trial allowance",
                capability.capability_id,
                "the free-trial allowance for visitor %r is spent (%s of %s used)" % (
                    self.visitor_id, trial.get("used", "?"), trial.get("max_trials", "?")
                ) if exhausted else (body.get("detail") or "no detail given"),
                float(needed or 0),
                "" if exhausted else (
                    " and the trial allowance was not counted (no visitor id reached the hub)"
                    if not trial else ""
                ),
                self.base_url,
            )
            return InvokeResult(
                ok=False, capability_id=capability.capability_id, not_about_input=True,
                error=(
                    (
                        f"the free trial for this capability is used up "
                        f"({trial.get('used', '?')} of {trial.get('max_trials', '?')} calls) and "
                        f"it now costs ${float(needed or 0):.4f} per call"
                        if exhausted else
                        f"requires payment: ${float(needed or 0):.4f} per call"
                    )
                    + ", and this client has no funded payment channel. Not something to retry "
                      "with different arguments — the operator has to fund one"
                ),
            )

        # The hub's safety plugin blocks before the provider runs, so nothing was billed.
        if body.get("safety_blocked"):
            if price:
                self._refund(price)
            return InvokeResult(
                ok=False, capability_id=capability.capability_id, not_about_input=True,
                error=f"blocked by the hub's safety gate: {body.get('reason') or body.get('error') or 'no reason given'}",
            )

        receipt = body.get("receipt") if isinstance(body.get("receipt"), dict) else None
        if receipt:
            self.last_receipt = receipt

        # A capability refusal. `ok` is absent on some paths, so an explicit error field is
        # treated as a refusal too rather than being reported as a successful empty result.
        error = body.get("error")
        if body.get("ok") is False or (error and "output" not in body and "result" not in body):
            if price and not receipt:
                # No receipt means nothing was metered, so the reservation is released. With
                # a receipt the call WAS billed even though it refused, and pretending
                # otherwise would let a loop of refusals spend without ever showing it.
                self._refund(price)
            return InvokeResult(
                ok=False, capability_id=capability.capability_id,
                error=str(error or "refused without an explanation"),
                price_usd=price if receipt else 0.0, receipt=receipt,
            )

        # `output` is the v2 field; `result` appears on older hubs and some proxies.
        output = body.get("output", body.get("result"))
        check = (
            self._keys.check(
                receipt,
                source_hub=capability.source_hub,
                # Bind the receipt to THIS call. A valid signature says who signed a record;
                # it does not say the record is about the call just made. Without this a
                # provider could answer a $0.15 invoke with a genuinely-signed receipt for a
                # $0.001 capability and the bridge reported verified=True.
                expect=Expected(
                    capability_id=capability.capability_id,
                    product_id=capability.product_id,
                    price_usd=price if price else None,
                ),
            )
            if self._verify and receipt
            else None
        )
        return InvokeResult(
            ok=True,
            output=output,
            price_usd=price,
            receipt=receipt,
            receipt_verified=check.verified if check else None,
            receipt_verify_reason=check.reason if check else "",
            capability_id=capability.capability_id,
        )

    def close(self) -> None:
        self._keys.close()
        closer = getattr(self._agent, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> HubClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
