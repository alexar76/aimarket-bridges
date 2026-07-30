"""Receipt verification against the key that actually signed it.

A federated receipt is signed by the capability's ORIGIN, not by the hub that routed the
call. The hub is a broker: it forwards the invoke, and what comes back carries the provider's
own Ed25519 signature so the buyer can check the work without trusting the middleman. That
is the whole point of the design — and it is why verifying against the hub's key is wrong.

Measured on the live hub, 2026-07-29:

    hub `modelmarket.dev`        signer_public_key sVjlCo52rBsmBH69iSXQ3oIB3LbWo4BgXT3iBhabDeM=
    origin `oracles.…/family`    signer_public_key YkAOwWNbRFti2cqEzD6zfuI4OTLsGUoObpCmlwZqaTQ=

42 of the 47 catalogue entries are federated, so verifying everything against the hub's key
reports `invalid-signature` for 89% of the catalogue — on receipts that are perfectly valid.
The same receipt verifies as `ok` against the origin's key. A false alarm on exactly the
promise the protocol is sold on is worse than no check at all: it teaches the reader that the
signal means nothing.

So the key is resolved per capability, from ``source_hub``, and cached. A capability served
by the hub itself is verified against the hub's own key, which is the case the reference
client happened to get right.

The SDK gained the same fix in ``aimarket-agent`` 2.2.0 (``receipts.OriginVerifiers``), so
this is no longer the only correct implementation — but it stays, for two reasons. The floor
here is ``>=2.1``, and 2.1.x is what is installed on any machine that has not upgraded, so
delegating would silently reintroduce the false alarm for those users. And this returns a
three-state answer: ``verified is None`` means "not checked", which the SDK's ``VerifyResult``
cannot express because its ``__bool__`` collapses "could not look" into the same False as
"the signature is wrong". That distinction is the whole point, so it is kept here.
"""

from __future__ import annotations

import logging
import threading
import time
from urllib.parse import urlsplit
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

__all__ = ["ReceiptCheck", "Expected", "OriginKeyResolver"]

WELL_KNOWN = "/.well-known/ai-market.json"


@dataclass(frozen=True)
class ReceiptCheck:
    """Outcome of checking one receipt, with the key it was checked against.

    ``verified is None`` means "not checked" — no key could be resolved, or the verifier
    dependency is absent. Deliberately distinct from False: "we could not look" and "the
    signature is wrong" call for completely different reactions, and collapsing them is how
    the false alarm above became invisible.
    """

    verified: bool | None
    reason: str = ""
    key: str = ""
    origin: str = ""
    #: What the receipt attests to, when that disagrees with what was invoked.
    attests_to: str = ""

    @property
    def trustworthy(self) -> bool:
        return self.verified is True


@dataclass(frozen=True)
class Expected:
    """What the receipt must attest to for it to be evidence about THIS call."""

    capability_id: str = ""
    product_id: str = ""
    price_usd: float | None = None


def _normalise_origin(origin: str) -> str:
    """A ``source_hub`` reduced to scheme://host[:port][/path], or "" if unusable.

    The fragment and query are DROPPED, and that is the security of this function. Appending
    a fixed suffix to a URL that may carry either hands the path to whoever wrote the URL —
    measured on the unfixed code against a raw TCP listener:

        source_hub = http://HOST/_cluster/health#          -> GET /_cluster/health
        source_hub = http://HOST/v1/secret/data/prod?list=true&
                                                           -> GET /v1/secret/data/prod?list=true&/.well-known/…

    The first is exact path control: the ``#`` sends everything after it to the fragment,
    which is never transmitted, so the suffix vanishes. After normalisation both become a
    request under ``/.well-known/ai-market.json`` and nothing else.

    Only http and https are accepted. httpx already refuses file:// and gopher:// (verified:
    zero connections), but relying on a transport's refusal for a security property means the
    property moves when the transport does.

    Deliberately NOT an address filter. Blocking loopback and private ranges would refuse the
    project's own documented deployments — docs/running.md health-checks the hub at
    http://localhost:9083/.well-known/ai-market.json, and core services reach each other as
    http://hub:9083 on a docker bridge inside 172.16/12 — which would silently downgrade every
    receipt in every self-hosted stack from verified to "unchecked". That is the false-signal
    failure this module exists to remove. `source_hub` is also hub-authored: the crawler
    overwrites whatever a peer claims with the URL it actually crawled, and screens that URL
    against its own SSRF guard before indexing.
    """
    raw = (origin or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if parsed.scheme not in ("http", "https"):
        logger.warning(
            "ignoring source_hub %r: only http and https are fetched for a signing key", raw
        )
        return ""
    if not parsed.netloc:
        return ""
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _binding_mismatch(receipt: dict[str, Any], expect: "Expected | None") -> str:
    """Why this receipt is not about the expected call, or "" when it is.

    Only fields the canonical SIGNS are compared. Checking an unsigned field would be
    security theatre: a relay can edit it freely without breaking the signature, so a
    disagreement there proves nothing and an agreement proves less.

    ``price_usd`` is compared with a tolerance because it crosses JSON as a float and the
    catalogue's own figure is a float too; half a hundredth of a cent is far below the
    cheapest capability ($0.001) and far above float noise.
    """
    if expect is None or not isinstance(receipt, dict):
        return ""
    problems: list[str] = []

    if expect.capability_id:
        got = str(receipt.get("capability_id") or "")
        if got != expect.capability_id:
            problems.append(f"capability_id is {got!r}, expected {expect.capability_id!r}")

    if expect.product_id:
        got = str(receipt.get("product_id") or "")
        # A hub may route a capability under a product id the catalogue did not name, so a
        # MISSING product_id is not a mismatch — a contradicting one is.
        if got and got != expect.product_id:
            problems.append(f"product_id is {got!r}, expected {expect.product_id!r}")

    if expect.price_usd is not None:
        try:
            got_price = float(receipt.get("price_usd", 0) or 0)
        except (TypeError, ValueError):
            problems.append(f"price_usd is not a number: {receipt.get('price_usd')!r}")
        else:
            if abs(got_price - float(expect.price_usd)) > 5e-6:
                problems.append(
                    f"price_usd is {got_price}, expected {float(expect.price_usd)}"
                )

    return "; ".join(problems)


class OriginKeyResolver:
    """Caches ``source_hub`` → signing key, and verifies receipts with the right one."""

    #: How long a FAILED lookup is remembered. Long enough that a keyless origin does not
    #: cost two HTTP round trips per receipt, short enough that a brief outage does not
    #: silently disable verification for the rest of the process's life.
    MISS_TTL_S = 60.0

    def __init__(
        self,
        hub_url: str,
        *,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ):
        self.hub_url = (hub_url or "").rstrip("/")
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._keys: dict[str, str] = {}
        # Origins whose lookup FAILED, with when. A miss is cached so a keyless origin does
        # not re-pay two failed round trips on every receipt — but only briefly: caching it
        # forever meant one 503 at the wrong moment disabled verification for that origin for
        # the life of the process, and nothing said so again after the first warning.
        # Measured before this: origin 503s once, then serves a valid key, and calls 2 and 3
        # still answer "no signing key published" having made zero further HTTP attempts.
        self._misses: dict[str, float] = {}
        self._lock = threading.Lock()

    # ── key resolution ───────────────────────────────────────────────────────

    def _http(self) -> httpx.Client:
        if self._client is None:
            # follow_redirects is OFF. A well-known document has no legitimate reason to
            # redirect, and following one let an allowlisted public origin 302-pivot anywhere
            # — reproduced: public peer -> 302 -> GET /latest/meta-data/iam/security-credentials/.
            # The SDK's session never followed them, so this was the bridge being the laxer of
            # the two for no reason anyone had chosen.
            self._client = httpx.Client(timeout=self._timeout, follow_redirects=False)
        return self._client

    def _candidate_urls(self, origin: str) -> list[str]:  # noqa: D401
        """Where an origin's well-known document might live.

        A federated ``source_hub`` is a base URL that may already include a path — the oracle
        family publishes at ``…/family``, and its document is at ``…/family/.well-known/…``,
        not at the domain root. Both are tried, deepest first, because the path-scoped one is
        the specific answer and the root is the fallback.
        """
        base = _normalise_origin(origin)
        if not base:
            return []
        urls = [f"{base}{WELL_KNOWN}"]
        if "//" in base:
            scheme, _, rest = base.partition("//")
            root = f"{scheme}//{rest.split('/', 1)[0]}"
            if root != base:
                urls.append(f"{root}{WELL_KNOWN}")
        return urls

    def key_for(self, source_hub: str) -> str:
        """Signing key for a capability's origin. Empty string if none can be found."""
        origin = (source_hub or "").strip()
        if not origin or origin == "local":
            origin = self.hub_url

        now = time.monotonic()
        with self._lock:
            if origin in self._keys:
                return self._keys[origin]
            failed_at = self._misses.get(origin)
            if failed_at is not None and now - failed_at < self.MISS_TTL_S:
                return ""

        key = ""
        for url in self._candidate_urls(origin):
            try:
                response = self._http().get(url)
                response.raise_for_status()
                document = response.json()
            except Exception as exc:
                logger.debug("no well-known at %s: %s", url, exc)
                continue
            if isinstance(document, dict):
                candidate = document.get("signer_public_key") or document.get("public_key")
                if isinstance(candidate, str) and candidate.strip():
                    key = candidate.strip()
                    break

        if not key:
            with self._lock:
                self._misses[origin] = now
            logger.warning(
                "no signing key published at %s — receipts from it cannot be verified, and "
                "will be reported as unchecked rather than as valid", origin,
            )
        with self._lock:
            if key:
                self._keys[origin] = key
                self._misses.pop(origin, None)
        return key

    # ── verification ─────────────────────────────────────────────────────────

    def check(
        self, receipt: Any, *, source_hub: str = "", expect: "Expected | None" = None
    ) -> ReceiptCheck:
        """Verify a receipt against its origin's key AND bind it to the call it is about.

        A valid signature answers "who signed this record". It does not answer "is this
        record about the call I just made", and those are different questions. Until the
        binding below existed, a provider could return a genuinely-signed receipt for a
        DIFFERENT, cheaper capability and this reported ``verified=True``: measured
        2026-07-30, buyer invoked skopos.security.posture@v1 at $0.15, receipt attested
        sortes.draw@v1 at $0.001, verified True, $0.15 billed, no mismatch reported. The
        signature was real. It was simply evidence about something else.

        A mismatch is ``verified=False``, not ``None``: this is a positive finding that the
        receipt is not evidence for this call, which is the opposite of "could not look".
        """
        if not isinstance(receipt, dict) or not receipt:
            return ReceiptCheck(None, "no receipt to check")

        origin = (source_hub or "").strip() or "local"
        key = self.key_for(source_hub)
        if not key:
            return ReceiptCheck(None, f"no signing key published by {origin}", origin=origin)

        try:
            from aimarket_agent import ReceiptVerifier
        except ImportError:  # pragma: no cover - packaging guard
            return ReceiptCheck(None, "aimarket-agent is not installed", key=key, origin=origin)

        try:
            result = ReceiptVerifier(key).verify(receipt)
        except Exception as exc:
            # An exception here is a verifier problem, not a bad signature. Saying "invalid"
            # would be the same false alarm this module exists to remove.
            return ReceiptCheck(None, f"verifier raised {type(exc).__name__}: {exc}",
                                key=key, origin=origin)

        if not result:
            return ReceiptCheck(
                False,
                str(getattr(result, "reason", "") or "invalid-signature"),
                key=key, origin=origin,
            )

        mismatch = _binding_mismatch(receipt, expect)
        if mismatch:
            return ReceiptCheck(
                False, f"signature valid but the receipt is about a different call: {mismatch}",
                key=key, origin=origin,
                attests_to=f"{receipt.get('capability_id', '?')} @ ${receipt.get('price_usd', 0)}",
            )

        return ReceiptCheck(True, "ok", key=key, origin=origin)

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
