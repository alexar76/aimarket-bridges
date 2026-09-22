"""AIMarket capabilities as CrewAI tools.

Built and measured against **crewai 1.15.8** with pydantic 2.12.5 (crewai pins pydantic
down, so this is the version that matters), 2026-07-29. Five things about that version
shaped the code below, and all five were measured rather than assumed:

1. ``crewai.tools.BaseTool`` **is itself a pydantic BaseModel**, with no ``extra="allow"``.
   Anything a tool needs to carry — the capability, the hub client — has to be a declared
   field or pydantic refuses the assignment. Its config does set
   ``arbitrary_types_allowed``, so :class:`HubClient` is storable as-is, and a frozen
   dataclass field keeps its identity (no silent re-construction): verified.

2. ``args_schema`` must be a pydantic **model** — but not for the reason the older note here
   gave. 1.15.8 no longer refuses a raw JSON Schema dict outright: the ``args_schema``
   validator feeds it to ``create_model_from_schema``. That converter is the problem. It
   raises ``ValueError: Unsupported JSON schema type: ['string', 'integer']`` on the union
   types this catalogue is full of — **10 of the 47 live capabilities cannot be built from
   their raw schema at all** — and where it does succeed it names fields by its own rule, so
   :meth:`AIMarketTool._payload`, which inverts ``schema._safe_name``, would be inverting a
   rule that no longer applied. :meth:`Capability.args_model` is the only source that both
   builds for all 47 and names fields the way this module can undo.

3. **crewai fills in every optional argument before calling ``_run``, and it does it with a
   bare ``model_dump()``.** Both call paths (``BaseTool._validate_kwargs`` and
   ``CrewStructuredTool._parse_args``) validate against ``args_schema`` and then pass
   ``validated.model_dump()`` — no ``by_alias=True``, no ``exclude_none``. Two consequences,
   and both reach the whole nested body, not just its top-level keys: an argument the model
   never mentioned arrives as an explicit ``None`` (``{"ecosystem": null}`` where the manifest
   says ``string``, which buys a refusal), and a property whose name the args model had to
   rewrite arrives under the rewritten name rather than its alias (``from_`` inside every
   ``fermat.route@v1`` edge). :meth:`AIMarketTool._payload` therefore walks the value it is
   handed and undoes both at every depth.

4. ``description`` is **no longer rewritten** at registration. In earlier versions
   ``_generate_description`` overwrote the field with a ``Tool Name/Tool Arguments/Tool
   Description`` composite; in 1.15.8 that hook is an explicit no-op and the composite lives
   on ``formatted_description``. So ``Capability.tool_description()`` survives verbatim in
   the field *and* survives composition into the prompt — measured both ways, because a
   silently rewritten description is how the price disappears from what the model reads.

5. **An exception out of ``_run`` costs SIX hub calls, not one.** ``ToolUsage._use`` (and
   ``_ause``) wrap the invoke in ``try: tool.invoke(...) except Exception: tool.invoke(...)``
   — the fallback is meant for the argument-filtering step above it, but it re-invokes the
   tool on *any* failure — and the enclosing handler then retries the whole attempt until
   ``_run_attempts > _max_parsing_attempts`` (3). Measured: one raising ``_run`` reached the
   stubbed hub 6 times. Every one of those is a real request against a paid endpoint, and
   because the bridge refunds a reservation whose call did not return, ``spent_usd`` reads
   ``$0.00`` while the hub saw six invokes. That is why :meth:`AIMarketTool._run` answers in
   text and lets nothing escape.

The other framework-forced shape is ``cache_function``, which is discussed at
:func:`never_cache`: it must be a module-level named function, and for a marketplace it must
say no.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import httpx
from crewai.tools import BaseTool
from crewai.utilities.string_utils import sanitize_tool_name
from pydantic import Field

from aimarket_bridges.catalog import CatalogError, Capability, fetch_catalog
from aimarket_bridges.client import BudgetExceeded, HubClient, HubUnavailable, InvokeResult

# _safe_name is the exact rule model_from_schema used to turn a JSON property name into a
# pydantic field name, and this module has to invert it (see _payload). Importing the private
# helper is deliberate: re-implementing the rule here would let the two copies drift, and the
# symptom of drift is a paid call that silently omits a required argument.
from aimarket_bridges.schema import _safe_name

logger = logging.getLogger(__name__)

__all__ = ["AIMarketTool", "aimarket_tools", "never_cache"]


def never_cache(_arguments: Any = None, _result: Any = None) -> bool:
    """Refuse to cache a marketplace call. This is the default, and it is deliberate.

    Caching is opt-in per crew in 1.15.8 (``Crew(cache=False)`` is the default), but once it
    is on, crewai's OWN default — ``base_tool._default_cache_function`` — returns True for
    every tool. So this function is the difference between one draw and one draw replayed:
    measured through ``execute_single_native_tool_call`` with a real ``CacheHandler``, two
    identical calls reach the hub twice with ``never_cache`` and **once** with crewai's
    default, the second answer coming back ``from_cache``.

    The cache is keyed on ``tool name + json.dumps(arguments)``: on a hit the tool is
    never invoked and the previous output is replayed. For a catalogue of paid capabilities
    that is wrong in both directions. ``sortes.draw@v1`` and ``platon.random@v1`` return fresh
    verifiable randomness — a replayed draw is the *same number sold twice*, with a receipt
    for one call, and any downstream commit-reveal built on it is broken. The same applies to
    everything time-varying in the live catalogue: ``skopos.security.posture@v1``,
    ``platon.beacon@v1``, ``gaia`` sensor reads. Nothing in a manifest lets a bridge prove a
    capability is pure, so the safe default is the honest one: every call reaches the hub, the
    provider is paid for what the agent consumes, and the agent gets a current answer.

    An operator who *knows* a specific capability is deterministic can pass their own
    ``cache_function`` to :func:`aimarket_tools`. It must be a **module-level named
    function**: the field is typed ``SerializableCallable``, and crewai emits a UserWarning
    ("cannot be serialized and will prevent checkpointing") for a lambda or closure. That is
    also why this is a module-level function and not the obvious one-line lambda.
    """
    return False


def _branches(spec: Any) -> list[dict[str, Any]]:
    """A spec and its ``oneOf``/``anyOf`` branches, as the nodes that jointly declare it.

    Branches are merged rather than matched. Picking the one branch a value satisfies would
    mean re-running JSON Schema validation here, and getting it wrong would drop a real
    argument; merging can only ever be too permissive, and too permissive costs a refusal
    message from the provider rather than a silently mangled body. A ``fermat.route@v1`` edge
    is the live case — array branch OR object branch — and only the object branch declares
    anything.
    """
    if not isinstance(spec, dict):
        return []
    return [spec] + [
        b for b in (spec.get("oneOf") or spec.get("anyOf") or []) if isinstance(b, dict)
    ]


def _declared(spec: Any) -> tuple[dict[str, Any], set[str]]:
    """The ``(properties, required)`` an object spec declares, merging its union branches."""
    properties: dict[str, Any] = {}
    required: set[str] = set()
    for node in _branches(spec):
        for prop, sub in (node.get("properties") or {}).items():
            properties.setdefault(prop, sub)
        required.update(node.get("required") or [])
    return properties, required


def _item_spec(spec: Any) -> Any:
    """The element schema of an array, looking inside union branches for it.

    ``items`` can sit on a branch rather than on the node itself — a ``kantor.transport@v1``
    point is ``oneOf: [{type: array, items: {...}}, {type: number}]`` — and reading only
    ``spec["items"]`` there returns None, which stops :func:`_restore` from descending. No
    live capability has a rename below such a branch today (the three that need one are
    ``fourier.verify@v1``'s ``lambda`` and the ``from`` inside a fermat edge, both reachable
    without this), so this is the guard rather than the cure: the day one appears, the failure
    would be a paid call whose nested key went out spelled ``from_``.
    """
    for node in _branches(spec):
        items = node.get("items")
        if items is not None:
            return items
    return None


def _restore(value: Any, spec: Any) -> Any:
    """Undo, at every depth, what crewai's validation layer did to one argument value.

    Field names go back to property names and pydantic-inserted nulls come out. See
    :meth:`AIMarketTool._payload` for why both are necessary and why neither can be left to
    the args model.
    """
    if isinstance(value, list):
        items = _item_spec(spec)
        return [_restore(item, items) for item in value]
    if not isinstance(value, dict):
        return value

    properties, required = _declared(spec)
    if not properties:
        # A free-form object (``{"type": "object"}`` with no properties becomes
        # ``dict[str, Any]``). pydantic neither renames nor fills anything inside it, so any
        # null in there is the model's own and dropping it would edit the agent's argument.
        return dict(value)

    renamed = {_safe_name(prop): prop for prop in properties}
    body: dict[str, Any] = {}
    for key, item in value.items():
        prop = renamed.get(key, key)
        if item is None and prop not in required:
            continue
        body[prop] = _restore(item, properties.get(prop))
    return body


class AIMarketTool(BaseTool):
    """One AIMarket capability, as a CrewAI tool.

    A named class rather than a closure factory, because crewai reaches for the tool object
    in ways a closure cannot serve: both executors look ``cache_function`` up on the ORIGINAL
    tool behind the structured one (``_original_tool``), ``format_output_for_agent`` is
    discovered with ``inspect.getattr_static`` on the class, and ``BaseTool.__init_subclass__``
    registers the type for checkpoint deserialization. It also gives the operator somewhere to
    read the receipt from after a call.
    """

    # Declared fields, because pydantic will not accept undeclared instance attributes. The
    # capability is per-tool; the client is shared, so the spend ceiling is a ceiling for the
    # whole toolset rather than one per tool.
    capability: Capability
    # exclude=True on the two non-data fields: crewai serialises tools for checkpointing with
    # model_dump(mode="json"), and a live budgeted HTTP client is not JSON. Excluding it keeps
    # a checkpoint dump from raising; restoring one still needs a client passed back in, which
    # is the honest outcome — a resumed run has to be told what budget it may spend.
    # repr=False as well as exclude=True. `exclude` keeps a field out of model_dump; it does
    # NOT keep it out of `repr()`, and pydantic's default repr prints every field. So the
    # signed receipt — and the client, whose agent holds the hub URL and an affiliate id —
    # appeared in any log line, traceback frame or debugger view that touched the tool or an
    # Agent holding it. A receipt is not a credential, but it is provenance about a paid call
    # and it has no business in a log nobody asked to have it in.
    client: HubClient = Field(exclude=True, repr=False)
    last_result: InvokeResult | None = Field(default=None, exclude=True, repr=False)

    @classmethod
    def for_capability(
        cls,
        capability: Capability,
        client: HubClient,
        *,
        name: str = "",
        cache_function: Callable[..., bool] = never_cache,
    ) -> AIMarketTool:
        """Build the tool for one capability, with its own argument model.

        There is deliberately no ``result_as_answer`` argument. crewai reads that flag off the
        tool AFTER ``_run`` returns and, when set, wraps whatever came back in an
        ``AgentFinish`` that ends the task — measured: with ``result_as_answer=True`` a
        refusing stub made ``sortes.draw@v1 refused this input: bad alpha`` the crew's final
        answer, and the same route would promote the spend-limit stop, which is an instruction
        addressed to the model, into the deliverable. It also removes the retry that the
        refusal-as-text contract in :mod:`aimarket_bridges.client` exists for: the model never
        gets the turn in which it would fix the argument. A capability's output can still be a
        crew's answer — by being what the task asks the agent to report.
        """
        if capability.schema_gaps:
            # The declared arguments are looser than the capability's real contract, so the
            # model may be told an input is acceptable that the provider will reject — after
            # it has been billed. Worth saying at build time, once, not at call time.
            logger.warning(
                "%s: argument schema keywords %s are not modelled, so this tool advertises a "
                "looser interface than the capability enforces",
                capability.capability_id, ", ".join(capability.schema_gaps),
            )
        return cls(
            name=name or capability.tool_name,
            description=capability.tool_description(),
            args_schema=capability.args_model(),
            capability=capability,
            client=client,
            cache_function=cache_function,
            result_as_answer=False,
        )

    # ── invocation ───────────────────────────────────────────────────────────

    def _payload(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        """The JSON body for this call, from whatever crewai handed ``_run``.

        Two corrections happen here, both measured against the live catalogue, and both have
        to be applied at EVERY level of the body rather than only to its top-level keys —
        crewai validates the whole nested structure, so it damages the whole nested structure.

        *Field names are mapped back to property names.* ``model_from_schema`` has to produce
        legal Python identifiers, so ``fourier.verify@v1``'s required ``lambda`` becomes the
        field ``lambda_``, and the ``from`` key inside a ``fermat.route@v1`` edge object
        becomes ``from_``. The args model carries an alias, but crewai serialises with a bare
        ``model_dump()`` — no ``by_alias=True`` — so what arrives at ``_run`` is the field
        spelling at every depth, and the alias only fixes what pydantic *accepts*, not what it
        emits. Sending ``from_`` is a paid call whose every edge has lost its source node.
        (``by_alias`` cannot be forced from here: ``_run`` never sees the model instance, only
        the dict crewai already dumped. And pydantic's ``serialize_by_alias`` config is 2.11+,
        while this package floors at 2.7.2.)

        *Optional nulls are dropped.* crewai's own validation layer inserts them (see the
        module docstring); the model never asked for them, and the hub's schema does not
        accept them. Nested objects get this worse than the top level, because crewai's native
        schema pass runs ``ensure_all_properties_required`` over the whole tree — a fermat edge
        is advertised to the model with all eleven of its optional keys required and
        ``"type": "null"`` permitted, so the model is *instructed* to send
        ``{"from": "a", "to": "b", "cost": null, "latency": null, …}``. A ``None`` on a
        REQUIRED property is passed through untouched, because there the provider's own
        validator gives the authoritative message and inventing one here would guess at a
        contract this bridge does not own.
        """
        payload: dict[str, Any] = {}
        if args:
            # Positional arguments reach _run only through crewai's legacy paths, which either
            # skip validation entirely (``BaseTool.run(*args)``) or zip positionals onto the
            # schema's field order (``CrewStructuredTool._run``). Zipping is therefore the
            # framework's own convention, and it is followed literally — including for a lone
            # dict. Reading that dict as "the whole argument mapping" is the tempting shortcut
            # and it is wrong: five live capabilities (``fermat.route@v1``,
            # ``ablation.cascade@v1``, ``aestus.open@v1`` …) take an OBJECT as their first
            # argument, so the shortcut would turn ``run({"cost": 1})`` into a top-level
            # ``cost`` key the hub has never heard of instead of ``blend={"cost": 1}``.
            # Callers who mean a mapping have ``run(**mapping)``, or
            # ``to_structured_tool().invoke(mapping)`` — ``BaseTool`` itself has no ``invoke``.
            payload.update(dict(zip(self.args_schema.model_fields, args)))
        # kwargs last, because that is the order ``CrewStructuredTool._run`` uses when it
        # receives both (``input_dict = zip(...); input_dict.update(kwargs)``): the named
        # spelling wins. Python would call the same call a TypeError, so there is no
        # independently correct answer — only the framework's, and following it means a
        # mixed call sends the same body here as it would through crewai's own wrapper.
        payload.update(kwargs)

        return _restore(payload, self.capability.input_schema)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Call the capability and hand back something the agent can read.

        ``*args, **kwargs`` is not laziness — it is the abstract signature crewai declares,
        and narrowing it would make ``BaseTool._default_args_schema`` derive a schema from
        these parameters instead of using the capability's own.
        """
        payload = self._payload(args, kwargs)

        try:
            result = self.client.invoke(self.capability, payload)
        except BudgetExceeded as exc:
            # Text, not an exception, and worded as a stop rather than a retry. Nothing the
            # model can do to its arguments makes a spend ceiling go away, so it is told to
            # stop asking rather than invited to try again.
            logger.warning("%s: %s", self.name, exc)
            return (
                f"{self.capability.capability_id} was NOT called: {exc}. The spend limit for "
                "this run is reached — do not call this or any other paid tool again, and "
                "answer with what you already have."
            )
        except HubUnavailable as exc:
            # Also text, and this one is a money decision rather than a readability one. It
            # would be tidier to let a transport failure out of a bridge and let the operator
            # see the traceback — but crewai never shows it one. Both executor paths turn it
            # into text anyway (``Error executing tool: …`` natively, the tool_usage_exception
            # message on the ReAct path), and the ReAct path pays for the privilege: it
            # re-invokes on the exception and then retries the attempt, so ONE raising _run
            # reached a stubbed hub SIX times (module docstring, item 5). Six requests against
            # a paid endpoint, and because a reservation whose call never returned is refunded,
            # ``spent_usd`` reads $0.00 for all six — the budget cannot bound what it cannot
            # see. So it is answered once, here. Nothing is hidden: this is an operator-level
            # error in the log, ``last_result`` records the failure, and the text tells the
            # model the call did not happen rather than letting it believe it did.
            logger.error("%s: %s", self.name, exc)
            self.last_result = InvokeResult(
                ok=False, capability_id=self.capability.capability_id, error=str(exc)
            )
            return (
                f"{self.capability.capability_id} could NOT be reached: {exc}. No result and "
                "no receipt came back, so this run recorded no charge for it. This is a "
                "transport or configuration fault, not a problem with your arguments — "
                "rewording them will not help."
            )

        self.last_result = result
        if result.receipt_verified is False:
            # Loud for the operator, silent for the model: the output may not have come from
            # the provider it claims. Kept out of the tool text because a receipt is a blob no
            # model reads, and the agent cannot act on it either way.
            logger.warning(
                "%s: receipt did NOT verify against the origin key (%s) — treat the output as "
                "unattributed", self.capability.capability_id, result.receipt_verify_reason,
            )
        return result.for_model()

    def format_output_for_agent(self, raw_result: Any) -> str:
        """Render the result for the agent's context as JSON, not as a Python repr.

        crewai's default is ``str(raw_result)``, which turns a capability's JSON output into
        ``{'ok': True, 'beta': None}`` — single quotes, ``True``, ``None``. The capability
        answered in JSON and the next thing the model does is quote fields of it back into
        another tool call, so it is handed JSON. Non-serialisable values fall back to their
        string form rather than failing the call. This hook is crewai's own extension point:
        both the ReAct path and the native-function-calling path look it up on the tool.
        """
        if isinstance(raw_result, (dict, list)):
            try:
                return json.dumps(raw_result, ensure_ascii=False, default=str)
            except (TypeError, ValueError):  # pragma: no cover - default=str covers this
                return str(raw_result)
        return str(raw_result)


def aimarket_tools(
    base_url: str,
    *,
    intent: str = "",
    limit: int = 0,
    max_price_usd: float | None = None,
    free_only: bool = False,
    budget_usd: float = 1.0,
    capabilities: list[Capability] | None = None,
    http_client: httpx.Client | None = None,
    cache_function: Callable[..., bool] = never_cache,
    **kw: Any,
) -> list[AIMarketTool]:
    """Every capability the hub offers that passes the filters, as CrewAI tools.

    ``intent``, ``limit``, ``max_price_usd`` and ``free_only`` are build-time filters — a tool
    the operator cannot afford must never reach the agent's registry, because after that the
    agent decides when to call it. ``budget_usd`` is the run-time ceiling, and it is shared:
    **one** :class:`HubClient` backs every tool returned, so fifty tools spend one budget
    instead of fifty. Read the spend back off any of them (``tools[0].client.spent_usd``).

    ``budget_usd=0`` means "spend nothing" and is honoured: the tools are built, the model
    sees them, every paid call is refused. It used to mean **no ceiling**, because
    :class:`HubClient` read a falsy budget as unlimited — a footgun all three adapters had
    separately grown a warning or a veto against, which is how it became clear the fix belonged
    in the core. ``None`` is the spelling for no ceiling now, and that is what gets warned
    about here.

    A hub that cannot be read raises :class:`CatalogError` out of here, uncaught. Catching it
    would hand the framework an agent holding zero tools, which is indistinguishable from a
    hub with nothing to sell and is the failure the catalogue layer exists to prevent.

    ``capabilities`` accepts an already-fetched catalogue and skips the HTTP call, which is
    what an operator who has just listed the catalogue for a human should pass. The price and
    ``limit`` filters still apply to it, so a filter argument means the same thing however the
    catalogue arrived; ``intent`` is the one exception, because relevance ranking is the hub's
    to compute and cannot be reproduced offline. Remaining keyword arguments go to
    :class:`HubClient` (``timeout``, ``verify_receipts``, ``affiliate_id``, ``agent``).
    """
    if capabilities is None:
        caps = fetch_catalog(
            base_url,
            intent=intent,
            limit=limit,
            max_price_usd=max_price_usd,
            free_only=free_only,
            client=http_client,
        )
    else:
        caps = list(capabilities)
        if intent:
            logger.warning(
                "intent=%r was ignored: ranking is computed by the hub's search, and an "
                "already-fetched catalogue cannot be re-ranked without asking it", intent,
            )
        if free_only:
            caps = [c for c in caps if c.is_free]
        elif max_price_usd is not None:
            caps = [c for c in caps if c.price_usd <= max_price_usd]
        if limit:
            caps = caps[:limit]

    if not caps:
        # fetch_catalog raises on failure, so an empty list here means the filters excluded
        # everything — and the most likely cause is free_only against a catalogue where
        # nothing is free (none of the 47 live capabilities are). Said out loud, because an
        # agent handed zero tools looks identical to an agent whose hub has nothing to sell.
        logger.warning(
            "no capability at %s passed the filters (intent=%r, limit=%s, "
            "max_price_usd=%s, free_only=%s), so no tools were built",
            base_url, intent, limit, max_price_usd, free_only,
        )
        return []

    if budget_usd is None:
        # Every one of the 47 live capabilities is paid, so an unlimited client handed to an
        # agent that decides its own call count is the one configuration that can run a wallet
        # down without anything in the stack objecting.
        logger.warning(
            "budget_usd=None disables the spend ceiling entirely: %d paid tools will be built "
            "with NO limit on what the agent may spend. Pass a number for a ceiling, or 0 to "
            "forbid every paid call.", len(caps),
        )

    client = HubClient(base_url, budget_usd=budget_usd, **kw)

    # crewai rewrites the tool name it shows the model — sanitize_tool_name lowercases and
    # folds anything outside [a-z0-9_] to '_' — and then matches tool calls on the sanitized
    # form. The catalogue de-duplicates on the pre-sanitized name, so two ids differing only
    # by '-' vs '_' or by case survive it and collide here, after which crewai's lookup
    # returns whichever came first and the agent is billed for a capability it did not ask
    # for. The live 47 are collision-free (only 'security-rules_sec-feed_v1' is rewritten at
    # all); this keeps them that way as the catalogue grows.
    tools: list[AIMarketTool] = []
    taken: dict[str, str] = {}
    for cap in caps:
        name = cap.tool_name
        key = sanitize_tool_name(name)
        if key in taken:
            for i in range(2, 100):
                candidate = f"{name}_{i}"
                if sanitize_tool_name(candidate) not in taken:
                    logger.warning(
                        "%s would appear to the model as %r, which %s already claimed; "
                        "renaming it to %r", cap.capability_id, key, taken[key], candidate,
                    )
                    name, key = candidate, sanitize_tool_name(candidate)
                    break
            else:
                # Falling through used to leave `key` colliding and overwrite the winner's
                # entry, which is the exact billing mix-up the block exists to prevent —
                # silently, and only for whoever came second. `tool_name_for` raises in the
                # same corner, so this does too.
                raise CatalogError(
                    f"cannot find a tool name for {cap.capability_id!r} that survives "
                    f"crewai's sanitiser without colliding with {taken[key]!r}"
                )
        taken[key] = cap.capability_id
        try:
            tools.append(
                AIMarketTool.for_capability(cap, client, name=name,
                                            cache_function=cache_function)
            )
        except CatalogError:
            raise
        except Exception as exc:  # noqa: BLE001 - see the note below
            logger.warning(
                "skipping %s: its argument schema cannot be turned into a crewai tool (%s: %s). "
                "The other %d capabilities are unaffected",
                cap.capability_id, type(exc).__name__, exc, len(caps) - 1,
            )
    return tools
