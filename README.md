<!-- aicom-mirror-notice -->
> **📖 Read-only mirror.** `aimarket-bridges` is published from the canonical AI-Factory monorepo.
> **Pull requests are not accepted** — any commit pushed here is overwritten by
> `scripts/mirror_satellites.sh` on the next sync.
> 🐞 Found a bug or have a request? Please **[open an issue](https://github.com/alexar76/aimarket-bridges/issues)**.

# aimarket-bridges

**Give your LangGraph, CrewAI or AutoGen agent 47 more tools in two lines.**

```python
from aimarket_bridges.langchain import aimarket_tools

tools = aimarket_tools("https://modelmarket.dev", intent="verifiable randomness")
```

Every tool is a real capability on the [AIMarket](https://modelmarket.dev) network — a
verifiable random draw, a Byzantine-resistant consensus number, a proof of elapsed sequential
work, an optimal-transport plan — each returning a **signed receipt** your agent can verify
without trusting the marketplace.

## Install

```bash
pip install "aimarket-bridges[langgraph]"
```

```bash
pip install "aimarket-bridges[crewai]"
```

```bash
pip install "aimarket-bridges[autogen]"
```

The core has no framework dependency. Install only the one you use — CrewAI and AutoGen do not
agree on a pydantic version and cannot share an environment.

## The three adapters

| Framework | Import | Returns |
|---|---|---|
| LangChain / LangGraph | `aimarket_bridges.langchain` | `list[StructuredTool]` |
| CrewAI | `aimarket_bridges.crewai` | `list[crewai.tools.BaseTool]` |
| AutoGen | `aimarket_bridges.autogen` | `list[autogen_core.tools.BaseTool]` |

Same signature everywhere:

```python
aimarket_tools(
    base_url,               # the hub, e.g. "https://modelmarket.dev"
    intent="",              # rank by relevance instead of taking the whole catalogue
    limit=0,                # cap how many tools the agent sees
    max_price_usd=None,     # never hand over a tool you cannot afford
    free_only=False,
    budget_usd=1.0,         # hard ceiling across every call these tools make
)
```

## What it costs

Capabilities are priced per call, from $0.001. The `budget_usd` ceiling is enforced **before**
each call and is safe across threads, so an agent that loops cannot outspend it. `max_price_usd`
filters at build time — the honest place for a limit, because once a tool is in the agent's
registry the agent decides when to call it.

There is a free trial tier: an agent with no wallet can try capabilities before any payment is
set up.

## Receipts actually verify

Every call returns a signed 7-field receipt. It is kept out of the tool's text result — it would
cost the model context for a blob it never reads — and is available through the framework's own
metadata channel and through `HubClient.last_receipt`.

Verification uses the key of the capability's **origin**, not the hub's. This matters: 42 of the
47 live capabilities are federated, so their receipts are signed by the provider, not by the
broker that routed the call. Checking against the hub's key reports `invalid-signature` on
perfectly valid receipts for 89% of the catalogue.

## Refusals are results, not crashes

When a capability rejects its input, your agent gets a sentence it can act on:

```
sortes.draw@v1 refused this input: 'num_bytes' must be an integer, got str
```

and it corrects the argument on the next turn. Nothing raises, so one bad argument does not
abort the surrounding graph or crew. Transport and configuration failures **do** raise — those
the model cannot fix, and swallowing them yields an agent that reports success having called
nothing.

## Without a framework

The core is usable on its own if you are wiring something with no adapter yet:

```python
from aimarket_bridges import fetch_catalog, HubClient

capabilities = fetch_catalog("https://modelmarket.dev", intent="consensus")
with HubClient("https://modelmarket.dev", budget_usd=0.50) as hub:
    result = hub.invoke(capabilities[0], {"values": [1.0, 2.0, 3.0, 100.0]})
    print(result.output, result.receipt_verified)
```

`fetch_catalog` **raises** when the hub is unreachable rather than returning an empty list. An
agent that starts up believing it has no capabilities is a much worse failure than one that
refuses to start.

## Notes on the frameworks

Written against langchain-core 1.5.2, langgraph 1.2.10, crewai 1.15.8, autogen-core 0.7.5,
each verified by introspection rather than from documentation. They disagree about how a tool
declares its arguments, which is most of what this package absorbs:

- **langchain-core** accepts a raw JSON Schema dict, so a capability's schema passes straight
  through.
- **crewai** requires a pydantic model and reads `.model_fields` off it.
- **autogen** derives schemas from Python type annotations, so a runtime-shaped capability
  needs an explicit args model.

The `oneOf` in `kantor.transport@v1` (a point is an array of numbers *or* a number) and in
`fermat.route@v1` (an edge is an array *or* an object) becomes a proper union in all three.

Apache-2.0.
