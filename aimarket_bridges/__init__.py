"""AIMarket capabilities as native tools in the frameworks people actually build with.

    from aimarket_bridges.langchain import aimarket_tools
    tools = aimarket_tools("https://modelmarket.dev", intent="verifiable randomness")

The framework adapters live in submodules and are NOT imported here, on purpose: importing
this package must not pull in langchain, crewai or autogen. Those three do not even agree on
a pydantic version — crewai pins it lower than autogen — so they cannot share one
environment, let alone one import graph. Ask for the one you use:

    aimarket_bridges.langchain   LangChain / LangGraph  (StructuredTool)
    aimarket_bridges.crewai      CrewAI                 (crewai.tools.BaseTool)
    aimarket_bridges.autogen     AutoGen                (autogen_core.tools.BaseTool)

What every adapter shares is in the core, and is usable on its own if you are wiring a
framework that has no adapter yet:

    Capability / fetch_catalog   what the hub sells, in tool-ready form
    HubClient / InvokeResult     one call, with the budget, the refusal and the receipt
    model_from_schema            JSON Schema -> pydantic, for frameworks that demand a model
"""

from __future__ import annotations

from aimarket_bridges.catalog import Capability, CatalogError, fetch_catalog, tool_name_for
from aimarket_bridges.client import (
    BudgetExceeded,
    HubClient,
    HubUnavailable,
    InvokeResult,
)
from aimarket_bridges.receipts import OriginKeyResolver, ReceiptCheck
from aimarket_bridges.schema import SchemaError, model_from_schema, unsupported_keywords

__all__ = [
    "Capability",
    "CatalogError",
    "fetch_catalog",
    "tool_name_for",
    "HubClient",
    "InvokeResult",
    "BudgetExceeded",
    "HubUnavailable",
    "OriginKeyResolver",
    "ReceiptCheck",
    "model_from_schema",
    "unsupported_keywords",
    "SchemaError",
    "__version__",
]
__version__ = "0.1.0"
