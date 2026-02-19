"""Company extensions entry point.

This module is auto-imported by ``aia_forecaster`` at startup.
Register all custom pairs, feeds, data sources, and config here.
"""

from company.pairs import register_custom_pairs

# Register custom currency pairs
register_custom_pairs()

# Import search subpackage so @data_source decorators run
import company.search  # noqa: F401

# Uncomment to register a custom LLM backend (Azure, Anthropic, Ollama, etc.)
# from company.llm import register_llm_connector
# register_llm_connector()
