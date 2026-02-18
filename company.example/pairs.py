"""Example: Register custom currency pairs (exotics, NDFs, etc.)."""

from aia_forecaster.fx.pairs import PairConfig, register_pair
from aia_forecaster.search.rss import register_currency_keywords


def register_custom_pairs() -> None:
    """Register company-specific currency pairs and keywords."""

    # Example: offshore Chinese yuan
    register_pair(PairConfig(
        pair="USDCNH", base="USD", quote="CNH",
        pip_size=0.0001, typical_daily_range=0.005,
    ))
    register_currency_keywords("CNH", [
        "renminbi", "yuan", "pboc", "china", "chinese",
    ])

    # Example: Singapore dollar
    # register_pair(PairConfig(
    #     pair="USDSGD", base="USD", quote="SGD",
    #     pip_size=0.0001, typical_daily_range=0.004,
    # ))
    # register_currency_keywords("SGD", [
    #     "singapore", "mas", "sgd",
    # ])
