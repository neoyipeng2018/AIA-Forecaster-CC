"""Example: Company-specific configuration overrides.

Import and use these values in your company extensions.
"""

import os

# Bloomberg
BLOOMBERG_API_KEY = os.environ.get("BLOOMBERG_API_KEY", "")

# Additional blacklisted domains (added on import)
EXTRA_BLACKLISTED_DOMAINS = [
    # "internal-wiki.example.com",
]

if EXTRA_BLACKLISTED_DOMAINS:
    from aia_forecaster.search.web import add_blacklisted_domains
    add_blacklisted_domains(EXTRA_BLACKLISTED_DOMAINS)
