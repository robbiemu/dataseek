import httpx

from seek.components.search_graph.litesearch import AsyncRateLimitManager

# -------------------------
# Globals for tool clients
# -------------------------

# Use a single async rate manager and http client for async search providers
RATE_MANAGER = AsyncRateLimitManager()
HTTP_CLIENT = httpx.AsyncClient()

# Dedicated synchronous client for page fetches (markdown, pre-scan, etc.)
SYNC_HTTP_CLIENT = httpx.Client(follow_redirects=True)
