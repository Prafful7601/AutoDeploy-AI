"""
A thin, honest GitHub REST API client. Not a general-purpose SDK — just the
handful of calls the Stage 3 Layer 2 extractor needs, with auth and rate
limits handled explicitly rather than left to trial and error at 2am when a
workflow run fails.

Auth: reads `GITHUB_TOKEN` from the environment (via python-dotenv if a
.env file is present). Never hardcoded, never logged. Works unauthenticated
too (public repos, 60 requests/hour instead of 5,000) — useful for a quick
local check, not for Layer 3's Action, which should always set the token.
"""

import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()  # no-op if there's no .env; never overwrites a real env var

API_ROOT = "https://api.github.com"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


class GitHubRateLimitError(RuntimeError):
    """Raised when the token's rate limit is exhausted. Carries the reset
    time so a caller can decide whether to wait or give up, rather than
    this client silently blocking for an unbounded amount of time."""

    def __init__(self, reset_epoch: int):
        self.reset_epoch = reset_epoch
        wait_s = max(0, reset_epoch - int(time.time()))
        super().__init__(
            f"GitHub API rate limit exhausted. Resets in {wait_s}s "
            f"(at epoch {reset_epoch}). Set GITHUB_TOKEN for a 5,000/hour "
            f"limit instead of the unauthenticated 60/hour."
        )


class GitHubClient:
    def __init__(self, token: Optional[str] = None):
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN")
        self.session = requests.Session()
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self.session.headers.update(headers)
        self.authenticated = bool(self.token)

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = endpoint if endpoint.startswith("http") else f"{API_ROOT}{endpoint}"
        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.request(method, url, timeout=15, **kwargs)
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue

            if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
                raise GitHubRateLimitError(int(resp.headers.get("X-RateLimit-Reset", "0")))

            if resp.status_code >= 500:
                # Transient server-side failure — bounded retry, not a loop.
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue

            return resp

        raise RuntimeError(f"GitHub API request to {url} failed after {MAX_RETRIES} retries: {last_exc}")

    # NOTE: every method below takes the API endpoint as `endpoint`, never
    # `path` — GitHub's own query parameter for filtering commits by file
    # path is ALSO called `path` (used by the extractor's
    # commits_on_touched_files computation), and a positional/keyword name
    # collision there silently swallowed every one of those calls during
    # testing (`count_via_last_page() got multiple values for argument
    # 'path'`, caught by an over-broad except and reported as "0 history"
    # instead of an error). Renamed everywhere rather than working around
    # it at just the one call site.

    def get(self, endpoint: str, **params) -> requests.Response:
        return self._request("GET", endpoint, params=params)

    def get_json(self, endpoint: str, **params) -> Any:
        resp = self.get(endpoint, **params)
        resp.raise_for_status()
        return resp.json()

    def get_json_or_none(self, endpoint: str, **params) -> Optional[Any]:
        """Like get_json, but returns None on 404 instead of raising — for
        calls where 'this doesn't exist' is a normal, expected outcome
        (e.g. a commit with no associated pull requests)."""
        resp = self.get(endpoint, **params)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def get_all_pages(self, endpoint: str, max_pages: int, items_key: Optional[str] = None, **params) -> List[Any]:
        """Follow `Link: rel="next"` until exhausted or max_pages hit.
        max_pages bounds API cost on repos with very long histories — see
        the parity report for what this caps and why.

        Most list endpoints (`/commits`, `/contributors`) return a bare
        JSON array. A few (`/actions/runs`) wrap it in an object, e.g.
        `{"total_count": N, "workflow_runs": [...]}` — pass `items_key`
        ("workflow_runs") for those, or this happily iterates the dict's
        keys as if they were items."""
        items: List[Any] = []
        url = f"{API_ROOT}{endpoint}"
        page_params = {**params, "per_page": 100}
        for _ in range(max_pages):
            resp = self._request("GET", url, params=page_params)
            resp.raise_for_status()
            body = resp.json()
            items.extend(body[items_key] if items_key else body)
            next_url = resp.links.get("next", {}).get("url")
            if not next_url:
                break
            url, page_params = next_url, {}  # next_url already has query params baked in
        return items

    def count_via_last_page(self, endpoint: str, **params) -> int:
        """GitHub has no 'give me the count' endpoint for most list resources.
        Standard trick: request 1 item per page and read the page number out
        of the `Link: rel="last"` header — that number IS the total count.
        If there's no 'last' link, there's 0 or 1 items (checked directly)."""
        resp = self.get(endpoint, **{**params, "per_page": 1})
        resp.raise_for_status()
        last_url = resp.links.get("last", {}).get("url")
        if not last_url:
            return len(resp.json())  # 0 or 1
        # Parse `page=N` out of the last-page URL.
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(last_url).query)
        return int(qs.get("page", ["1"])[0])
