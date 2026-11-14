import asyncio
import contextvars
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import aiohttp
from pydantic import Field

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


################
# INPUT SCHEMA #
################
class DeedPageToolInputSchema(BaseIOSchema):
    """
    Publish one or more static files (HTML, CSS, JS, images as text) to a public HTTPS URL on deed.page.
    Use this when the user wants a report, page, or small site shared as a link.
    The page is public: never include secrets or private data.
    """

    files: Dict[str, str] = Field(
        ...,
        description="Map of file path to file content. Include an 'index.html' as the entry page, e.g. {'index.html': '<!doctype html>...'}.",
    )
    slug: Optional[str] = Field(
        None,
        pattern=r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$",
        description="Optional subdomain (lowercase letters, digits, hyphens). Omit it to get a random one.",
    )
    ttl_seconds: Optional[int] = Field(
        None,
        ge=60,
        description="Optional lifetime in seconds after which the page is removed. Leave empty to keep it.",
    )


####################
# OUTPUT SCHEMA(S) #
####################
class DeedPageToolOutputSchema(BaseIOSchema):
    """Result of publishing to deed.page."""

    url: str = Field(..., description="Public HTTPS URL of the published page.")
    slug: str = Field(..., description="Slug (subdomain) of the published site.")
    version: Optional[str] = Field(None, description="Version identifier of this publish, e.g. 'v1'.")
    expires_at: Optional[str] = Field(None, description="Expiry time if a TTL was set, otherwise empty.")


#################
# CONFIGURATION #
#################
class DeedPageToolConfig(BaseToolConfig):
    """Configuration for the DeedPageTool."""

    api_token: str = Field(
        default="",
        description=(
            "Optional deed.page owner token. Falls back to the DEEDPAGE_TOKEN environment variable. "
            "Without one, the first publish creates a token, which the tool keeps in memory (never in the output)."
        ),
    )
    base_url: str = Field(default="https://deed.page", description="deed.page API base URL.")
    user_agent: str = Field(
        default="atomic-agents-deed-page-tool/1.0 (+https://github.com/eigenwise/atomic-agents)",
        description="User agent for HTTP requests.",
    )
    timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="HTTP request timeout in seconds.")


#####################
# MAIN TOOL & LOGIC #
#####################
class DeedPageTool(BaseTool[DeedPageToolInputSchema, DeedPageToolOutputSchema]):
    """Tool for publishing static files to a public URL via the deed.page API."""

    def __init__(self, config: DeedPageToolConfig = DeedPageToolConfig()):
        super().__init__(config)
        # The owner token is needed to update or delete sites. It is kept on the
        # instance and deliberately left out of the output schema, so it never
        # reaches the model's context.
        self.token = config.api_token or os.getenv("DEEDPAGE_TOKEN", "")
        # Each tokenless publish creates a new owner. Only one publish may run
        # without a token at a time, so concurrent first-use calls (sync or
        # async) wait for that token and publish under the same owner.
        self._bootstrap_lock = threading.Lock()
        self.base_url = config.base_url.rstrip("/")
        self.user_agent = config.user_agent
        self.timeout = config.timeout

    def _headers(self) -> Dict[str, str]:
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def run_async(self, params: DeedPageToolInputSchema) -> DeedPageToolOutputSchema:
        if not params.files:
            raise ValueError("files must contain at least one file, usually 'index.html'.")
        if self.token:
            return await self._publish(params)

        # A threading lock (polled, so the event loop is never blocked and
        # cancellation is safe) because run() executes each call on its own
        # event loop in a worker thread.
        while not self._bootstrap_lock.acquire(blocking=False):
            await asyncio.sleep(0.05)
        try:
            return await self._publish(params)
        finally:
            self._bootstrap_lock.release()

    async def _publish(self, params: DeedPageToolInputSchema) -> DeedPageToolOutputSchema:
        body: Dict[str, object] = {"files": params.files}
        if params.slug:
            body["slug"] = params.slug
        headers = self._headers()
        if params.ttl_seconds:
            headers["X-Ttl"] = str(params.ttl_seconds)

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            async with session.post(f"{self.base_url}/v1/sites", json=body) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 400 or not data.get("ok"):
                    error = data.get("error") or data.get("message") or resp.reason
                    raise Exception(f"deed.page publish failed: {resp.status} {error}")

        if data.get("token") and not self.token:
            self.token = data["token"]
        return DeedPageToolOutputSchema(
            url=data["url"],
            slug=data["slug"],
            version=data.get("version"),
            expires_at=data.get("expires_at"),
        )

    async def delete_async(self, slug: str) -> None:
        """Delete a site owned by this tool's token. Not exposed to the agent."""
        if not self.token:
            raise ValueError("No deed.page token: set DeedPageToolConfig.api_token or DEEDPAGE_TOKEN, or publish first.")
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(headers=self._headers(), timeout=timeout) as session:
            async with session.delete(f"{self.base_url}/v1/sites/{slug}") as resp:
                if resp.status >= 400:
                    raise Exception(f"deed.page delete failed for '{slug}': {resp.status} {resp.reason}")

    def run(self, params: DeedPageToolInputSchema) -> DeedPageToolOutputSchema:
        with ThreadPoolExecutor() as executor:
            return executor.submit(contextvars.copy_context().run, asyncio.run, self.run_async(params)).result()

    def delete(self, slug: str) -> None:
        with ThreadPoolExecutor() as executor:
            executor.submit(contextvars.copy_context().run, asyncio.run, self.delete_async(slug)).result()


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":  # pragma: no cover
    import urllib.error
    import urllib.request

    from rich.console import Console

    console = Console()

    def status(url: str) -> int:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.status
        except urllib.error.HTTPError as e:
            return e.code

    tool = DeedPageTool()
    output = tool.run(
        DeedPageToolInputSchema(
            files={"index.html": "<!doctype html><title>Atomic Agents</title><h1>Published with Atomic Agents</h1>"},
            ttl_seconds=600,
        )
    )
    console.print(f"[bold]Published:[/bold] {output.url} -> {status(output.url)}")
    tool.delete(output.slug)
    console.print(f"[bold]Deleted:[/bold] {output.url} -> {status(output.url)}")
