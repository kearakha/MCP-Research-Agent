"""MCP client layer: spawn stdio MCP servers, discover their tools, call them.

The agent loop is sync but the `mcp` SDK is async, so a daemon thread owns an
asyncio loop that keeps every server session open for the whole run; callers
use the sync `call()`. Tool names are exposed as `<server>__<tool>` so two
servers (or a server and a native tool) can never collide. Like tools.py,
nothing here raises at the agent: failures come back as `{"error": ...}`.

    with MCPClient({"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}}) as mcp:
        mcp.schemas                              # OpenAI-style, same shape as tools.TOOL_SCHEMAS
        mcp.call("fetch__fetch", {"url": "..."}) # -> {"text": ...} or {"error": ...}
"""
import asyncio
import threading
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

START_TIMEOUT = 60  # first run of uvx/npx may download the server
CALL_TIMEOUT = 60
MAX_RESULT_CHARS = 8000  # same cap as tools.MAX_PAGE_CHARS — keep context small


class MCPClient:
    def __init__(self, servers):
        """servers: {name: {"command": str, "args": [...], "env": {...}}}"""
        self._servers = servers
        self._tools = {}  # prefixed name -> (session, original tool name)
        self.schemas = []
        self.errors = {}  # server name -> why it failed to start
        self._loop = None
        self._thread = None

    def __enter__(self):
        if self._servers:
            ready = threading.Event()
            self._thread = threading.Thread(
                target=asyncio.run, args=(self._main(ready),), daemon=True
            )
            self._thread.start()
            ready.wait()
        return self

    def __exit__(self, *exc):
        if self._thread:
            self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(timeout=10)

    async def _main(self, ready):
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        stacks = []
        for name, cfg in self._servers.items():
            stack = AsyncExitStack()
            try:
                async with asyncio.timeout(START_TIMEOUT):
                    await self._connect(stack, name, cfg)
                stacks.append(stack)
            except Exception as e:  # one bad server must not sink the others
                self.errors[name] = str(e) or type(e).__name__
                try:
                    await stack.aclose()
                except Exception:
                    pass
        ready.set()
        await self._stop.wait()
        for stack in reversed(stacks):
            try:
                await stack.aclose()
            except Exception:
                pass

    async def _connect(self, stack, name, cfg):
        params = StdioServerParameters(
            command=cfg["command"], args=cfg.get("args", []), env=cfg.get("env")
        )
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
        # register only after the whole server came up, so a half-started
        # server leaves no dangling tools behind
        for tool in listed.tools:
            prefixed = f"{name}__{tool.name}"
            self._tools[prefixed] = (session, tool.name)
            self.schemas.append({
                "type": "function",
                "function": {
                    "name": prefixed,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            })

    def call(self, name, args):
        if name not in self._tools:
            return {"error": f"unknown MCP tool: {name}"}
        session, tool_name = self._tools[name]
        fut = asyncio.run_coroutine_threadsafe(
            session.call_tool(tool_name, args), self._loop
        )
        try:
            # ponytail: a hung call is abandoned, not killed — the server
            # process lives until the run ends. Per-call restart if it matters.
            result = fut.result(CALL_TIMEOUT)
        except TimeoutError:
            fut.cancel()
            return {"error": f"{name} timed out after {CALL_TIMEOUT}s"}
        except Exception as e:
            return {"error": f"{name} failed: {e}"}
        if not isinstance(result, CallToolResult):
            return {"error": f"{name} returned an unsupported result type"}
        text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
        text = text[:MAX_RESULT_CHARS]
        return {"error": text or f"{name} failed"} if result.is_error else {"text": text}
