"""The research agent as an MCP server: one tool, research(goal) -> markdown report.

    python serve.py                         # stdio, for MCP clients that spawn it
    python serve.py --http --port <port>    # streamable HTTP at http://127.0.0.1:<port>/mcp

Claude Code: claude mcp add research -- python /path/to/serve.py
HTTP mode is for clients that can't spawn a process (n8n). The tool spends LLM
and search credits, so HTTP requires MCP_AUTH_TOKEN (sent as `Authorization:
Bearer <token>`) and an explicit --port; it binds 127.0.0.1 unless --host says
otherwise. The agent's own MCP servers (mcp_servers.json) start alongside.
Logs go to stderr — stdout belongs to the MCP protocol, never print() there.
"""
import argparse
import hmac
import os
import sys
import threading

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.server.mcpserver.exceptions import ToolError
from starlette.responses import PlainTextResponse

import agent

MIN_TOKEN_LEN = 16

server = MCPServer("research-agent")

# ponytail: one research run at a time. Each run is ~35-40 LLM calls on a free
# tier; concurrent runs would just trade 429s. Queue/limit properly if needed.
_one_run_at_a_time = threading.Lock()


@server.tool()
def research(goal: str) -> str:
    """Research a goal on the web (plan subtopics, search and read sources,
    self-critique) and return a markdown report with sources. Takes minutes."""
    with _one_run_at_a_time:
        try:
            state = agent.run_pipeline(goal)
        # ToolError (not a bare exception) is what makes the SDK show the message to
        # the client; SystemExit: call_llm exits when there is no API key
        except (Exception, SystemExit) as e:
            raise ToolError(f"research failed: {e}") from e
    if state["stop_reason"] == "pipeline_failed":
        raise ToolError("research failed: every subtopic failed (LLM unreachable?)")
    return agent.report_markdown(state)


class BearerAuth:
    """ASGI middleware: 401 unless the request carries `Authorization: Bearer <token>`."""

    def __init__(self, app, token):
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            sent = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(sent, self.expected):
                return await PlainTextResponse("unauthorized", status_code=401)(
                    scope, receive, send)
        await self.app(scope, receive, send)


def _security(extra_hosts):
    """None = SDK default (localhost only). With extras: the same localhost
    allowlist plus the given hosts, rebinding protection still on."""
    if not extra_hosts:
        return None
    local = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=local + extra_hosts,
        allowed_origins=[f"http://{h}" for h in local + extra_hosts],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Research agent as an MCP server")
    parser.add_argument("--http", action="store_true",
                        help="serve streamable HTTP (needs --port and MCP_AUTH_TOKEN)")
    parser.add_argument("--port", type=int,
                        help="HTTP port; no default on purpose, pick one that is free")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address")
    parser.add_argument("--allow-host", action="append", default=[], metavar="HOST:PORT",
                        help="extra Host header to accept, e.g. host.docker.internal:8765 "
                             "(DNS-rebinding protection stays on; repeatable)")
    args = parser.parse_args(argv)
    if args.http and args.port is None:
        parser.error("--http needs an explicit --port")
    if args.port is not None and not args.http:
        parser.error("--port only makes sense with --http")

    agent.load_dotenv()
    token = os.environ.get("MCP_AUTH_TOKEN", "")
    if args.http and len(token) < MIN_TOKEN_LEN:
        sys.exit(f"MCP_AUTH_TOKEN must be set (>= {MIN_TOKEN_LEN} chars) for --http, "
                 "e.g. `openssl rand -hex 24`")

    with agent.mcp_tools():
        if args.http:
            app = BearerAuth(server.streamable_http_app(
                host=args.host, transport_security=_security(args.allow_host)), token)
            uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        else:
            server.run("stdio")


if __name__ == "__main__":
    main()
