"""serve.py checks — no network, no API key. Spawns serve.py over real stdio
and talks to it with our own MCPClient.

Run: python test_serve.py
"""
import os
import sys

from mcp_client import MCPClient
import serve
from mcp.server.mcpserver.exceptions import ToolError

HERE = os.path.dirname(os.path.abspath(__file__))
# no MCP servers of its own, and an empty key so a stray .env can't make a
# test spend real credits
SERVE = {"serve": {
    "command": sys.executable,
    "args": [os.path.join(HERE, "serve.py")],
    "env": {"MCP_SERVERS_FILE": "/nonexistent.json", "TOKENROUTER_API_KEY": ""},
}}


def _state(**over):
    return {"goal": "g", "stop_reason": "pipeline_done", "iteration": 1,
            "tool_call_count": 1, "sources": ["http://s"], "answer": "the answer",
            "subtopics": ["t"], "critic_verdict": "ok", **over}


def test_research_returns_markdown_report():
    serve.agent.run_pipeline = lambda goal: _state(goal=goal)
    out = serve.research("my goal")
    assert out.startswith("# my goal") and "the answer" in out and "- http://s" in out


def test_research_failure_becomes_error_not_exit():
    def boom(goal):
        raise SystemExit("TOKENROUTER_API_KEY is not configured (set it in .env)")
    serve.agent.run_pipeline = boom
    try:
        serve.research("g")
        assert False, "should have raised"
    except ToolError as e:
        assert "TOKENROUTER_API_KEY" in str(e)


def test_pipeline_failed_state_is_an_error():
    serve.agent.run_pipeline = lambda goal: _state(stop_reason="pipeline_failed", answer="")
    try:
        serve.research("g")
        assert False, "should have raised"
    except ToolError as e:
        assert "every subtopic failed" in str(e)


def test_served_over_stdio_lists_tool_and_reports_errors():
    with MCPClient(SERVE) as mcp:
        assert mcp.errors == {}
        (schema,) = [s["function"] for s in mcp.schemas]
        assert schema["name"] == "serve__research"
        assert schema["parameters"]["required"] == ["goal"]
        result = mcp.call("serve__research", {"goal": "anything"})
        assert "TOKENROUTER_API_KEY" in result["error"]


# --- HTTP mode ---------------------------------------------------------

TOKEN = "0123456789abcdef0123"


def _run_serve(args, token=None):
    """Run serve.py to completion in an empty cwd (no stray .env); returns the process."""
    import subprocess
    import tempfile
    env = {"PATH": os.environ["PATH"], "MCP_SERVERS_FILE": "/nonexistent.json"}
    if token:
        env["MCP_AUTH_TOKEN"] = token
    with tempfile.TemporaryDirectory() as d:
        return subprocess.run([sys.executable, os.path.join(HERE, "serve.py"), *args],
                              env=env, cwd=d, capture_output=True, text=True, timeout=60)


def test_bearer_auth_blocks_requests_without_the_token():
    import asyncio
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def call(headers):
        sent = []

        async def send(msg):
            sent.append(msg)
        await serve.BearerAuth(inner, TOKEN)(
            {"type": "http", "headers": headers}, None, send)
        return sent[0]["status"]

    assert asyncio.run(call([])) == 401
    assert asyncio.run(call([(b"authorization", b"Bearer wrong")])) == 401
    assert asyncio.run(call([(b"authorization", "Bearer é".encode())])) == 401  # non-ascii must not 500
    assert seen == []  # rejected requests never reach the app
    assert asyncio.run(call([(b"authorization", f"Bearer {TOKEN}".encode())])) == 200
    assert seen == ["http"]


def test_http_mode_refuses_to_start_without_port_or_token():
    r = _run_serve(["--http"], token=TOKEN)
    assert r.returncode == 2 and "--port" in r.stderr  # no silent default port
    r = _run_serve(["--http", "--port", "1"])
    assert r.returncode != 0 and "MCP_AUTH_TOKEN" in r.stderr
    r = _run_serve(["--http", "--port", "1"], token="short")
    assert r.returncode != 0 and "MCP_AUTH_TOKEN" in r.stderr
    r = _run_serve(["--port", "1"], token=TOKEN)
    assert r.returncode == 2  # --port without --http


def test_security_settings_default_and_extra_hosts():
    assert serve._security([]) is None
    sec = serve._security(["host.docker.internal:*"])
    assert sec.enable_dns_rebinding_protection is True
    assert "host.docker.internal:*" in sec.allowed_hosts and "localhost:*" in sec.allowed_hosts


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _list_tools_over_http(url):
    from mcp import ClientSession
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
    async with create_mcp_http_client(headers={"Authorization": f"Bearer {TOKEN}"}) as http:
        async with streamable_http_client(url, http_client=http) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                return [t.name for t in (await session.list_tools()).tools]


def test_served_over_http_needs_token_and_known_host():
    import asyncio
    import socket
    import subprocess
    import tempfile
    import time
    import urllib.error
    import urllib.request

    port = _free_port()
    env = {"PATH": os.environ["PATH"], "MCP_SERVERS_FILE": "/nonexistent.json",
           "MCP_AUTH_TOKEN": TOKEN, "TOKENROUTER_API_KEY": ""}
    url = f"http://127.0.0.1:{port}/mcp"
    with tempfile.TemporaryDirectory() as d:
        proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "serve.py"), "--http", "--port", str(port),
             "--allow-host", "host.docker.internal:*"],
            env=env, cwd=d, stderr=subprocess.DEVNULL)
        try:
            for _ in range(60):  # wait until it accepts connections
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                    break
                except OSError:
                    time.sleep(0.25)

            def post(headers):
                req = urllib.request.Request(url, data=b"{}", method="POST", headers=headers)
                try:
                    return urllib.request.urlopen(req, timeout=10).status
                except urllib.error.HTTPError as e:
                    return e.code

            assert post({"Content-Type": "application/json"}) == 401
            assert post({"Authorization": "Bearer nope", "Content-Type": "application/json"}) == 401
            bad_host = {"Authorization": f"Bearer {TOKEN}", "Host": "evil.example:1",
                        "Content-Type": "application/json"}
            assert post(bad_host) == 421  # DNS-rebinding protection
            assert asyncio.run(_list_tools_over_http(url)) == ["research"]
        finally:
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
