"""mcp_client.py checks — spawns a tiny fake MCP server (this same file, run
with --server) over real stdio. No network, no API key.

Run: python test_mcp_client.py
"""
import sys
import time

import mcp_client
from mcp_client import MCPClient

FAKE = {"fake": {"command": sys.executable, "args": [__file__, "--server"]}}


def _serve():
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("fake")

    @server.tool()
    def echo(text: str) -> str:
        """Echo text back."""
        return f"echo:{text}"

    @server.tool()
    def boom() -> str:
        """Always fails."""
        raise ValueError("kaboom")

    @server.tool()
    def slow() -> str:
        """Sleeps past the call timeout."""
        time.sleep(5)
        return "late"

    server.run("stdio")


def test_discovers_tools_as_prefixed_openai_schemas():
    with MCPClient(FAKE) as mcp:
        assert mcp.errors == {}
        by_name = {s["function"]["name"]: s for s in mcp.schemas}
        assert set(by_name) == {"fake__echo", "fake__boom", "fake__slow"}
        echo = by_name["fake__echo"]
        assert echo["type"] == "function"
        assert echo["function"]["description"] == "Echo text back."
        assert echo["function"]["parameters"]["required"] == ["text"]


def test_call_returns_text():
    with MCPClient(FAKE) as mcp:
        assert mcp.call("fake__echo", {"text": "hi"}) == {"text": "echo:hi"}


def test_tool_error_comes_back_as_error_dict():
    with MCPClient(FAKE) as mcp:
        result = mcp.call("fake__boom", {})
        assert "error" in result and "text" not in result


def test_unknown_tool_is_error_not_raise():
    with MCPClient(FAKE) as mcp:
        assert "unknown MCP tool" in mcp.call("fake__nope", {})["error"]


def test_call_timeout_is_error_not_hang():
    mcp_client.CALL_TIMEOUT = 0.5
    try:
        with MCPClient(FAKE) as mcp:
            assert "timed out" in mcp.call("fake__slow", {})["error"]
    finally:
        mcp_client.CALL_TIMEOUT = 60


def test_bad_server_is_reported_and_good_one_still_works():
    servers = {"bad": {"command": "/nonexistent/binary"}, **FAKE}
    with MCPClient(servers) as mcp:
        assert "bad" in mcp.errors and "fake" not in mcp.errors
        assert mcp.call("fake__echo", {"text": "ok"}) == {"text": "echo:ok"}


def test_allowlist_hides_other_tools_and_blocks_their_calls():
    servers = {"fake": {**FAKE["fake"], "tools": ["echo"]}}
    with MCPClient(servers) as mcp:
        assert [s["function"]["name"] for s in mcp.schemas] == ["fake__echo"]
        assert mcp.errors == {}
        assert mcp.call("fake__echo", {"text": "hi"}) == {"text": "echo:hi"}
        assert "unknown MCP tool" in mcp.call("fake__boom", {})["error"]


def test_allowlist_typo_is_reported_but_valid_names_still_load():
    servers = {"fake": {**FAKE["fake"], "tools": ["echo", "ecoh"]}}
    with MCPClient(servers) as mcp:
        assert "ecoh" in mcp.errors["fake"]
        assert [s["function"]["name"] for s in mcp.schemas] == ["fake__echo"]


def test_no_servers_is_a_noop():
    with MCPClient({}) as mcp:
        assert mcp.schemas == [] and mcp.errors == {}
        assert "unknown MCP tool" in mcp.call("x__y", {})["error"]


if __name__ == "__main__":
    if sys.argv[1:] == ["--server"]:
        _serve()
    else:
        for name, fn in list(globals().items()):
            if name.startswith("test_"):
                fn()
                print(f"ok  {name}")
        print("all passed")
