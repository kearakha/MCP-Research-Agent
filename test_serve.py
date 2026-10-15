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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
