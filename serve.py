"""The research agent as an MCP server: one tool, research(goal) -> markdown report.

    python serve.py        # stdio, for MCP clients that spawn it

Claude Code: claude mcp add research -- python /path/to/serve.py
The agent's own MCP servers (mcp_servers.json) start alongside, so the
research it does can use fetch/Tavily too. Logs go to stderr — stdout belongs
to the MCP protocol, never print() there.
"""
import threading

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import agent

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


def main():
    agent.load_dotenv()
    with agent.mcp_tools():
        server.run("stdio")


if __name__ == "__main__":
    main()
