import os
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from mcp import StdioServerParameters

def get_it_mcp_tool():
    """Conecta el agente al servidor MCP local de gestión de tickets via Stdio (Subproceso)."""
    server_path = os.path.join(os.path.dirname(__file__), "server.py")
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="uv",
                args=["run", "python", server_path]
            )
        ),
        tool_filter=["crear_ticket_it"]
    )
