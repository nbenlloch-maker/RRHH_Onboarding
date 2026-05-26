from google.adk import Agent
from onboarding.mcp_server.client import get_it_mcp_tool
from onboarding.model_config import get_model

it_agent = Agent(
    model=get_model(),
    name="agente_it",
    description="Especialista en gestionar peticiones de hardware, accesos y problemas técnicos.",
    instruction=(
        "Eres el coordinador de IT corporativo. "
        "Si un usuario necesita material (monitor, portátil) o reporta un fallo, "
        "usa la herramienta MCP para registrar un ticket de soporte. "
        "Si el usuario no te da su nombre o la urgencia del problema, asume una urgencia 'Media'."
    ),
    tools=[get_it_mcp_tool()]
)
