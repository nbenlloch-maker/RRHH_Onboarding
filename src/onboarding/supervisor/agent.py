from google.adk import Agent
from google.adk.tools.agent_tool import AgentTool
from onboarding.agents.hr_agent import hr_agent
from onboarding.agents.it_agent import it_agent
from onboarding.model_config import get_model

supervisor_agent = Agent(
    model=get_model(),
    name="supervisor_onboarding",
    description="Orquestador principal del sistema de Onboarding.",
    instruction=(
        "Eres el asistente supervisor de Onboarding. Tu trabajo es enrutar las consultas al sub-agente correcto.\n"
        "1. Para dudas de políticas, vacaciones, dietas o festivos, invoca a agente_rrhh.\n"
        "2. Para peticiones de ordenadores, monitores, o fallos técnicos, invoca a agente_it.\n"
        "3. Si la solicitud combina ambos temas, llama a ambos agentes secuencialmente.\n"
        "4. Si el usuario pregunta por temas totalmente ajenos a la empresa, declina educadamente la solicitud.\n"
        "Devuelve siempre una respuesta unificada, clara y profesional."
    ),
    tools=[
        AgentTool(agent=hr_agent),
        AgentTool(agent=it_agent),
    ],
)

# Expose as root_agent for Google ADK CLI / web UI runner compatibility
root_agent = supervisor_agent
