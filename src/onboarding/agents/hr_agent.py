from google.adk import Agent
from onboarding.retrieval.rag_tool import get_hr_rag_tool
from onboarding.retrieval.search_tool import get_web_search_tool
from onboarding.model_config import get_model

hr_agent = Agent(
    model=get_model(),
    name="agente_rrhh",
    description="Especialista en resolver dudas sobre políticas de empresa, vacaciones y normativa interna.",
    instruction=(
        "Eres un consultor experto en Recursos Humanos. "
        "Usa la herramienta 'consultar_manual_rrhh' para buscar respuestas en las políticas internas. "
        "Usa 'buscar_internet' solo si te preguntan por normativas públicas o festivos nacionales. "
        "Responde siempre de manera precisa y corporativa."
    ),
    tools=[get_hr_rag_tool(), get_web_search_tool()]
)
