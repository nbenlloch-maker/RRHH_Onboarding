# Asistente de Onboarding (Sistema Multiagente)

Este proyecto implementa una arquitectura por capas que orquesta:
- **Capa de Orquestación:** Un agente supervisor.
- **Capa de Agentes:** Sub-agentes de RRHH e IT.
- **Capa de Herramientas:** RAG local (FAISS), búsqueda web y servidor MCP.

## Instalación y Ejecución
1. Instala las dependencias: `uv sync`
2. Copia `.env.example` a `.env` y añade tu API Key.
3. En la Terminal 1, lanza el servidor MCP: `uv run python src/onboarding/mcp_server/server.py`
4. En la Terminal 2, lanza la UI del orquestador: `uv run adk web src/onboarding/supervisor/agent.py`