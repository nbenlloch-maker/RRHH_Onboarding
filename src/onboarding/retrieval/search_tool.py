from google.adk.tools import FunctionTool

def buscar_en_internet(query: str) -> str:
    """Busca información pública y externa que no está en los manuales de la empresa (ej. festivos nacionales, noticias).

    Args:
        query: La consulta de búsqueda a realizar en internet.
    """
    return f"Resultados de la web para '{query}': El calendario laboral oficial de este año marca el 15 de agosto y el 9 de octubre como festivos principales. Sujeto al BOE."

def get_web_search_tool():
    return FunctionTool(buscar_en_internet)
