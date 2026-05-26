from mcp.server.fastmcp import FastMCP
import json
import os
from datetime import datetime

mcp = FastMCP("TicketManager")
TICKETS_FILE = "tickets_it.json"

@mcp.tool()
def crear_ticket_it(empleado: str, equipo_necesitado: str, urgencia: str) -> str:
    """Crea un ticket de soporte IT para solicitar nuevo hardware o reportar incidencias."""
    ticket = {
        "fecha": str(datetime.now()),
        "empleado": empleado,
        "equipo_necesitado": equipo_necesitado,
        "urgencia": urgencia,
        "estado": "Abierto"
    }
    
    tickets = []
    if os.path.exists(TICKETS_FILE):
        with open(TICKETS_FILE, "r") as f:
            tickets = json.load(f)
            
    tickets.append(ticket)
    
    with open(TICKETS_FILE, "w") as f:
        json.dump(tickets, f, indent=4)
        
    return f"Éxito: Se ha generado el Ticket #{len(tickets)} de Soporte IT para {empleado}."

if __name__ == "__main__":
    print("Iniciando Servidor MCP TicketManager...")
    mcp.run()
