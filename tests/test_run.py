import os
import sys
import asyncio
from pathlib import Path

# Ensure repo root and src/ are in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env file
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from google.adk.runners import InMemoryRunner
from google.genai import types
from onboarding.supervisor.agent import root_agent

async def test_query(input_text: str):
    print(f"\n=== User Query: '{input_text}' ===")
    runner = InMemoryRunner(agent=root_agent, app_name="test_app")
    session = await runner.session_service.create_session(
        app_name="test_app", user_id="test_user"
    )
    
    async for event in runner.run_async(
        user_id="test_user",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=input_text)]),
    ):
        for fc in event.get_function_calls():
            print(f"  [Tool Call] -> {fc.name}({fc.args})")
        for fr in event.get_function_responses():
            print(f"  [Tool Response] <- {fr.name}: {fr.response}")
        if event.content:
            for part in event.content.parts:
                if part.text:
                    if event.is_final_response():
                        print(f"  [Final Response] -> {part.text}")
                    else:
                        print(f"  [Thought] -> {part.text}")

async def main():
    # 1. Ask HR agent about vacations
    await test_query("¿Cuántos días de vacaciones al año tengo permitidos?")
    
    # 2. Ask IT agent to request hardware
    await test_query("Necesito solicitar un monitor nuevo para mi puesto. Mi nombre es Juan.")
    
    # 3. Ask unrelated query
    await test_query("¿Cómo se prepara una tortilla de patatas?")

if __name__ == "__main__":
    asyncio.run(main())
