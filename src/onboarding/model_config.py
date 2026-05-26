"""Model configuration — switch between Gemini, Groq, and Vertex via MODEL_PROVIDER env var."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

def load_env_file() -> None:
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

# Load env variables at import time
load_env_file()

def get_model():
    provider = os.getenv("MODEL_PROVIDER", "gemini").lower()

    if provider == "groq":
        from google.adk.models.lite_llm import LiteLlm
        return LiteLlm(model=os.getenv("GROQ_MODEL", "groq/qwen/qwen3-32b"))

    if provider == "gemini":
        from google.adk.models.lite_llm import LiteLlm
        model_name = os.getenv("GEMINI_MODEL", "gemma-3-27b-it")
        if not model_name.startswith("gemini/"):
            model_name = "gemini/" + model_name
        return LiteLlm(model=model_name)

    if provider == "vertex":
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
        return os.getenv("VERTEX_MODEL", "gemini-2.5-flash-lite")

    return os.getenv("GEMINI_MODEL", "gemma-3-27b-it")


def get_embedding_model():
    """Return the embedding model configured for the current provider."""
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

    provider = os.getenv("MODEL_PROVIDER", "gemini").lower()
    default_model = "text-embedding-004" if provider == "vertex" else "gemini-embedding-001"
    model_name = os.getenv("LOCAL_RAG_EMBEDDING_MODEL", default_model)

    if provider == "vertex":
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "").strip()
        if not project or not location:
            raise RuntimeError(
                "Vertex embeddings require GOOGLE_CLOUD_PROJECT and "
                "GOOGLE_CLOUD_LOCATION env vars."
            )

        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
        return GoogleGenAIEmbedding(
            model_name=model_name,
            vertexai_config={"project": project, "location": location},
        )

    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not google_api_key:
        raise RuntimeError(
            "Gemini API embeddings require GOOGLE_API_KEY. To use Vertex AI "
            "embeddings, set MODEL_PROVIDER=vertex plus GOOGLE_CLOUD_PROJECT "
            "and GOOGLE_CLOUD_LOCATION."
        )

    return GoogleGenAIEmbedding(
        model_name=model_name,
        api_key=google_api_key,
    )
