import os
from langchain_openai import ChatOpenAI

def llm_for_role(role: str, model_id: str, temperature: float = 0.3):
    # Pick agent-specific key if available; fallback to GEMINI_API_KEY
    key_env = {
        "gen_a": "GEMINI_API_KEY_GEN_A",
        "gen_b": "GEMINI_API_KEY_GEN_B",
        "supervisor": "GEMINI_API_KEY_SUP",
        "exec": "GEMINI_API_KEY_EXEC",
        "researcher": "GEMINI_API_KEY_RES",
    }.get(role, "GEMINI_API_KEY")
    api_key = os.getenv(key_env) or os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/openai")

    if api_key is None:
        raise RuntimeError(f"Missing API key for role {role} (env {key_env} or GEMINI_API_KEY)")

    return ChatOpenAI(
        model=model_id,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        timeout=120,
        max_retries=2,
    )