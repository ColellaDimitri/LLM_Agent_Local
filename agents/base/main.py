import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
GENERAL_MODEL = os.getenv("GENERAL_MODEL", "phi3:mini")
CODE_MODEL = os.getenv("CODE_MODEL", "qwen2.5-coder:3b")
CHAT_ENDPOINT = f"{OLLAMA}/api/chat"
GENERATE_ENDPOINT = f"{OLLAMA}/api/generate"

app = FastAPI(title="Agent Base", version="0.1")


class ChatIn(BaseModel):
    message: str
    # "general" ou "code" (selection manuelle)
    profile: str = "general"
    # override explicite si tu veux (ex: "llama3.2:3b")
    model: str | None = None


@app.get("/health")
def health():
    return {"ok": True, "general_model": GENERAL_MODEL, "code_model": CODE_MODEL}


@app.post("/chat")
async def chat(payload: ChatIn):
    if payload.model:
        model = payload.model
    else:
        model = CODE_MODEL if payload.profile.lower() == "code" else GENERAL_MODEL

    chat_payload = {
        "model": model,
        "messages": [{"role": "user", "content": payload.message}],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(CHAT_ENDPOINT, json=chat_payload)
        if response.status_code == 404:
            # Fallback pour les versions d'Ollama sans /api/chat
            generate_payload = {
                "model": model,
                "prompt": payload.message,
                "stream": False,
            }
            response = await client.post(GENERATE_ENDPOINT, json=generate_payload)
        response.raise_for_status()
        data = response.json()

    content = data.get("message", {}).get("content") or data.get("response", "")
    return {"used_model": model, "response": content}
