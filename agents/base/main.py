import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
GENERAL_MODEL = os.getenv("GENERAL_MODEL", "phi3:mini")
CODE_MODEL = os.getenv("CODE_MODEL", "qwen2.5-coder:3b")

app = FastAPI(title="Agent Base", version="0.1")

class ChatIn(BaseModel):
    message: str
    # "general" ou "code" (sélection manuelle)
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

    data = {
        "model": model,
        "messages": [{"role": "user", "content": payload.message}],
        "stream": False
    }

    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{OLLAMA}/api/chat", json=data)
        r.raise_for_status()
        j = r.json()

    return {"used_model": model, "response": j.get("message", {}).get("content", "")}
