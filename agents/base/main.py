import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
GENERAL_MODEL = os.getenv("GENERAL_MODEL", "phi3:mini")
CODE_MODEL = os.getenv("CODE_MODEL", "qwen2.5-coder:3b")
CHAT_ENDPOINT = f"{OLLAMA}/api/chat"
GENERATE_ENDPOINT = f"{OLLAMA}/api/generate"
PULL_ENDPOINT = f"{OLLAMA}/api/pull"

app = FastAPI(title="Agent Base", version="0.1")


class ChatIn(BaseModel):
    message: str
    # "general" ou "code" (selection manuelle)
    profile: str = "general"
    # override explicite si tu veux (ex: "llama3.2:3b")
    model: str | None = None


def _extract_response_content(data: dict) -> str:
    return data.get("message", {}).get("content") or data.get("response", "") or ""


def _format_error(response: httpx.Response) -> str:
    snippet = (response.text or "").strip()
    if snippet:
        return snippet
    return f"Ollama returned {response.status_code} for {response.request.url}" if response.request else "Unexpected response from Ollama"


async def _pull_model(client: httpx.AsyncClient, model: str) -> None:
    pull_response = await client.post(PULL_ENDPOINT, json={"model": model})
    if pull_response.status_code == 404:
        raise HTTPException(status_code=502, detail=f"Modele '{model}' introuvable sur Ollama (pull).")
    pull_response.raise_for_status()


@app.get("/health")
def health():
    return {"ok": True, "general_model": GENERAL_MODEL, "code_model": CODE_MODEL}


@app.post("/chat")
async def chat(payload: ChatIn):
    model = payload.model or (CODE_MODEL if payload.profile.lower() == "code" else GENERAL_MODEL)
    chat_payload = {
        "model": model,
        "messages": [{"role": "user", "content": payload.message}],
        "stream": False,
    }
    generate_payload = {
        "model": model,
        "prompt": payload.message,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(CHAT_ENDPOINT, json=chat_payload)
        used_generate = False

        if response.status_code == 404:
            # Probablement une version d'Ollama sans /api/chat -> fallback generate.
            response = await client.post(GENERATE_ENDPOINT, json=generate_payload)
            used_generate = True
            if response.status_code == 404:
                # Modele manquant: on tente un pull automatique puis on relance.
                await _pull_model(client, model)
                response = await client.post(GENERATE_ENDPOINT, json=generate_payload)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            detail = _format_error(err.response)
            raise HTTPException(status_code=502, detail=detail) from err

        data = response.json()

    return {"used_model": model, "response": _extract_response_content(data), "used_generate": used_generate}
