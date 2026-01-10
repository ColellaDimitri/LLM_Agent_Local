import logging
import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
GENERAL_MODEL = os.getenv("GENERAL_MODEL", "phi3:mini")
CODE_MODEL = os.getenv("CODE_MODEL", "qwen2.5-coder:3b")
CHAT_ENDPOINT = f"{OLLAMA}/api/chat"
GENERATE_ENDPOINT = f"{OLLAMA}/api/generate"
PULL_ENDPOINT = f"{OLLAMA}/api/pull"
LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)
logging.basicConfig(level=LOG_LEVEL, format="[agent-base] %(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent-base")
logger.setLevel(LOG_LEVEL)

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
        return snippet[:500]
    if response.request is not None:
        return f"Ollama returned {response.status_code} for {response.request.url}"
    return f"Unexpected response from Ollama (status {response.status_code})"


async def _pull_model(client: httpx.AsyncClient, model: str) -> None:
    logger.info("Pulling missing model %s", model)
    pull_response = await client.post(PULL_ENDPOINT, json={"model": model})
    if pull_response.status_code == 404:
        raise HTTPException(status_code=502, detail=f"Modele '{model}' introuvable sur Ollama (pull)")
    pull_response.raise_for_status()


@app.get("/health")
def health():
    return {"ok": True, "general_model": GENERAL_MODEL, "code_model": CODE_MODEL}


@app.post("/chat")
async def chat(payload: ChatIn):
    model = payload.model or (CODE_MODEL if payload.profile.lower() == "code" else GENERAL_MODEL)
    logger.info("Incoming chat request profile=%s explicit_model=%s resolved_model=%s", payload.profile, payload.model, model)
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
    used_generate = False
    chosen_endpoint = CHAT_ENDPOINT
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(CHAT_ENDPOINT, json=chat_payload)

        if response.status_code == 404:
            logger.warning("/api/chat not available (status 404), falling back to /api/generate")
            response = await client.post(GENERATE_ENDPOINT, json=generate_payload)
            chosen_endpoint = GENERATE_ENDPOINT
            used_generate = True
            if response.status_code == 404:
                logger.warning("Model %s missing on Ollama, attempting automatic pull", model)
                await _pull_model(client, model)
                response = await client.post(GENERATE_ENDPOINT, json=generate_payload)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            detail = _format_error(err.response)
            logger.error(
                "Ollama call failed endpoint=%s status=%s detail=%s",
                chosen_endpoint,
                err.response.status_code,
                detail,
            )
            raise HTTPException(status_code=502, detail=detail) from err

        data = response.json()

    content = _extract_response_content(data)
    duration = time.perf_counter() - start
    logger.info(
        "chat served endpoint=%s used_generate=%s duration=%.2fs chars=%d",
        "/api/generate" if used_generate else "/api/chat",
        used_generate,
        duration,
        len(content),
    )

    return {"used_model": model, "response": content, "used_generate": used_generate}
