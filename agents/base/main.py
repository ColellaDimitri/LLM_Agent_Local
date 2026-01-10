import logging
import os
import time
from typing import Any, Dict

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


def _parse_timeout(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        return 600.0
    return max(0.0, seconds)


HTTP_TIMEOUT_SECONDS = _parse_timeout(os.getenv("OLLAMA_HTTP_TIMEOUT", "900"))
CONNECT_TIMEOUT = 30.0 if HTTP_TIMEOUT_SECONDS == 0 else min(30.0, HTTP_TIMEOUT_SECONDS)
HTTPX_TIMEOUT = None
if HTTP_TIMEOUT_SECONDS > 0:
    HTTPX_TIMEOUT = httpx.Timeout(
        HTTP_TIMEOUT_SECONDS,
        connect=CONNECT_TIMEOUT,
        read=HTTP_TIMEOUT_SECONDS,
        write=HTTP_TIMEOUT_SECONDS,
    )

app = FastAPI(title="Agent Base", version="0.1")


class ChatIn(BaseModel):
    message: str
    # "general" ou "code" (selection manuelle)
    profile: str = "general"
    # override explicite si tu veux (ex: "llama3.2:3b")
    model: str | None = None


def _extract_response_content(data: Dict[str, Any]) -> str:
    return data.get("message", {}).get("content") or data.get("response", "") or ""


def _format_error(response: httpx.Response) -> str:
    snippet = (response.text or "").strip()
    if snippet:
        return snippet[:500]
    if response.request is not None:
        return f"Ollama returned {response.status_code} for {response.request.url}"
    return f"Unexpected response from Ollama (status {response.status_code})"


def _timeout_detail(endpoint: str) -> str:
    if HTTP_TIMEOUT_SECONDS == 0:
        return f"Ollama n'a pas repondu (timeout illimite) sur {endpoint}"
    seconds = int(HTTP_TIMEOUT_SECONDS)
    return (
        f"Ollama n'a pas repondu apres {seconds}s sur {endpoint}. "
        "Augmente OLLAMA_HTTP_TIMEOUT si besoin."
    )


async def _post_with_timeout(client: httpx.AsyncClient, endpoint: str, payload: Dict[str, Any]) -> httpx.Response:
    try:
        return await client.post(endpoint, json=payload)
    except httpx.ReadTimeout as err:
        detail = _timeout_detail(endpoint)
        logger.error("Timeout contacting %s after %.1fs", endpoint, HTTP_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail=detail) from err


async def _pull_model(client: httpx.AsyncClient, model: str) -> None:
    logger.info("Pulling missing model %s", model)
    pull_response = await _post_with_timeout(client, PULL_ENDPOINT, {"model": model})
    if pull_response.status_code == 404:
        raise HTTPException(status_code=502, detail=f"Modele '{model}' introuvable sur Ollama (pull)")
    pull_response.raise_for_status()


@app.get("/health")
def health():
    return {"ok": True, "general_model": GENERAL_MODEL, "code_model": CODE_MODEL}


@app.post("/chat")
async def chat(payload: ChatIn):
    model = payload.model or (CODE_MODEL if payload.profile.lower() == "code" else GENERAL_MODEL)
    logger.info(
        "Incoming chat request profile=%s explicit_model=%s resolved_model=%s",
        payload.profile,
        payload.model,
        model,
    )

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

    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
        response = await _post_with_timeout(client, CHAT_ENDPOINT, chat_payload)

        if response.status_code == 404:
            logger.warning("/api/chat not available (status 404), falling back to /api/generate")
            response = await _post_with_timeout(client, GENERATE_ENDPOINT, generate_payload)
            chosen_endpoint = GENERATE_ENDPOINT
            used_generate = True
            if response.status_code == 404:
                logger.warning("Model %s missing on Ollama, attempting automatic pull", model)
                await _pull_model(client, model)
                response = await _post_with_timeout(client, GENERATE_ENDPOINT, generate_payload)

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
