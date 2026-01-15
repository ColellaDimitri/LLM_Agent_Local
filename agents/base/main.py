import json
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
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

AGENT_OPTIONS = [
    {"id": "general", "label": f"General ({GENERAL_MODEL})"},
    {"id": "code", "label": f"Code ({CODE_MODEL})"},
]


class _EndpointUnavailable(Exception):
    """Raised when an Ollama endpoint (chat/generate) is missing."""


class _ModelMissing(Exception):
    """Raised when the requested model is not available locally."""


class ChatIn(BaseModel):
    message: str
    # "general" ou "code" (selection manuelle)
    profile: str = "general"
    # override explicite si tu veux (ex: "llama3.2:3b")
    model: str | None = None


def _extract_response_content(data: Dict[str, Any]) -> str:
    return data.get("message", {}).get("content") or data.get("response", "") or ""


def _resolve_model(payload: ChatIn) -> str:
    if payload.model:
        return payload.model
    if payload.profile.lower() == "code":
        return CODE_MODEL
    return GENERAL_MODEL


def _build_payloads(message: str, model: str, stream: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    chat_payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "stream": stream,
    }
    generate_payload = {
        "model": model,
        "prompt": message,
        "stream": stream,
    }
    return chat_payload, generate_payload


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


async def _post_generate_with_pull(
    client: httpx.AsyncClient,
    generate_payload: Dict[str, Any],
    model: str,
) -> httpx.Response:
    while True:
        response = await _post_with_timeout(client, GENERATE_ENDPOINT, generate_payload)
        if response.status_code != 404:
            return response
        logger.warning("Model %s missing on Ollama, attempting automatic pull", model)
        await _pull_model(client, model)


async def _stream_ollama(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    async with client.stream("POST", endpoint, json=payload) as response:
        if response.status_code == 404:
            if endpoint == CHAT_ENDPOINT:
                raise _EndpointUnavailable()
            raise _ModelMissing()

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            detail = _format_error(err.response)
            logger.error(
                "Streaming call failed endpoint=%s status=%s detail=%s",
                endpoint,
                err.response.status_code,
                detail,
            )
            raise HTTPException(status_code=502, detail=detail) from err

        async for line in response.aiter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Non-JSON chunk from %s: %s", endpoint, line)
                yield line
                continue
            if chunk.get("done"):
                break
            content = _extract_response_content(chunk)
            if content:
                yield content


async def _stream_generate_with_pull(
    client: httpx.AsyncClient,
    generate_payload: Dict[str, Any],
    model: str,
) -> AsyncGenerator[str, None]:
    while True:
        try:
            async for chunk in _stream_ollama(client, GENERATE_ENDPOINT, generate_payload):
                yield chunk
            return
        except _ModelMissing:
            logger.warning("Model %s missing on Ollama, attempting automatic pull", model)
            await _pull_model(client, model)


@app.get("/health")
def health():
    return {"ok": True, "general_model": GENERAL_MODEL, "code_model": CODE_MODEL}


@app.get("/agents")
def list_agents():
    return {"agents": AGENT_OPTIONS}


@app.get("/", response_class=HTMLResponse)
def index():
    options = "\n".join(
        f'<option value="{agent["id"]}">{agent["label"]}</option>' for agent in AGENT_OPTIONS
    )
    html = f"""<!DOCTYPE html>
<html lang=\"fr\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>LLM Console</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101218;
      --panel: #1b1f2a;
      --border: #2b3142;
      --accent: #5c7cfa;
      --text: #f2f4ff;
      --muted: #a3adc2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: clamp(0.8rem, 2vw, 2rem);
      background: radial-gradient(circle at top, #151929, var(--bg));
      font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      color: var(--text);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: stretch;
    }}
    .shell {{
      width: min(900px, 100%);
      min-height: calc(100vh - clamp(0.8rem, 2vw, 2rem)*2);
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.25rem;
      box-shadow: 0 25px 60px rgba(0,0,0,0.35);
    }}
    .panel.conversation {{
      flex: 1;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
    }}
    .title {{
      font-size: 1.25rem;
      font-weight: 600;
      letter-spacing: 0.04em;
    }}
    select {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      color: var(--text);
      font-size: 0.95rem;
      padding: 0.4rem 1rem;
      min-width: 220px;
    }}
    select:focus {{ outline: 1px solid var(--accent); }}
    #output {{
      flex: 1;
      min-height: 0;
      margin-top: 0.5rem;
      overflow-y: auto;
      line-height: 1.6;
      font-size: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      padding-right: 0.4rem;
    }}
    .message-row {{
      display: flex;
      width: 100%;
    }}
    .message-row.user {{ justify-content: flex-end; }}
    .message-row.assistant {{ justify-content: flex-start; }}
    .bubble {{
      max-width: 85%;
      padding: 0.85rem 1rem;
      border-radius: 18px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.25);
    }}
    .bubble.user {{
      background: var(--accent);
      color: #050712;
      border-bottom-right-radius: 6px;
    }}
    .bubble.assistant {{
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--border);
      border-bottom-left-radius: 6px;
      color: var(--text);
      white-space: pre-wrap;
    }}
    .status {{
      font-size: 0.9rem;
      color: var(--muted);
      min-height: 1.1rem;
    }}
    form {{
      display: flex;
      gap: 0.75rem;
    }}
    input[type=text] {{
      flex: 1;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 0.9rem 1.2rem;
      font-size: 1rem;
      color: var(--text);
    }}
    input[type=text]:focus {{ outline: 1px solid var(--accent); }}
    button {{
      background: linear-gradient(120deg, var(--accent), #82a0ff);
      color: #0a0d18;
      border: none;
      border-radius: 999px;
      padding: 0 1.5rem;
      font-weight: 600;
      font-size: 1rem;
      cursor: pointer;
      transition: transform 120ms ease;
    }}
    button:active {{ transform: scale(0.97); }}
  </style>
</head>
<body>
  <div class=\"shell\">
    <div class=\"panel conversation\">
      <div class=\"header\">
        <div class=\"title\">LLM Console</div>
        <div>
          <label for=\"agentSelect\" style=\"font-size:0.85rem;color:var(--muted);display:block;margin-bottom:0.2rem;\">Agent</label>
          <select id=\"agentSelect\" aria-label=\"Choisir un agent\">{options}</select>
        </div>
      </div>
      <div id=\"status\" class=\"status\"></div>
      <div id=\"output\"></div>
    </div>
    <form id=\"promptForm\" class=\"panel\">
      <input id=\"promptInput\" type=\"text\" placeholder=\"Pose ta question...\" autocomplete=\"off\" required />
      <button type=\"submit\">Envoyer</button>
    </form>
  </div>
  <script>
    const form = document.getElementById('promptForm');
    const promptInput = document.getElementById('promptInput');
    const output = document.getElementById('output');
    const status = document.getElementById('status');
    const agentSelect = document.getElementById('agentSelect');
    let controller = null;

    function appendBubble(text, role) {{
      const row = document.createElement('div');
      row.className = `message-row ${role}`;
      const bubble = document.createElement('div');
      bubble.className = `bubble ${role}`;
      bubble.textContent = text;
      row.appendChild(bubble);
      output.appendChild(row);
      output.scrollTop = output.scrollHeight;
      return bubble;
    }}

    async function sendPrompt(evt) {{
      evt.preventDefault();
      const message = promptInput.value.trim();
      if (!message) return;

      if (controller) controller.abort();
      controller = new AbortController();
      const signal = controller.signal;
      status.textContent = 'Generation en cours... (ECHAP pour annuler)';
      const userBubble = appendBubble(message, 'user');
      const assistantBubble = appendBubble('', 'assistant');
      promptInput.value = '';

      try {{
        const response = await fetch('/chat-stream', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ message, profile: agentSelect.value }}),
          signal,
        }});

        if (!response.ok || !response.body) {{
          throw new Error('Requete rejetee: ' + response.statusText);
        }}

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {{
          const {{ value, done }} = await reader.read();
          if (done) break;
          assistantBubble.textContent += decoder.decode(value, {{ stream: true }});
          output.scrollTop = output.scrollHeight;
        }}
        status.textContent = 'Termine';
      }} catch (err) {{
        if (err.name === 'AbortError') {{
          status.textContent = 'Generation interrompue';
          if (!assistantBubble.textContent) {{
            assistantBubble.textContent = '[Generation interrompue]';
          }}
        }} else {{
          status.textContent = 'Erreur: ' + err.message;
          assistantBubble.textContent = '⚠️ ' + err.message;
        }}
      }} finally {{
        controller = null;
        promptInput.focus();
      }}
    }}

    form.addEventListener('submit', sendPrompt);
    document.addEventListener('keydown', (event) => {{
      if (event.key === 'Escape' && controller) {{
        controller.abort();
      }}
    }});
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)

@app.post("/chat")
async def chat(payload: ChatIn):
    model = _resolve_model(payload)
    logger.info(
        "Incoming chat request profile=%s explicit_model=%s resolved_model=%s",
        payload.profile,
        payload.model,
        model,
    )

    chat_payload, generate_payload = _build_payloads(payload.message, model, stream=False)

    used_generate = False
    endpoint_used = "/api/chat"
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
        response = await _post_with_timeout(client, CHAT_ENDPOINT, chat_payload)

        if response.status_code == 404:
            logger.warning("/api/chat not available (status 404), falling back to /api/generate")
            response = await _post_generate_with_pull(client, generate_payload, model)
            endpoint_used = "/api/generate"
            used_generate = True

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            detail = _format_error(err.response)
            logger.error(
                "Ollama call failed endpoint=%s status=%s detail=%s",
                endpoint_used,
                err.response.status_code,
                detail,
            )
            raise HTTPException(status_code=502, detail=detail) from err

        data = response.json()

    content = _extract_response_content(data)
    duration = time.perf_counter() - start
    logger.info(
        "chat served endpoint=%s used_generate=%s duration=%.2fs chars=%d",
        endpoint_used,
        used_generate,
        duration,
        len(content),
    )

    return {"used_model": model, "response": content, "used_generate": used_generate}


@app.post("/chat-stream")
async def chat_stream(payload: ChatIn):
    model = _resolve_model(payload)
    logger.info(
        "Incoming streaming request profile=%s explicit_model=%s resolved_model=%s",
        payload.profile,
        payload.model,
        model,
    )

    chat_payload, generate_payload = _build_payloads(payload.message, model, stream=True)

    start = time.perf_counter()
    used_generate = False
    total_chars = 0
    endpoint_used = "/api/chat"

    async def event_stream() -> AsyncGenerator[str, None]:
        nonlocal used_generate, total_chars, endpoint_used
        try:
            async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
                try:
                    async for chunk in _stream_ollama(client, CHAT_ENDPOINT, chat_payload):
                        total_chars += len(chunk)
                        yield chunk
                    return
                except _EndpointUnavailable:
                    logger.warning("/api/chat not available for streaming, falling back to /api/generate")
                    used_generate = True
                    endpoint_used = "/api/generate"

                async for chunk in _stream_generate_with_pull(client, generate_payload, model):
                    total_chars += len(chunk)
                    yield chunk
        finally:
            duration = time.perf_counter() - start
            logger.info(
                "chat-stream served endpoint=%s duration=%.2fs chars=%d",
                endpoint_used,
                duration,
                total_chars,
            )

    response = StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")
    response.headers["X-Used-Model"] = model
    response.headers["Cache-Control"] = "no-store"
    return response
