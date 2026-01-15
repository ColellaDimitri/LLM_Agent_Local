import json
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Tuple

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
    {"id": "agent", "label": "Agent (multi-outils)"},
]

AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "4"))
AGENT_PLANNER_PROMPT = (
    "Tu es un agent autonome. Tu raisonnes en plusieurs etapes. "
    "Format JSON attendu: {\n"
    "  \"thinking\": \"...\",\n"
    "  \"action\": \"call_model\" ou \"final_answer\",\n"
    "  \"target\": \"general\" ou \"code\" (requis si action=call_model),\n"
    "  \"prompt\": \"texte\" (requis si action=call_model),\n"
    "  \"final_answer\": \"texte\" (requis si action=final_answer)\n"
    "}. Tu dois suivre strictement ce schema."
)


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
    # historique optionnel [{"role": "user|assistant|system", "content": "..."}]
    history: List[Dict[str, str]] | None = None


class AgentIn(BaseModel):
    message: str
    history: List[Dict[str, str]] | None = None


def _extract_response_content(data: Dict[str, Any]) -> str:
    return data.get("message", {}).get("content") or data.get("response", "") or ""


def _resolve_model(payload: ChatIn) -> str:
    if payload.model:
        return payload.model
    if payload.profile.lower() == "code":
        return CODE_MODEL
    return GENERAL_MODEL


def _normalize_history(history: List[Dict[str, str]] | None) -> List[Dict[str, str]]:
    if not history:
        return []
    normalized: List[Dict[str, str]] = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "")).lower()
        content = turn.get("content")
        if role not in {"user", "assistant", "system"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _history_prompt(history: List[Dict[str, str]], message: str) -> str:
    if not history:
        return message
    lines: List[str] = []
    role_map = {"user": "Utilisateur", "assistant": "Assistant", "system": "Systeme"}
    for turn in history:
        label = role_map.get(turn["role"], turn["role"].capitalize())
        lines.append(f"{label}: {turn['content']}")
    lines.append(f"Utilisateur: {message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _format_history_for_agent(history: List[Dict[str, str]]) -> str:
    if not history:
        return "Aucun contexte precedent."
    segments = []
    role_map = {"user": "Utilisateur", "assistant": "Assistant", "system": "Systeme"}
    for turn in history[-8:]:
        label = role_map.get(turn["role"], turn["role"].capitalize())
        segments.append(f"{label}: {turn['content']}")
    return "\n".join(segments)


def _build_payloads(
    message: str,
    model: str,
    stream: bool,
    history: List[Dict[str, str]] | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    normalized_history = _normalize_history(history)
    messages = [*normalized_history, {"role": "user", "content": message}]
    chat_payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    generate_payload = {
        "model": model,
        "prompt": _history_prompt(normalized_history, message),
        "stream": stream,
    }
    return chat_payload, generate_payload


async def _invoke_generate_text(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = await _post_generate_with_pull(client, payload, model)
    response.raise_for_status()
    data = response.json()
    return _extract_response_content(data)


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


def _extract_json_block(text: str) -> Dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _parse_agent_directive(raw: str) -> Dict[str, Any]:
    data = _extract_json_block(raw)
    if not data:
        return {"action": "final_answer", "final_answer": raw.strip(), "thinking": raw.strip()}
    return data


def _agent_prompt(task: str, steps: List[Dict[str, Any]], history: List[Dict[str, str]]) -> str:
    steps_dump = json.dumps(steps, ensure_ascii=False, indent=2) if steps else "Aucune"
    history_text = _format_history_for_agent(history)
    return (
        f"{AGENT_PLANNER_PROMPT}\n\n"
        f"Contexte utilisateur:\n{history_text}\n\n"
        f"Tache actuelle: {task}\n\n"
        f"Etapes deja effectuees:\n{steps_dump}\n\n"
        "Choisis l'action appropriee."
    )


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
    .bubble.trace {{
      background: rgba(255,255,255,0.02);
      color: var(--muted);
      border-style: dashed;
      font-size: 0.9rem;
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
    const history = [];

    function appendBubble(text, role) {{
      const row = document.createElement('div');
      row.className = 'message-row ' + role;
      const bubble = document.createElement('div');
      bubble.className = 'bubble ' + role;
      bubble.textContent = text;
      row.appendChild(bubble);
      output.appendChild(row);
      output.scrollTop = output.scrollHeight;
      return bubble;
    }}

    function renderAgentTraceEntry(step) {{
      const parts = [];
      if (step.step) parts.push('Etape ' + step.step);
      if (step.thinking) parts.push('[Reflexion] ' + step.thinking);
      if (step.action === 'call_model') {{
        parts.push('[Action] appel ' + (step.target || 'general'));
        if (step.prompt) parts.push('Prompt envoye:\\n' + step.prompt);
        if (step.result) parts.push('Resultat:\\n' + step.result);
      }}
      const text = parts.join('\\n\\n').trim();
      if (!text) return;
      const bubble = appendBubble(text, 'assistant');
      bubble.classList.add('trace');
    }}

    async function sendPrompt(evt) {{
      evt.preventDefault();
      const message = promptInput.value.trim();
      if (!message) return;

      if (controller) controller.abort();
      controller = new AbortController();
      const signal = controller.signal;
      const isAgentic = agentSelect.value === 'agent';
      const statusRunning = isAgentic ? 'Agent en reflexion...' : 'Generation en cours... (ECHAP pour annuler)';
      status.textContent = statusRunning;
      appendBubble(message, 'user');
      let assistantBubble = null;
      if (!isAgentic) {{
        assistantBubble = appendBubble('', 'assistant');
      }}
      promptInput.value = '';

      const payload = {{
        message,
        profile: agentSelect.value,
        history: history.map(function (turn) {{
          return {{ role: turn.role, content: turn.content }};
        }}),
      }};

      try {{
        if (isAgentic) {{
          await runAgenticConversation(payload, signal);
        }} else {{
          await runStreamingConversation(payload, assistantBubble, signal);
        }}
      }} catch (err) {{
        if (err.name === 'AbortError') {{
          status.textContent = isAgentic ? 'Agent interrompu' : 'Generation interrompue';
          if (assistantBubble && !assistantBubble.textContent) {{
            assistantBubble.textContent = '[Generation interrompue]';
          }}
        }} else {{
          status.textContent = 'Erreur: ' + err.message;
          if (assistantBubble) {{
            assistantBubble.textContent = '[Erreur] ' + err.message;
          }} else {{
            const errorBubble = appendBubble('[Erreur] ' + err.message, 'assistant');
            errorBubble.classList.add('trace');
          }}
        }}
      }} finally {{
        controller = null;
        promptInput.focus();
      }}
    }}

    async function runStreamingConversation(payload, assistantBubble, signal) {{
      const response = await fetch('/chat-stream', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload),
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
      history.push({{ role: 'user', content: payload.message }});
      history.push({{ role: 'assistant', content: assistantBubble.textContent }});
    }}

    async function runAgenticConversation(payload, signal) {{
      const response = await fetch('/agent', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ message: payload.message, history: payload.history }}),
        signal,
      }});
      if (!response.ok) {{
        const errorText = await response.text();
        throw new Error(errorText || response.statusText);
      }}
      const data = await response.json();
      if (Array.isArray(data.trace)) {{
        data.trace.forEach(renderAgentTraceEntry);
      }}
      const finalText = (data.final_answer || '').trim() || '[Aucune reponse]';
      appendBubble(finalText, 'assistant');
      status.textContent = 'Agent termine';
      history.push({{ role: 'user', content: payload.message }});
      history.push({{ role: 'assistant', content: finalText }});
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

    chat_payload, generate_payload = _build_payloads(
        payload.message,
        model,
        stream=False,
        history=payload.history,
    )

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

    chat_payload, generate_payload = _build_payloads(
        payload.message,
        model,
        stream=True,
        history=payload.history,
    )

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


@app.post("/agent")
async def agent(payload: AgentIn):
    normalized_history = _normalize_history(payload.history)
    steps: List[Dict[str, Any]] = []
    final_answer: str | None = None

    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
        for step_index in range(1, AGENT_MAX_STEPS + 1):
            planner_prompt = _agent_prompt(payload.message, steps, normalized_history)
            logger.info("[agent] Step %s planning", step_index)
            plan_text = await _invoke_generate_text(client, GENERAL_MODEL, planner_prompt)
            directive = _parse_agent_directive(plan_text)
            action = (directive.get("action") or "").lower()
            thinking = directive.get("thinking", "").strip()
            step_info: Dict[str, Any] = {
                "step": step_index,
                "action": action,
                "thinking": thinking,
            }

            if action == "call_model":
                target = (directive.get("target") or "general").lower()
                model = GENERAL_MODEL if target != "code" else CODE_MODEL
                prompt = directive.get("prompt") or payload.message
                step_info.update({"target": target, "prompt": prompt})
                logger.info("[agent] Step %s calling %s", step_index, model)
                result = await _invoke_generate_text(client, model, prompt)
                step_info["result"] = result
                steps.append(step_info)
                continue

            if action == "final_answer":
                final_answer = directive.get("final_answer") or directive.get("answer") or thinking
                step_info["final_answer"] = final_answer
                steps.append(step_info)
                break

            # fallback: treat as final
            final_answer = plan_text.strip()
            step_info["final_answer"] = final_answer
            steps.append(step_info)
            break

    if not final_answer:
        raise HTTPException(status_code=500, detail="Agent n'a pas pu produire de reponse")

    logger.info("[agent] Completed task with %d step(s)", len(steps))
    return {"final_answer": final_answer, "trace": steps}
