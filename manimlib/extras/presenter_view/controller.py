from __future__ import annotations

import inspect
import json
import os
import queue
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from manimlib.logger import log

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

if TYPE_CHECKING:
    from manimlib.scene.scene import Scene


_INDEX_HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #0f1722;
      --panel: #162235;
      --panel-soft: #1c2d45;
      --line: #284363;
      --text: #eaf2ff;
      --muted: #9eb5d8;
      --accent: #39c88f;
      --warn: #f7c948;
      --danger: #f26d6d;
      --radius: 14px;
      --shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: \"Segoe UI\", Tahoma, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 20% 10%, rgba(57, 200, 143, 0.15), transparent 40%),
        radial-gradient(circle at 85% 0%, rgba(80, 126, 255, 0.2), transparent 35%),
        linear-gradient(160deg, #0c141f 0%, #0f1722 100%);
      padding: 18px;
    }

    .shell {
      max-width: 1080px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 14px;
    }

    .panel.span-2 {
      grid-column: 1 / -1;
    }

    .panel {
      background: linear-gradient(180deg, var(--panel) 0%, var(--panel-soft) 100%);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 16px;
    }

    .panel h1,
    .panel h2 {
      margin: 0 0 10px;
      font-weight: 600;
      letter-spacing: 0.2px;
    }

    .panel h1 {
      font-size: 1.2rem;
    }

    .panel h2 {
      font-size: 1rem;
      color: var(--muted);
    }

    .status {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }

    .status-item {
      padding: 10px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: rgba(6, 17, 31, 0.3);
    }

    .status-label {
      font-size: 0.78rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }

    .status-value {
      font-size: 1rem;
      font-weight: 600;
      word-break: break-word;
    }

    .controls {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }

    button {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      background: #173049;
      color: var(--text);
      font-weight: 600;
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
    }

    button:hover {
      transform: translateY(-1px);
      border-color: #4d73a4;
      background: #1c3a58;
    }

    button[data-action=\"next\"] {
      border-color: rgba(57, 200, 143, 0.65);
      background: rgba(57, 200, 143, 0.2);
    }

    button[data-action=\"pause\"],
    button[data-action=\"toggle_hold\"] {
      border-color: rgba(247, 201, 72, 0.6);
      background: rgba(247, 201, 72, 0.18);
    }

    button[data-action=\"quit\"] {
      border-color: rgba(242, 109, 109, 0.7);
      background: rgba(242, 109, 109, 0.16);
    }

    .note-form {
      display: grid;
      gap: 8px;
    }

    .field-label {
      color: var(--muted);
      font-size: 0.82rem;
      margin-top: 4px;
    }

    textarea {
      width: 100%;
      min-height: 96px;
      resize: vertical;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: rgba(10, 19, 30, 0.7);
      color: var(--text);
      padding: 10px;
      font: inherit;
    }

    select {
      border-radius: 10px;
      border: 1px solid var(--line);
      background: rgba(10, 19, 30, 0.7);
      color: var(--text);
      padding: 9px 10px;
      font: inherit;
    }

    .row-controls {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }

    .notes-list {
      margin: 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 8px;
      max-height: 58vh;
      overflow: auto;
    }

    .note-item {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: rgba(8, 18, 30, 0.5);
    }

    .note-meta {
      margin-bottom: 5px;
      color: var(--muted);
      font-size: 0.8rem;
    }

    .hint {
      color: var(--muted);
      margin-top: 10px;
      font-size: 0.86rem;
    }

    @media (max-width: 900px) {
      .shell {
        grid-template-columns: 1fr;
      }

      .row-controls {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class=\"shell\">
    <section class=\"panel\">
      <h1>__TITLE__</h1>
      <div class=\"status\">
        <div class=\"status-item\">
          <div class=\"status-label\">Scene</div>
          <div id=\"scene-name\" class=\"status-value\">-</div>
        </div>
        <div class=\"status-item\">
          <div class=\"status-label\">Hold State</div>
          <div id=\"hold-state\" class=\"status-value\">-</div>
        </div>
        <div class=\"status-item\">
          <div class=\"status-label\">Play Count</div>
          <div id=\"play-count\" class=\"status-value\">0</div>
        </div>
        <div class=\"status-item\">
          <div class=\"status-label\">Scene Time</div>
          <div id=\"scene-time\" class=\"status-value\">0.00s</div>
        </div>
      </div>

      <div class=\"controls\">
        <button data-action=\"previous\">Previous</button>
        <button data-action=\"next\">Next</button>
        <button data-action=\"pause\">Pause</button>
        <button data-action=\"resume\">Resume</button>
        <button data-action=\"toggle_hold\">Toggle Hold</button>
        <button data-action=\"quit\">Quit Scene</button>
      </div>

      <p class=\"hint\">Keyboard shortcuts: Right/Space = next, Left = previous.</p>
      <p class=\"hint\" id=\"flash\"></p>
    </section>

    <section class=\"panel\">
      <h2>Presenter Notes</h2>
      <form id=\"note-form\" class=\"note-form\">
        <textarea id=\"note-input\" placeholder=\"Write a speaking note...\"></textarea>
        <button type=\"submit\">Add Note</button>
      </form>
      <p class=\"hint\" id=\"note-count\">0 notes</p>
      <ul id=\"notes-list\" class=\"notes-list\"></ul>
    </section>

    <section class=\"panel span-2\">
      <h2>LLM Controller</h2>
      <div class=\"status\">
        <div class=\"status-item\">
          <div class=\"status-label\">LLM Availability</div>
          <div id=\"llm-availability\" class=\"status-value\">Checking...</div>
        </div>
        <div class=\"status-item\">
          <div class=\"status-label\">LLM Status</div>
          <div id=\"llm-busy\" class=\"status-value\">Idle</div>
        </div>
        <div class=\"status-item\">
          <div class=\"status-label\">Completed Prompts</div>
          <div id=\"llm-completed\" class=\"status-value\">0</div>
        </div>
        <div class=\"status-item\">
          <div class=\"status-label\">Queued Prompts</div>
          <div id=\"llm-queued\" class=\"status-value\">0</div>
        </div>
      </div>

      <form id=\"llm-form\" class=\"note-form\">
        <label for=\"llm-response-mode\" class=\"field-label\">Response Mode</label>
        <select id=\"llm-response-mode\">
          <option value=\"actions\">actions</option>
          <option value=\"code\">code</option>
        </select>

        <label for=\"llm-prompt\" class=\"field-label\">Prompt</label>
        <textarea id=\"llm-prompt\" placeholder=\"Describe what should happen in the scene...\"></textarea>

        <label for=\"llm-system-prompt\" class=\"field-label\">Additional System Prompt (optional)</label>
        <textarea id=\"llm-system-prompt\" placeholder=\"Extra rules for this request\"></textarea>

        <div class=\"row-controls\">
          <button type=\"submit\">Send Prompt To LLM</button>
          <button type=\"button\" id=\"llm-clear-history\">Clear LLM History</button>
          <button type=\"button\" id=\"llm-refresh\">Refresh LLM Status</button>
        </div>
      </form>

      <p class=\"hint\" id=\"llm-feedback\"></p>
      <p class=\"hint\" id=\"llm-last-error\"></p>
    </section>
  </div>

  <script>
    const flashNode = document.getElementById("flash");
    const notesListNode = document.getElementById("notes-list");
    const noteCountNode = document.getElementById("note-count");
    const llmFeedbackNode = document.getElementById("llm-feedback");
    const llmLastErrorNode = document.getElementById("llm-last-error");

    async function postJson(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {}),
      });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Request failed");
      }
      return response.json();
    }

    async function sendAction(action, payload) {
      await postJson("/action", { action, payload: payload || {} });
      flash("Action sent: " + action);
    }

    function flash(message) {
      flashNode.textContent = message;
      setTimeout(() => {
        if (flashNode.textContent === message) {
          flashNode.textContent = "";
        }
      }, 1400);
    }

    function formatTime(seconds) {
      const value = Number(seconds || 0);
      return value.toFixed(2) + "s";
    }

    function formatClock(unixSeconds) {
      const date = new Date(Number(unixSeconds || 0) * 1000);
      return date.toLocaleTimeString();
    }

    function renderNotes(notes) {
      notesListNode.innerHTML = "";
      const reversed = (notes || []).slice().reverse();
      for (const note of reversed) {
        const item = document.createElement("li");
        item.className = "note-item";

        const meta = document.createElement("div");
        meta.className = "note-meta";
        meta.textContent =
          "#" + note.id +
          "  |  play " + note.play_index +
          "  |  t=" + formatTime(note.scene_time) +
          "  |  " + formatClock(note.created_at);

        const body = document.createElement("div");
        body.textContent = note.text;

        item.appendChild(meta);
        item.appendChild(body);
        notesListNode.appendChild(item);
      }
      noteCountNode.textContent = String((notes || []).length) + " notes";
    }

    function renderLlmState(llmState) {
      const state = llmState || {};
      const availabilityNode = document.getElementById("llm-availability");
      const busyNode = document.getElementById("llm-busy");
      const completedNode = document.getElementById("llm-completed");
      const queuedNode = document.getElementById("llm-queued");

      let availability = "Unavailable";
      if (state.available) {
        availability = "Available";
        if (state.source) {
          availability += " (" + state.source + ")";
        }
      } else if (state.reason) {
        availability = "Unavailable: " + state.reason;
      }
      availabilityNode.textContent = availability;

      busyNode.textContent = state.busy ? "Running" : "Idle";
      completedNode.textContent = String(state.completed_count || 0);
      queuedNode.textContent = String(state.queue_size || 0);

      llmLastErrorNode.textContent = state.last_error ? ("Last error: " + state.last_error) : "";
      if (state.last_message) {
        llmFeedbackNode.textContent = state.last_message;
      }

      const modeSelect = document.getElementById("llm-response-mode");
      if (state.default_response_mode && modeSelect && !modeSelect.dataset.userSet) {
        modeSelect.value = state.default_response_mode;
      }
    }

    async function refreshState() {
      try {
        const response = await fetch("/state");
        if (!response.ok) {
          return;
        }
        const state = await response.json();
        document.getElementById("scene-name").textContent = state.scene_name || "-";
        document.getElementById("hold-state").textContent = state.hold_on_wait ? "Holding" : "Running";
        document.getElementById("play-count").textContent = String(state.num_plays || 0);
        document.getElementById("scene-time").textContent = formatTime(state.scene_time);
        renderNotes(state.notes || []);
        renderLlmState(state.llm || {});
      } catch (error) {
        flash("Unable to connect to presenter view server");
      }
    }

    document.querySelectorAll("button[data-action]").forEach((button) => {
      button.addEventListener("click", async () => {
        const action = button.getAttribute("data-action");
        try {
          await sendAction(action, {});
        } catch (error) {
          flash(String(error));
        }
      });
    });

    document.getElementById("note-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const input = document.getElementById("note-input");
      const text = (input.value || "").trim();
      if (!text) {
        return;
      }
      try {
        await postJson("/note", { text });
        input.value = "";
        flash("Note queued");
      } catch (error) {
        flash(String(error));
      }
    });

    document.getElementById("llm-response-mode").addEventListener("change", () => {
      document.getElementById("llm-response-mode").dataset.userSet = "1";
    });

    document.getElementById("llm-form").addEventListener("submit", async (event) => {
      event.preventDefault();

      const promptInput = document.getElementById("llm-prompt");
      const systemPromptInput = document.getElementById("llm-system-prompt");
      const responseMode = document.getElementById("llm-response-mode").value || "actions";
      const prompt = (promptInput.value || "").trim();

      if (!prompt) {
        llmFeedbackNode.textContent = "Write a prompt before sending.";
        return;
      }

      try {
        await postJson("/llm", {
          prompt,
          response_mode: responseMode,
          additional_system_prompt: (systemPromptInput.value || "").trim() || null,
        });
        llmFeedbackNode.textContent = "Prompt queued";
        promptInput.value = "";
      } catch (error) {
        llmFeedbackNode.textContent = String(error);
      }
    });

    document.getElementById("llm-clear-history").addEventListener("click", async () => {
      const responseMode = document.getElementById("llm-response-mode").value || "actions";
      try {
        await postJson("/llm/history/clear", { response_mode: responseMode });
        llmFeedbackNode.textContent = "LLM history cleared for mode: " + responseMode;
      } catch (error) {
        llmFeedbackNode.textContent = String(error);
      }
    });

    document.getElementById("llm-refresh").addEventListener("click", async () => {
      await refreshState();
      llmFeedbackNode.textContent = "LLM status refreshed";
    });

    document.addEventListener("keydown", async (event) => {
      if (event.target && (event.target.tagName === "TEXTAREA" || event.target.tagName === "INPUT")) {
        return;
      }
      if (event.key === "ArrowRight" || event.key === " ") {
        event.preventDefault();
        await sendAction("next", {});
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        await sendAction("previous", {});
      }
    });

    refreshState();
    setInterval(refreshState, 350);
  </script>
</body>
</html>
"""


class _ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class PresenterViewController:
    """Serve a browser dashboard that can control a ManimGL scene."""

    allowed_actions = frozenset(
        {
            "next",
            "previous",
            "pause",
            "resume",
            "toggle_hold",
            "quit",
            "add_note",
        }
    )

    def __init__(
        self,
        scene: "Scene",
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        open_browser: bool = False,
        title: str = "ManimGL Presenter View",
        max_notes: int = 200,
        notes_file: str | Path | None = None,
        max_queued_actions: int = 256,
        llm_enabled: bool = True,
        llm_auto_init: bool = True,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key_env: str = "OPENAI_API_KEY",
        llm_model_env: str = "PRESENTER_VIEW_LLM_MODEL",
        llm_base_url_env: str = "PRESENTER_VIEW_LLM_BASE_URL",
        llm_default_response_mode: str = "actions",
        max_queued_llm_prompts: int = 8,
    ):
        self.scene = scene
        self.host = str(host)
        self.port = int(port)
        self.open_browser = bool(open_browser)
        self.title = str(title)
        self.max_notes = max(1, int(max_notes))
        self.max_queued_actions = max(8, int(max_queued_actions))
        self.llm_enabled = bool(llm_enabled)
        self.llm_auto_init = bool(llm_auto_init)
        self.llm_model = (
          str(llm_model).strip() if llm_model is not None else None
        ) or None
        self.llm_base_url = (
          str(llm_base_url).strip()
          if llm_base_url is not None
          else None
        ) or None
        self.llm_api_key_env = str(llm_api_key_env or "OPENAI_API_KEY")
        self.llm_model_env = str(
          llm_model_env or "PRESENTER_VIEW_LLM_MODEL"
        )
        self.llm_base_url_env = str(
          llm_base_url_env or "PRESENTER_VIEW_LLM_BASE_URL"
        )
        self.llm_default_response_mode = str(
          llm_default_response_mode or "actions"
        ).strip().lower()
        if self.llm_default_response_mode not in {"actions", "code"}:
          self.llm_default_response_mode = "actions"
        llm_reasoning_effort = os.getenv("PRESENTER_VIEW_LLM_REASONING_EFFORT", "").strip() or None
        self.llm_reasoning_effort = (
          str(llm_reasoning_effort).strip()
          if llm_reasoning_effort is not None
          else None
        ) or None
        self.max_queued_llm_prompts = max(
          1, int(max_queued_llm_prompts)
        )

        self.notes_file: Path | None
        if notes_file:
            self.notes_file = Path(notes_file).expanduser()
        else:
            self.notes_file = None

        self._actions: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=self.max_queued_actions
        )
        self._notes: list[dict[str, Any]] = []
        self._notes_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state: dict[str, Any] = {}
        self._next_note_id = 1

        self._server: _ReusableHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._closed = False
        self.url = ""

        self._llm_queue: queue.Queue[dict[str, Any]] = queue.Queue(
          maxsize=self.max_queued_llm_prompts
        )
        self._llm_lock = threading.Lock()
        self._llm_worker_stop = threading.Event()
        self._llm_worker_thread: threading.Thread | None = None
        self._llm_controller: Any | None = None
        self._llm_controller_source: str = ""
        self._llm_auto_init_attempted = False
        self._llm_auto_init_error = ""
        self._llm_last_error = ""
        self._llm_last_message = ""
        self._llm_busy = False
        self._llm_completed_count = 0
        self._llm_submitted_count = 0

        if self.llm_enabled:
          self._llm_worker_thread = threading.Thread(
            target=self._llm_worker,
            name="PresenterViewLLMWorker",
            daemon=True,
          )
          self._llm_worker_thread.start()

        self._scene_updater = self._build_scene_updater()
        self._refresh_state_snapshot()

    def _build_scene_updater(self) -> Callable[[float], None]:
        def _update(_: float) -> None:
            self._process_actions()
            self._refresh_state_snapshot()

        return _update
    
    def _get_llm_state(self) -> dict[str, Any]:
        with self._llm_lock:
            return {
                "available": self._llm_controller is not None,
                "source": self._llm_controller_source,
                "reason": self._llm_auto_init_error if not self._llm_controller else "",
                "busy": self._llm_busy,
                "completed_count": self._llm_completed_count,
                "queue_size": self._llm_queue.qsize(),
                "last_error": self._llm_last_error,
                "last_message": self._llm_last_message,
                "default_response_mode": self.llm_default_response_mode,
            }

    def get_state(self) -> dict[str, Any]:
        with self._state_lock:
            state = dict(self._state)
        state["notes"] = self.get_notes()
        if self.llm_enabled:
            state["llm"] = self._get_llm_state()
        return state

    def enqueue_llm_prompt(
        self,
        prompt: str,
        response_mode: str | None = None,
        additional_system_prompt: str | None = None,
    ) -> bool:
        if not self.llm_enabled:
            return False
        
        mode = str(response_mode).strip().lower() if response_mode else self.llm_default_response_mode
        if mode not in {"actions", "code"}:
            mode = "actions"

        item = {
            "prompt": str(prompt).strip(),
            "response_mode": mode,
            "additional_system_prompt": additional_system_prompt,
        }
        
        try:
            self._llm_queue.put_nowait(item)
            with self._llm_lock:
                self._llm_submitted_count += 1
                self._llm_last_message = f"Prompt en cola ({mode})"
                self._llm_last_error = ""
            return True
        except queue.Full:
            with self._llm_lock:
                self._llm_last_error = "La cola de prompts del LLM está llena"
            return False

    def clear_llm_history(self, response_mode: str | None = None) -> bool:
        if not self._llm_controller:
            return False
        mode = str(response_mode).strip().lower() if response_mode else None
        self._llm_controller.clear_chat_history(mode)
        return True
    
    def _llm_worker(self) -> None:
        # Importación tardía para que el presenter_view no falle si el extra llm no está instalado
        try:
            from openai import OpenAI
            from manimlib.extras.llm.scene_agent import LLMSceneController
        except ImportError as e:
            with self._llm_lock:
                self._llm_auto_init_error = f"Dependencias faltantes: {e}"
            return

        # Fase de auto-inicialización del cliente
        if self.llm_auto_init and self._llm_controller is None and not self._llm_auto_init_attempted:
            self._llm_auto_init_attempted = True
            
            api_key = os.environ.get(self.llm_api_key_env, "").strip()
            model = self.llm_model or os.environ.get(self.llm_model_env, "").strip() or "gpt-4o"
            base_url = self.llm_base_url or os.environ.get(self.llm_base_url_env, "").strip() or None

            if not api_key:
                with self._llm_lock:
                    self._llm_auto_init_error = f"No se encontró API Key en el entorno: {self.llm_api_key_env}"
            else:
                try:
                    client_kwargs = {"api_key": api_key}
                    if base_url:
                        client_kwargs["base_url"] = base_url
                        
                    client = OpenAI(**client_kwargs)
                    controller = LLMSceneController(self.scene, client=client, model=model)
                    
                    with self._llm_lock:
                        self._llm_controller = controller
                        self._llm_controller_source = f"{model} @ {base_url or 'OpenAI'}"
                except Exception as e:
                    with self._llm_lock:
                        self._llm_auto_init_error = f"Error al inicializar LLM: {e}"

        # Bucle principal del worker
        while not self._llm_worker_stop.is_set():
            try:
                # Comprobación temporal para poder escuchar el evento de parada (timeout)
                item = self._llm_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if not self._llm_controller:
                with self._llm_lock:
                    self._llm_last_error = "LLM Controller no está inicializado"
                self._llm_queue.task_done()
                continue

            with self._llm_lock:
                self._llm_busy = True
                self._llm_last_message = f"Procesando prompt ({item['response_mode']})..."
                self._llm_last_error = ""

            try:
                # La invocación al LLM
                # Nota: LLMSceneController pondrá de forma segura las ejecuciones en main_loop_callbacks
                success = self._llm_controller.run_prompt(
                    prompt=item["prompt"],
                    additional_system_prompt=item["additional_system_prompt"],
                    response_mode=item["response_mode"],
                )
                
                with self._llm_lock:
                    if success:
                        self._llm_completed_count += 1
                        self._llm_last_message = "Ejecución exitosa"
                    else:
                        self._llm_last_error = "El LLM falló al generar código/acciones válidas"
                        self._llm_last_message = "Ejecución fallida"
                        
            except Exception as e:
                with self._llm_lock:
                    self._llm_last_error = f"Error interno: {str(e)}"
                    self._llm_last_message = "La ejecución falló"
                    
            finally:
                with self._llm_lock:
                    self._llm_busy = False
                self._llm_queue.task_done()

    def start(self) -> None:
        if self._server is not None:
            return

        handler = self._build_http_handler()
        self._server = _ReusableHTTPServer((self.host, self.port), handler)
        self._server.daemon_threads = True

        self.port = int(self._server.server_address[1])
        browser_host = self.host if self.host not in {"", "0.0.0.0"} else "127.0.0.1"
        self.url = f"http://{browser_host}:{self.port}"

        self.scene.add_updater(self._scene_updater)

        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="PresenterViewServer",
            daemon=True,
        )
        self._server_thread.start()

        log.info(f"Presenter view available at {self.url}")
        if self.open_browser:
            try:
                webbrowser.open_new_tab(self.url)
            except Exception:
                log.exception("Failed to open presenter view browser tab")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        # Detener el hilo del LLM
        self._llm_worker_stop.set()
        if self._llm_worker_thread is not None and self._llm_worker_thread.is_alive():
            self._llm_worker_thread.join(timeout=1.5)
        self._llm_worker_thread = None

        if self._scene_updater in getattr(self.scene, "updaters", []):
            try:
                self.scene.updaters.remove(self._scene_updater)
            except ValueError:
                pass

        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass
            try:
                self._server.server_close()
            except Exception:
                pass
            self._server = None

        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=1.5)
        self._server_thread = None

    def enqueue_action(self, action: str, payload: dict[str, Any] | None = None) -> bool:
        normalized_action = (action or "").strip().lower()
        if normalized_action not in self.allowed_actions:
            return False

        item = {
            "action": normalized_action,
            "payload": payload or {},
        }
        self._queue_put_latest(item)
        return True

    def _queue_put_latest(self, item: dict[str, Any]) -> None:
        try:
            self._actions.put_nowait(item)
            return
        except queue.Full:
            pass

        try:
            self._actions.get_nowait()
        except queue.Empty:
            pass

        try:
            self._actions.put_nowait(item)
        except queue.Full:
            pass

    def _process_actions(self) -> None:
        while True:
            try:
                item = self._actions.get_nowait()
            except queue.Empty:
                return

            action = item.get("action")
            payload = item.get("payload") or {}

            if action == "next":
                self.scene.hold_on_wait = False
            elif action == "previous":
                try:
                    self.scene.undo()
                except Exception:
                    log.exception("Presenter view failed to undo scene state")
                self.scene.hold_on_wait = True
            elif action == "pause":
                self.scene.hold_on_wait = True
            elif action == "resume":
                self.scene.hold_on_wait = False
            elif action == "toggle_hold":
                self.scene.hold_on_wait = not bool(self.scene.hold_on_wait)
            elif action == "quit":
                self.scene.hold_on_wait = False
                self.scene.quit_interaction = True
            elif action == "add_note":
                text = str(payload.get("text", "")).strip()
                if text:
                    self._add_note(text=text, source="browser")

    def _add_note(self, *, text: str, source: str) -> None:
        note = {
            "id": self._next_note_id,
            "text": text,
            "source": source,
            "created_at": time.time(),
            "scene_time": float(getattr(self.scene, "time", 0.0)),
            "play_index": int(getattr(self.scene, "num_plays", 0)),
        }

        with self._notes_lock:
            self._notes.append(note)
            if len(self._notes) > self.max_notes:
                overflow = len(self._notes) - self.max_notes
                if overflow > 0:
                    del self._notes[:overflow]
            self._next_note_id += 1

        if self.notes_file is not None:
            try:
                self.notes_file.parent.mkdir(parents=True, exist_ok=True)
                with self.notes_file.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(note, ensure_ascii=False) + "\n")
            except Exception:
                log.exception("Failed to persist presenter note")

    def get_notes(self) -> list[dict[str, Any]]:
        with self._notes_lock:
            return list(self._notes)

    def _refresh_state_snapshot(self) -> None:
        with self._state_lock:
            self._state = {
                "scene_name": str(self.scene),
                "presenter_mode": bool(getattr(self.scene, "presenter_mode", False)),
                "hold_on_wait": bool(getattr(self.scene, "hold_on_wait", False)),
                "num_plays": int(getattr(self.scene, "num_plays", 0)),
                "scene_time": float(getattr(self.scene, "time", 0.0)),
                "is_window_closing": bool(getattr(self.scene, "is_window_closing", lambda: False)()),
                "url": self.url,
            }

    def get_state(self) -> dict[str, Any]:
        with self._state_lock:
            state = dict(self._state)
        state["notes"] = self.get_notes()
        return state

    def _build_http_handler(self) -> type[BaseHTTPRequestHandler]:
        controller = self

        class PresenterViewHandler(BaseHTTPRequestHandler):
            server_version = "ManimPresenterView/1.0"

            def _read_json_body(self) -> dict[str, Any]:
                try:
                    raw_length = self.headers.get("Content-Length", "0")
                    length = int(raw_length)
                except Exception:
                    length = 0

                if length <= 0:
                    return {}

                raw_body = self.rfile.read(length)
                if not raw_body:
                    return {}

                try:
                    decoded = json.loads(raw_body.decode("utf-8"))
                except json.JSONDecodeError:
                    return {}

                if isinstance(decoded, dict):
                    return decoded
                return {}

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _send_html(self, html: str) -> None:
                raw = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path in {"/", "/index.html"}:
                    html = _INDEX_HTML.replace("__TITLE__", _html_escape(controller.title))
                    self._send_html(html)
                    return

                if parsed.path == "/state":
                    self._send_json(HTTPStatus.OK, controller.get_state())
                    return

                if parsed.path == "/notes":
                    self._send_json(HTTPStatus.OK, {"notes": controller.get_notes()})
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                payload = self._read_json_body()

                if parsed.path == "/action":
                    action = str(payload.get("action", "")).strip().lower()
                    action_payload = payload.get("payload")
                    if not isinstance(action_payload, dict):
                        action_payload = {}

                    ok = controller.enqueue_action(action, action_payload)
                    if not ok:
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {
                                "ok": False,
                                "error": "invalid_action",
                                "allowed": sorted(controller.allowed_actions),
                            },
                        )
                        return

                    self._send_json(HTTPStatus.ACCEPTED, {"ok": True})
                    return

                if parsed.path == "/note":
                    text = str(payload.get("text", "")).strip()
                    if not text:
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"ok": False, "error": "empty_note"},
                        )
                        return

                    controller.enqueue_action("add_note", {"text": text})
                    self._send_json(HTTPStatus.ACCEPTED, {"ok": True})
                    return
                
                if parsed.path == "/llm":
                    prompt = str(payload.get("prompt", "")).strip()
                    if not prompt:
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"ok": False, "error": "empty_prompt"}
                        )
                        return
                    
                    response_mode = payload.get("response_mode")
                    additional_system_prompt = payload.get("additional_system_prompt")
                    
                    ok = controller.enqueue_llm_prompt(prompt, response_mode, additional_system_prompt)
                    if ok:
                        self._send_json(HTTPStatus.ACCEPTED, {"ok": True})
                    else:
                        self._send_json(
                            HTTPStatus.SERVICE_UNAVAILABLE, 
                            {"ok": False, "error": "llm_unavailable_or_queue_full"}
                        )
                    return

                if parsed.path == "/llm/history/clear":
                    response_mode = payload.get("response_mode")
                    controller.clear_llm_history(response_mode)
                    self._send_json(HTTPStatus.OK, {"ok": True})
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

            def log_message(self, format: str, *args: Any) -> None:
                return

        return PresenterViewHandler


def _html_escape(value: str) -> str:
    escaped = value.replace("&", "&amp;")
    escaped = escaped.replace("<", "&lt;")
    escaped = escaped.replace(">", "&gt;")
    escaped = escaped.replace('"', "&quot;")
    return escaped


def _install_teardown_hook(scene: "Scene") -> None:
    if getattr(scene, "_presenter_view_teardown_hook_installed", False):
        return

    original_tear_down = scene.tear_down

    def _tear_down_with_presenter_view_cleanup() -> None:
        try:
            original_tear_down()
        finally:
            controllers = list(getattr(scene, "_presenter_view_controllers", []))
            for controller in controllers:
                try:
                    controller.close()
                except Exception:
                    log.exception("Presenter view cleanup failed")

    scene.tear_down = _tear_down_with_presenter_view_cleanup
    scene._presenter_view_teardown_hook_installed = True


def bind_scene_to_presenter_view(
    scene: "Scene",
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    title: str = "ManimGL Presenter View",
    max_notes: int = 200,
    notes_file: str | Path | None = None,
) -> PresenterViewController:
    """Attach a browser presenter dashboard to a scene."""
    controller = PresenterViewController(
        scene,
        host=host,
        port=port,
        open_browser=open_browser,
        title=title,
        max_notes=max_notes,
        notes_file=notes_file,
    )

    controllers = getattr(scene, "_presenter_view_controllers", None)
    if controllers is None:
        controllers = []
        scene._presenter_view_controllers = controllers
    controllers.append(controller)

    _install_teardown_hook(scene)
    try:
        controller.start()
    except Exception:
        controllers.remove(controller)
        raise

    return controller


def unbind_scene_from_presenter_view(
    scene: "Scene",
    controller: PresenterViewController,
) -> None:
    """Detach and close a presenter view controller from a scene."""
    controllers = getattr(scene, "_presenter_view_controllers", [])
    if controller in controllers:
        controllers.remove(controller)
    controller.close()
