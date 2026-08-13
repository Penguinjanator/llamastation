"""
llamastation_telegram.py — Puente Telegram <-> LlamaStation

Bot de Telegram que habla directo con el llama-server que tienes cargado en
LlamaStation, vía su endpoint local /v1/chat/completions (OpenAI-compatible).
Long-polling puro a la Bot API (sin librerías extra) + streaming real:
va editando el mismo mensaje de Telegram a medida que llegan tokens.

Incluye comandos típicos de agente: /new /stop /status /model /restart /system /help

No depende de companion_bot ni de nada externo, solo `requests`.
"""

import threading, time, json, base64, requests

TG_API = "https://api.telegram.org/bot{token}/{method}"
TG_FILE_API = "https://api.telegram.org/file/bot{token}/{file_path}"

MAX_TG_CHARS = 3900  # margen bajo el límite real de Telegram (4096)

HELP_TEXT = (
    "Comandos disponibles:\n"
    "/new — borra el historial y empieza conversación nueva\n"
    "/stop — corta la generación en curso\n"
    "/status — estado del servidor y modelo cargado\n"
    "/model — nombre del modelo actual\n"
    "/system <texto> — fija un system prompt solo para este chat\n"
    "/system clear — vuelve al system prompt del perfil de LlamaStation\n"
    "/web on|off — activa/desactiva la búsqueda web para este chat\n"
    "/restart — reinicia el servidor llama-server\n"
    "/help — esta ayuda"
)

WEB_TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Busca informacion actualizada en internet. Usala cuando el usuario pregunte sobre "
                        "noticias recientes, eventos actuales, precios, o cualquier informacion que pueda "
                        "haber cambiado desde tu entrenamiento.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "La consulta de busqueda"}
            },
            "required": ["query"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Entra en una URL/link concreto y lee su contenido completo. Usala cuando el usuario "
                        "pegue un link o pida leer/resumir/analizar el contenido de una pagina web especifica.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "La URL completa a leer, ej. https://ejemplo.com/articulo"}
            },
            "required": ["url"]
        }
    }
}]


class TelegramBridge(threading.Thread):
    """
    Hilo daemon que hace polling a Telegram y reenvía cada mensaje al
    endpoint local de LlamaStation.

    Parámetros (todos callbacks para no acoplarse a la GUI):
        token              : str, token del bot
        allowed_chat_ids   : iterable de chat_id (str/int) permitidos. Vacío = permite a todos.
        get_server_info()  -> (host: str, port: str, running: bool)
        get_system_prompt()-> str
        get_gen_params()   -> dict opcional con temperature/top_k/top_p/min_p/repeat_penalty/max_tokens
        get_model_name()   -> str, opcional, nombre/ruta del modelo cargado
        restart_server_fn()-> callable sin argumentos, opcional, reinicia el llama-server
        log_fn(str)        -> callback para volcar logs a la GUI
        history_limit      : nº de mensajes (user+assistant) que se guardan por chat
    """

    def __init__(self, token, allowed_chat_ids, get_server_info, get_system_prompt,
                 get_gen_params=None, get_model_name=None, restart_server_fn=None,
                 get_thinking_params=None, web_enabled_fn=None, run_web_tool_fn=None,
                 log_fn=None, history_limit=20):
        super().__init__(daemon=True)
        self.token = (token or "").strip()
        self.allowed = {str(c).strip() for c in allowed_chat_ids if str(c).strip()}
        self.get_server_info = get_server_info
        self.get_system_prompt = get_system_prompt
        self.get_gen_params = get_gen_params or (lambda: {})
        self.get_model_name = get_model_name or (lambda: "desconocido")
        self.restart_server_fn = restart_server_fn
        self.get_thinking_params = get_thinking_params or (lambda: {"enable_thinking": False})
        self.web_enabled_fn = web_enabled_fn or (lambda: False)
        self.run_web_tool_fn = run_web_tool_fn  # (name, args_dict) -> str
        self.log_fn = log_fn or (lambda s: None)
        self.history_limit = history_limit

        self._stop_evt = threading.Event()
        self._offset = 0
        self._sessions = {}          # chat_id -> [ {role, content}, ... ]
        self._system_override = {}   # chat_id -> system prompt custom para ese chat
        self._web_override = {}      # chat_id -> True/False, anula el switch global de la pestaña
        self._active = {}            # chat_id -> bool, True mientras se está generando
        self._edit_min_interval = 1.2  # seg. mínimos entre ediciones del mismo mensaje (rate limit TG)

    def stop(self):
        self._stop_evt.set()

    def _log(self, s):
        try:
            self.log_fn(str(s))
        except Exception:
            pass

    # ── Telegram HTTP helpers ───────────────────────────────────────────
    def _call(self, method, **params):
        url = TG_API.format(token=self.token, method=method)
        try:
            r = requests.post(url, json=params, timeout=35)
            data = r.json()
            if not data.get("ok", True):
                self._log(f"{method} -> {data.get('description')}")
            return data
        except Exception as e:
            self._log(f"error llamando a {method}: {e}")
            return {}

    def _get_updates(self):
        url = TG_API.format(token=self.token, method="getUpdates")
        try:
            r = requests.get(url, params={"offset": self._offset, "timeout": 25}, timeout=40)
            return r.json().get("result", [])
        except requests.exceptions.ReadTimeout:
            return []
        except Exception as e:
            self._log(f"error de polling: {e}")
            time.sleep(3)
            return []

    # ── Loop principal ───────────────────────────────────────────────────
    def run(self):
        if not self.token:
            self._log("token vacío, el bridge no arranca")
            return

        self._log("bridge iniciado, esperando mensajes...")
        # Descarta updates pendientes de antes de arrancar (no contestar a mensajes viejos)
        try:
            initial = requests.get(TG_API.format(token=self.token, method="getUpdates"),
                                    params={"timeout": 0}, timeout=10).json().get("result", [])
            if initial:
                self._offset = initial[-1]["update_id"] + 1
        except Exception:
            pass

        while not self._stop_evt.is_set():
            for upd in self._get_updates():
                self._offset = upd["update_id"] + 1
                msg = upd.get("message") or upd.get("edited_message")
                if not msg or not ("text" in msg or "photo" in msg or "caption" in msg):
                    continue
                try:
                    self._handle_message(msg)
                except Exception as e:
                    self._log(f"error procesando mensaje: {e}")

        self._log("bridge detenido")

    # ── Comandos ─────────────────────────────────────────────────────────
    def _handle_command(self, chat_id, text):
        """Devuelve True si el texto era un comando y ya quedó resuelto."""
        cmd, _, arg = text.partition(" ")
        cmd = cmd.lower()
        # En chats de grupo Telegram añade "@nombredelbot" al comando (ej. "/web@MiBot"),
        # lo que rompía la comparación exacta más abajo.
        cmd = cmd.split("@", 1)[0]
        arg = arg.strip()

        if cmd == "/start":
            self._call("sendMessage", chat_id=chat_id,
                        text="Conectado a LlamaStation. Escribe normal para hablar con el "
                             "modelo cargado ahora mismo.\n\n" + HELP_TEXT)
            return True

        if cmd in ("/new", "/reset", "/clear"):
            self._sessions.pop(chat_id, None)
            self._call("sendMessage", chat_id=chat_id, text="🆕 Conversación nueva, historial borrado.")
            return True

        if cmd == "/stop":
            if self._active.get(chat_id):
                self._active[chat_id] = False
                self._call("sendMessage", chat_id=chat_id, text="⏹ Generación detenida.")
            else:
                self._call("sendMessage", chat_id=chat_id, text="No hay ninguna generación en curso.")
            return True

        if cmd == "/web":
            if arg.lower() in ("on", "1", "true", "activar"):
                self._web_override[chat_id] = True
                self._call("sendMessage", chat_id=chat_id, text="🌐 Búsqueda web activada para este chat.")
            elif arg.lower() in ("off", "0", "false", "desactivar"):
                self._web_override[chat_id] = False
                self._call("sendMessage", chat_id=chat_id, text="🌐 Búsqueda web desactivada para este chat.")
            else:
                estado = self._web_override.get(chat_id)
                if estado is None:
                    estado = self.web_enabled_fn()
                self._call("sendMessage", chat_id=chat_id,
                            text=f"Búsqueda web ahora mismo: {'ON' if estado else 'OFF'}. Usa /web on o /web off.")
            return True

        if cmd == "/status":
            host, port, running = self.get_server_info()
            estado = "🟢 corriendo" if running else "🔴 parado"
            model = self.get_model_name() or "—"
            msgs = len(self._sessions.get(chat_id, []))
            sys_ov = "sí" if chat_id in self._system_override else "no"
            web_state = self._web_override.get(chat_id)
            if web_state is None:
                web_state = self.web_enabled_fn()
            self._call("sendMessage", chat_id=chat_id,
                        text=(f"Servidor: {estado} ({host}:{port})\n"
                              f"Modelo: {model}\n"
                              f"Mensajes en historial de este chat: {msgs}\n"
                              f"System prompt personalizado: {sys_ov}\n"
                              f"Búsqueda web: {'ON' if web_state else 'OFF'}"))
            return True

        if cmd == "/model":
            self._call("sendMessage", chat_id=chat_id, text=f"Modelo actual: {self.get_model_name() or '—'}")
            return True

        if cmd == "/system":
            if not arg or arg.lower() == "clear":
                self._system_override.pop(chat_id, None)
                self._call("sendMessage", chat_id=chat_id,
                            text="System prompt de este chat restablecido al del perfil de LlamaStation.")
            else:
                self._system_override[chat_id] = arg
                self._call("sendMessage", chat_id=chat_id,
                            text=f"System prompt de este chat actualizado:\n{arg}")
            return True

        if cmd == "/restart":
            if not self.restart_server_fn:
                self._call("sendMessage", chat_id=chat_id, text="Reinicio no disponible.")
                return True
            self._call("sendMessage", chat_id=chat_id, text="🔄 Reiniciando servidor...")
            try:
                self.restart_server_fn()
            except Exception as e:
                self._call("sendMessage", chat_id=chat_id, text=f"Error al reiniciar: {e}")
            return True

        if cmd == "/help":
            self._call("sendMessage", chat_id=chat_id, text=HELP_TEXT)
            return True

        return False

    # ── Fotos ────────────────────────────────────────────────────────────
    def _download_photo_b64(self, file_id):
        """Descarga una foto de Telegram por file_id y la devuelve como
        (mime, base64_str), o (None, None) si algo falla."""
        try:
            info = self._call("getFile", file_id=file_id)
            file_path = (info.get("result") or {}).get("file_path")
            if not file_path:
                self._log(f"getFile sin file_path para {file_id}: {info}")
                return None, None
            url = TG_FILE_API.format(token=self.token, file_path=file_path)
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else "jpg"
            mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                        "webp": "image/webp", "gif": "image/gif"}
            mime = mime_map.get(ext, "image/jpeg")
            return mime, base64.b64encode(r.content).decode("ascii")
        except Exception as e:
            self._log(f"error descargando foto de Telegram: {e}")
            return None, None

    # ── Lógica de un mensaje ─────────────────────────────────────────────
    def _handle_message(self, msg):
        chat_id = str(msg["chat"]["id"])
        text = (msg.get("text") or msg.get("caption") or "").strip()
        sender = msg.get("from", {}).get("first_name", chat_id)

        if self.allowed and chat_id not in self.allowed:
            self._log(f"mensaje ignorado, chat_id no autorizado: {chat_id} ({sender})")
            return

        if text.startswith("/"):
            if self._handle_command(chat_id, text):
                return

        # ── Foto adjunta: se descarga y se manda como mensaje de vision ──
        photo_content = None
        if msg.get("photo"):
            # El array "photo" trae varios tamaños; el último es el de mayor resolución.
            largest = msg["photo"][-1]
            self._call("sendChatAction", chat_id=chat_id, action="upload_photo")
            mime, b64 = self._download_photo_b64(largest["file_id"])
            if b64:
                photo_content = [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": text or "Describe esta imagen."},
                ]
            else:
                self._call("sendMessage", chat_id=chat_id,
                            text="⚠️ No pude descargar la imagen desde Telegram.")
                return

        host, port, running = self.get_server_info()
        if not running:
            self._call("sendMessage", chat_id=chat_id,
                        text="⚠️ El servidor de LlamaStation está apagado. Arráncalo desde la app o manda /restart.")
            return

        log_txt = ("[imagen] " + text[:80]) if photo_content else text[:80]
        self._log(f"{sender} ({chat_id}): {log_txt}")
        self._call("sendChatAction", chat_id=chat_id, action="typing")

        history = self._sessions.setdefault(chat_id, [])
        history.append({"role": "user", "content": photo_content if photo_content else text})
        self._sessions[chat_id] = history[-self.history_limit:]

        sys_prompt = self._system_override.get(chat_id)
        if sys_prompt is None:
            sys_prompt = (self.get_system_prompt() or "").strip()

        gen = self.get_gen_params()
        tp = self.get_thinking_params() or {}

        web_state = self._web_override.get(chat_id)
        web_on = web_state if web_state is not None else bool(self.web_enabled_fn())
        use_tools = web_on and self.run_web_tool_fn is not None

        if use_tools:
            # Recordatorio añadido al system prompt (no al historial guardado):
            # sin esto el modelo a veces contesta de memoria en vez de buscar,
            # sobre todo en preguntas de actualidad/fechas recientes. Se mete en
            # el mismo mensaje "system" inicial (no como uno aparte) porque varias
            # plantillas de chat solo procesan un system y al principio.
            _web_hint = ("Tienes acceso a las herramientas web_search y fetch_url. Para cualquier "
                         "pregunta sobre eventos actuales, fechas recientes, resultados o cualquier "
                         "dato que pueda haber cambiado, USA web_search en vez de responder de memoria. "
                         "Si el primer resultado no es concluyente, prueba con otra consulta distinta "
                         "o usa fetch_url sobre una fuente fiable antes de responder.")
            sys_prompt = (sys_prompt + "\n\n" + _web_hint).strip() if sys_prompt else _web_hint

        messages = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + self._sessions[chat_id]

        def _base_payload(msgs):
            payload = {
                "model": "local",
                "messages": msgs,
                "stream": True,
                "temperature": gen.get("temperature", 0.7),
                "top_k": gen.get("top_k", 40),
                "top_p": gen.get("top_p", 0.95),
                "min_p": gen.get("min_p", 0.05),
                "repeat_penalty": gen.get("repeat_penalty", 1.1),
                "max_tokens": gen.get("max_tokens", 1024) or -1,
                "chat_template_kwargs": {
                    "enable_thinking": bool(tp.get("enable_thinking", False)),
                    **({"thinking_budget": tp["thinking_budget"]}
                       if tp.get("enable_thinking") and tp.get("thinking_budget") is not None else {}),
                },
            }
            if tp.get("enable_thinking") and tp.get("reasoning_control"):
                payload["reasoning_control"] = True
            if use_tools:
                payload["tools"] = WEB_TOOLS
                payload["tool_choice"] = "auto"
            return payload

        # Mensaje placeholder que se va editando conforme llegan tokens (streaming real)
        sent = self._call("sendMessage", chat_id=chat_id, text="…")
        tg_msg_id = sent.get("result", {}).get("message_id")

        self._active[chat_id] = True
        current_messages = list(messages)
        full = ""
        stopped_by_user = False
        # Antes en 5: el modelo se quedaba sin rondas para reformular la búsqueda
        # o leer una segunda fuente y acababa respondiendo de memoria (p.ej. "no se
        # ha celebrado" para eventos recientes). Lo igualamos más al chat principal
        # de LlamaStation (que permite 40) para darle margen real a investigar.
        max_tool_rounds = 20

        in_think = False
        in_channel = False

        try:
            for round_i in range(max_tool_rounds + 1):
                raw_buf = ""     # buffer crudo para detectar tags <think>...</think> partidos entre chunks
                think_buf = ""   # buffer crudo para <|channel>...<channel|> (formato Gemma4/harmony)
                in_think = False
                in_channel = False
                last_edit = 0.0
                tool_calls_acc = {}
                finish_reason = None
                round_stopped = False

                resp = requests.post(f"http://{host}:{port}/v1/chat/completions",
                                      json=_base_payload(current_messages), stream=True, timeout=180)
                for line in resp.iter_lines():
                    if self._stop_evt.is_set():
                        resp.close(); round_stopped = True; break
                    if not self._active.get(chat_id, True):
                        resp.close(); stopped_by_user = True; round_stopped = True; break
                    if not line:
                        continue
                    line = line.decode(errors="ignore")
                    if not line.startswith("data: "):
                        continue
                    d = line[6:]
                    if d == "[DONE]":
                        break
                    try:
                        chunk = json.loads(d)
                        choice = chunk["choices"][0]
                        delta_obj = choice.get("delta", {})
                        delta = delta_obj.get("content", "") or ""
                        finish_reason = choice.get("finish_reason") or finish_reason
                        # Acumular tool calls fragmentados entre chunks
                        for tc in delta_obj.get("tool_calls", []):
                            tidx = tc.get("index", 0)
                            if tidx not in tool_calls_acc:
                                tool_calls_acc[tidx] = {"id": "", "name": "", "arguments": ""}
                            fn = tc.get("function", {})
                            if fn.get("name"):      tool_calls_acc[tidx]["name"]      += fn["name"]
                            if fn.get("arguments"): tool_calls_acc[tidx]["arguments"] += fn["arguments"]
                            if tc.get("id"):        tool_calls_acc[tidx]["id"]         = tc["id"]
                        # El razonamiento nativo (reasoning_content) se descarta directamente
                        # si llega a aparecer; con --reasoning-format none el servidor lo mete
                        # todo (Qwen: <think>, Gemma4: <|channel>) dentro de "content".
                    except Exception:
                        continue
                    if not delta:
                        continue

                    visible = ""

                    # Filtrar <|channel>...<channel|> (Gemma4/harmony) — igual que el chat principal.
                    if "<|channel>" in delta or in_channel:
                        think_buf += delta
                        if not in_channel and "<|channel>" in think_buf:
                            before = think_buf[:think_buf.find("<|channel>")]
                            think_buf = think_buf[think_buf.find("<|channel>") + len("<|channel>"):]
                            in_channel = True
                            if before:
                                visible += before
                        if in_channel and "<channel|>" in think_buf:
                            idx_end = think_buf.find("<channel|>")
                            after = think_buf[idx_end + len("<channel|>"):]
                            think_buf = ""
                            in_channel = False
                            if after:
                                visible += after
                        if not visible:
                            continue
                    else:
                        # Red de seguridad: si el backend mete el pensamiento inline como
                        # <think>...</think> dentro de content, lo filtramos aquí igual que
                        # hace el chat principal.
                        raw_buf += delta
                        while True:
                            if not in_think:
                                idx = raw_buf.find("<think>")
                                if idx == -1:
                                    hold = 0
                                    for i in range(1, min(len("<think>"), len(raw_buf)) + 1):
                                        if raw_buf.endswith("<think>"[:i]):
                                            hold = i
                                    cut = len(raw_buf) - hold
                                    visible += raw_buf[:cut]
                                    raw_buf = raw_buf[cut:]
                                    break
                                else:
                                    visible += raw_buf[:idx]
                                    raw_buf = raw_buf[idx + len("<think>"):]
                                    in_think = True
                            else:
                                idx = raw_buf.find("</think>")
                                if idx == -1:
                                    hold = 0
                                    for i in range(1, min(len("</think>"), len(raw_buf)) + 1):
                                        if raw_buf.endswith("</think>"[:i]):
                                            hold = i
                                    raw_buf = raw_buf[len(raw_buf) - hold:]
                                    break
                                else:
                                    raw_buf = raw_buf[idx + len("</think>"):]
                                    in_think = False

                    if not visible:
                        continue
                    full += visible
                    now = time.time()
                    if tg_msg_id and (now - last_edit) >= self._edit_min_interval:
                        last_edit = now
                        self._call("editMessageText", chat_id=chat_id, message_id=tg_msg_id,
                                    text=(full[-MAX_TG_CHARS:] + " ▌"))

                if round_stopped:
                    break

                if use_tools and tool_calls_acc and finish_reason == "tool_calls":
                    tc_list = []
                    for tidx in sorted(tool_calls_acc.keys()):
                        tc = tool_calls_acc[tidx]
                        tc_list.append({
                            "id": tc["id"] or f"call_{tidx}",
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]}
                        })
                    current_messages.append({"role": "assistant", "content": None, "tool_calls": tc_list})

                    for tc in tc_list:
                        name = tc["function"]["name"]
                        try:
                            args = json.loads(tc["function"]["arguments"] or "{}")
                        except Exception:
                            args = {}
                        if name == "web_search":
                            status = f"🌐 Buscando: {args.get('query', '')}"
                        elif name == "fetch_url":
                            status = f"🔗 Leyendo: {args.get('url', '')}"
                        else:
                            status = f"🔧 {name}"
                        if tg_msg_id:
                            self._call("editMessageText", chat_id=chat_id, message_id=tg_msg_id, text=status)
                        self._log(status)
                        try:
                            result = self.run_web_tool_fn(name, args)
                        except Exception as e:
                            result = f"Error ejecutando {name}: {e}"
                        current_messages.append({
                            "role": "tool", "tool_call_id": tc["id"] or "call_0", "content": str(result)
                        })
                    continue  # siguiente ronda con los resultados ya en el contexto

                break  # respuesta final, sin más tool calls
        except Exception as e:
            self._log(f"error generando respuesta: {e}")
            full = full or f"Error al generar respuesta: {e}"
        finally:
            self._active.pop(chat_id, None)

        full = full.strip()
        if not full:
            if stopped_by_user:
                full = "(generación detenida)"
            elif in_think or in_channel:
                # El stream terminó (normalmente por acabarse max_tokens) sin llegar a
                # cerrar el bloque de razonamiento, así que no hay respuesta final que
                # mostrar: todo el presupuesto de tokens se fue en el "pensamiento".
                full = ("⚠️ El modelo se quedó pensando y se acabaron los tokens antes de "
                        "llegar a la respuesta final. Prueba a subir max_tokens en el perfil "
                        "o a bajar el thinking budget.")
            else:
                full = "(sin respuesta)"

        if tg_msg_id:
            chunks = [full[i:i + MAX_TG_CHARS] for i in range(0, len(full), MAX_TG_CHARS)] or [full]
            self._call("editMessageText", chat_id=chat_id, message_id=tg_msg_id, text=chunks[0])
            for extra in chunks[1:]:
                self._call("sendMessage", chat_id=chat_id, text=extra)
        else:
            self._call("sendMessage", chat_id=chat_id, text=full[:4000])

        if not stopped_by_user:
            history.append({"role": "assistant", "content": full})
            self._sessions[chat_id] = history[-self.history_limit:]
