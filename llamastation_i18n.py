"""
LlamaStation — Internacionalización (i18n)
Añade un idioma nuevo: copia el bloque "en" y traduce los valores.
"""

STRINGS = {
    "es": {
        # ── Navegación ────────────────────────────────────────────────
        "nav_chat":        "Chat",
        "nav_server":      "Servidor",
        "nav_logs":        "Logs",
        "nav_info":        "Info modelo",
        "nav_download":    "Descargar",
        "nav_api":         "API Docs",
        "nav_voice":       "Voz",
        "nav_about":       "Acerca de",

        # ── Sidebar derecha ───────────────────────────────────────────
        "loaded_model":    "MODELO CARGADO",
        "no_model":        "Ninguno",
        "my_models":       "📂  Mis modelos",
        "download_models": "🌐  Descargar modelos",
        "server_label":    "SERVIDOR",
        "server_stopped":  "Detenido",
        "server_port":     "puerto: —",
        "start_server":    "▶  Iniciar servidor",
        "stop_server":     "■  Detener",
        "backend_label":   "BACKEND",
        "update_llama":    "⬆  Actualizar llama.cpp",
        "controls":        "Controles",

        # ── Sidebar izquierda ─────────────────────────────────────────
        "app_title":       "⚡ LlamaStation",
        "conversations":   "CONVERSACIONES",
        "new_chat":        "＋  Nuevo chat",
        "no_history":      "Sin conversaciones\nEmpieza a chatear →",

        # ── Chat ──────────────────────────────────────────────────────
        "chat_title":      "Chat",
        "clear_chat":      "Limpiar",
        "thinking_on":         "👁 Razonamiento",
        "thinking_off":        "👁 Razonamiento OFF",
        "enable_thinking_on":  "🧠 Think: ON",
        "enable_thinking_off": "⚡ Think: OFF",
        "web_on":          "🌐 Web ON",
        "web_off":         "🌐 Web OFF",
        "system_label":    "System:",
        "system_ph":       "System prompt...",
        "send":            "Enviar",
        "sending":         "...",
        "you":             "Tú",
        "model":           "Modelo",

        # ── Servidor tab ──────────────────────────────────────────────
        "server_tab_title":   "Configuración del Servidor",
        "server_exe":         "🔧 Ejecutable llama-server",
        "server_network":     "🌐 Red",
        "server_port_lbl":    "Puerto",
        "server_host_lbl":    "Host",
        "server_save":        "💾 Guardar",
        "server_cmd":         "📋 Comando generado",
        "server_cmd_preview": "Actualizar previsualización",
        "server_saved_ok":    "Configuración guardada.",
        "browse":             "Buscar...",
        "server_headless_sec": "🖥️  Modo Headless",
        "server_headless_desc": (
            "Arranca LlamaStation sin ventana gráfica.\n"
            "Usa el perfil guardado del modelo seleccionado."
        ),
        "server_headless_cmd_label": "Comando (copia y pega en tu terminal):",
        "server_headless_no_model":  "← Carga un modelo primero",
        "copy":               "📋 Copiar",

        # ── Logs tab ──────────────────────────────────────────────────
        "logs_title":      "Logs del servidor",
        "logs_clear":      "Limpiar",

        # ── Info tab ──────────────────────────────────────────────────
        "info_title":      "Info del modelo",
        "info_no_server":  "⚠ El servidor no está corriendo.",

        # ── Modal de carga ────────────────────────────────────────────
        "modal_vram":      "Uso estimado de VRAM",
        "modal_vram_hint": "depende del modelo y ctx",
        "modal_cancel":    "Cancelar",
        "modal_load":      "Cargar Modelo  ↵",
        "sec_hardware":    "Hardware",
        "sec_multigpu":    "Multi-GPU  (2x RTX 3060)",
        "sec_cpu":         "CPU / NPU — Sin GPU dedicada",
        "sec_context":     "Contexto y Batch",
        "sec_sampling":    "Sampling",
        "sec_flags":       "Opciones Avanzadas",
        "sec_rope":        "RoPE Scaling  (para extender contexto)",
        "sec_vision":      "Vision (mmproj)",
        "sec_extra":       "Argumentos Extra (texto libre)",

        # Sliders — LoadModelDialog
        "sl_gpu_layers":   "GPU Layers  — capas del modelo en GPU (-1 = todas, reduce si te quedas sin VRAM)",
        "sl_threads":      "Threads  — hilos CPU para la parte que no va a GPU",
        "sl_threads_b":    "Threads Batch  — hilos CPU para procesar el prompt inicial",
        "sl_max_conc":     "Max Concurrent  — peticiones simultáneas, experimental, deja en 1",
        "sl_ctx":          "Context Size  — tokens que recuerda el modelo, más = más VRAM",
        "sl_batch":        "Batch Size  — tokens procesados a la vez al cargar el prompt, más = más rápido",
        "sl_ubatch":       "UBatch Size  — subdivisión del batch, debe ser igual o menor que Batch Size",
        "sl_max_tok":      "Max Tokens  — límite de tokens a generar por respuesta (-1 = ilimitado)",
        "sl_temp":         "Temperature  — creatividad: 0=determinista, 0.7=equilibrado, >1=más aleatorio",
        "sl_topk":         "Top K  — limita el muestreo a los K tokens más probables, 0=desactivado",
        "sl_topp":         "Top P  — muestreo por núcleo: usa tokens hasta acumular esta probabilidad",
        "sl_minp":         "Min P  — descarta tokens con probabilidad menor a este % del token más probable",
        "sl_rep_pen":      "Repeat Penalty  — penaliza repetir palabras, 1.0=sin penalización",
        "sl_rep_n":        "Repeat Last N  — cuántos tokens atrás se miran para el Repeat Penalty",
        "sl_seed":         "Seed  — semilla aleatoria, -1=diferente cada vez, fijar para reproducir resultados",
        "sl_rope_base":    "RoPE Freq Base  — frecuencia base para escalar el contexto, 0=Auto",
        "sl_rope_scale":   "RoPE Freq Scale  — factor de escala de frecuencia, 0=Auto",

        # Flags
        "fl_flash":        "Flash Attention",
        "fl_flash_tip":    "Acelera el cálculo de atención en GPU. Recomendado siempre activado.",
        "fl_mmap":         "mmap()",
        "fl_mmap_tip":     "Memory-map: carga el modelo mapeando el archivo en memoria. Más eficiente.",
        "fl_mlock":        "mlock",
        "fl_mlock_tip":    "Bloquea el modelo en RAM para que el SO no lo mande a swap.",
        "fl_contbatch":    "Continuous Batching",
        "fl_contbatch_tip":"Atiende varias peticiones a la vez sin esperar. Recomendado activado.",
        "fl_kvoff":        "KV Cache Offload",
        "fl_kvoff_tip":    "Mueve el KV cache a VRAM. Permite contextos más largos.",
        "fl_keepmem":      "Keep In Memory",
        "fl_keepmem_tip":  "No descarga el modelo al parar el servidor. El siguiente arranque será más rápido.",
        "fl_embed":        "Embeddings endpoint",
        "fl_embed_tip":    "Activa /v1/embeddings en la API. Solo necesario si vas a generar vectores de texto.",
        "kv_label":        "K/V Cache — puedes elegir tipos distintos para K y V (asimetría recomendada con TurboQuant)",
        "kv_zero":         "0 = Auto (recomendado). Útil para NTK/YaRN.",
        "sys_prompt":      "System Prompt",
        "sys_prompt_ph":   "Eres un asistente útil...",
        "extra_tip":       "Para flags futuros: TurboQuant, etc.",
        "extra_ph":        "--override-kv ...",
        "split_tip":       "Split Mode — cómo se reparte el modelo entre GPUs. 'layer' = reparto por capas (recomendado). 'row' = por filas de tensores.",
        "tensor_tip":      "Tensor Split — proporción de VRAM por GPU (ej: '1,1' = 50/50, '3,1' = 75/25). Vacío = auto.",
        "tensor_ph":       "1,1  (dejar vacío para auto)",
        "cpu_mode_tip":    "Modo: GPU (defecto), Solo CPU+RAM, Hibrido o Vulkan (AMD/Intel/NPU)",
        "cpu_tip2":        "Consejo CPU: sube Threads (sección Hardware) al número de núcleos físicos de tu CPU para máximo rendimiento.",
        "mmproj_tip":      "Archivo mmproj para modelos con vision. Se detecta automaticamente si esta en la misma carpeta que el modelo.",
        "mmproj_none":     "Sin mmproj detectado — este modelo no tiene vision o ponlo manualmente.",
        "mmproj_ph":       "Ruta al archivo mmproj-*.gguf  (opcional)",
        "mmproj_select":   "Selecciona mmproj",
        "mmproj_disable":  "Deshabilitar mmproj (ahorra VRAM, desactiva visión)",

        # Botones modal / browser
        "load_btn":        "Cargar  ↵",
        "cancel":          "Cancelar",
        "delete_model":    "🗑️",
        "delete_model_confirm_title": "Borrar modelo",
        "delete_model_confirm_msg":   "¿Borrar permanentemente este archivo?\n\n{name}\n\n({size})",
        "delete_model_ok":  "Modelo borrado.",
        "delete_model_err": "Error al borrar: {err}",
        "cleanup_backups_sec":   "🧹  Limpiar backups antiguos",
        "cleanup_backups_desc":  "Backups de instalaciones anteriores de llama.cpp/TurboQuant que ya no son necesarios.",
        "cleanup_backups_scan":  "Buscar backups",
        "cleanup_backups_none":  "No se encontraron backups.",
        "cleanup_backups_found": "{n} backup(s) encontrado(s)  •  {size} en total",
        "cleanup_backups_del":   "Borrar todos",
        "cleanup_backups_done":  "✓ Backups eliminados.",
        "change":          "Cambiar",
        "manual_browse":   "📂  Buscar archivo manualmente...",
        "folder_label":    "Carpeta:",
        "no_models_dir":   "No hay modelos en esta carpeta.\n\nDescarga modelos desde el tab 🌐 Descargar\no cambia la carpeta arriba.",
        "no_models_dir2":  "No hay modelos .gguf en esta carpeta.\n\nDescarga desde 🌐 Descargar o cambia la carpeta.",
        "my_models_title": "📂  Mis modelos",
        "models_folder":   "Carpeta de modelos",
        "select_gguf":     "Selecciona modelo GGUF",

        # Mensajes de error / info
        "err_no_model":    "Selecciona un modelo primero.",
        "err_no_server":   "Ve a 'Servidor' y especifica la ruta a llama-server.exe",
        "err_not_found":   "No se encontró",
        "err_no_model_title":   "Sin modelo",
        "err_server_title":     "llama-server no encontrado",
        "err_title":            "Error",
        "warn_stopped":         "Servidor detenido",
        "warn_stopped_msg":     "Inicia el servidor antes de chatear.",
        "warn_backend_title":   "Backend no encontrado",
        "saved_title":          "Guardado",
        "save_file_title":      "Guardar archivo generado",
        "saved_msg":            "Archivo guardado en:\n",
        "err_save":             "No se pudo guardar:\n",
        "err_read":             "No se pudo leer",
        "downloader_missing":   "llamastation_downloader.py no encontrado.\nPon el archivo en la misma carpeta.",
        "attach_image":         "Adjuntar imagen",
        "attach_files":         "Adjuntar archivo(s)",
        "save_generated":       "Guardar",
        "describe_image":       "Describe esta imagen.",
        "review_files":         "Revisa los archivos.",

        # ── API Docs tab ──────────────────────────────────────────────
        "api_title":       "📡  API Docs",
        "api_subtitle":    "llama.cpp OpenAI-compatible API",
        "api_intro":       (
            "  Tu servidor llama.cpp expone una API compatible con OpenAI en:\n"
            "  http://127.0.0.1:{port}/v1\n\n"
            "  Cualquier app que use la API de OpenAI funciona apuntando a esa URL."
        ),
        "api_sec_chat":    "💬  Chat Completions",
        "api_sec_stream":  "⚡  Streaming (respuesta en tiempo real)",
        "api_sec_models":  "📋  Modelos",
        "api_sec_health":  "🩺  Estado del servidor",
        "api_sec_headless":"🖥️  Modo Headless (sin ventana)",
        "api_chat_desc":   "Genera una respuesta de chat",
        "api_stream_desc": "Streaming SSE — tokens en tiempo real",
        "api_models_desc": "Lista los modelos cargados",
        "api_health_desc": "Comprueba si el servidor está listo",

        # ── Tema / idioma ─────────────────────────────────────────────
        "theme_to_light":  "☀  Modo claro",
        "theme_to_dark":   "🌙  Modo oscuro",
        "lang_btn":        "🌐  English",

        # ── Actualización ─────────────────────────────────────────────
        "update_title":    "⬆  Actualizar llama.cpp",
        "update_subtitle": "solo backend oficial · forks no afectados",
        "update_close":    "Cerrar",
        "update_install":  "Instalar actualización",
        "update_uptodate": "Ya estás al día",
        "update_installing":"Instalando...",
        "update_done":     "✓ Instalado",
        "update_retry":    "Reintentar",

        # Servidor — status
        "srv_running":     "Corriendo",
        "srv_running_ext": "Corriendo (externo)",
        "srv_starting":    "Iniciando...",
        "srv_port":        "puerto: {port}",
        "srv_stopped_log": "Servidor detenido. VRAM liberada.",

        # ── Watchdog ─────────────────────────────────────────────────
        "watchdog_sec":           "🛡️  Watchdog del servidor",
        "watchdog_auto_relaunch": "Reinicar servidor automáticamente si se cae",
        "watchdog_crashed_title": "⚠️  Servidor caído",
        "watchdog_crashed_msg":   "El servidor se ha detenido inesperadamente (código {rc}).\n\nÚltimas líneas del log:\n\n{log}\n\nRevisa la pestaña Logs para más detalles.",
        "watchdog_relaunch_log":  "🔄 Watchdog: reiniciando servidor en {delay}s...",
        "watchdog_relaunching":   "Reiniciando...",

        # ── Sonido ───────────────────────────────────────────────────
        "sound_on":        "🔔  Sonido ON",
        "sound_off":       "🔕  Sonido OFF",

        # ── Downloader ───────────────────────────────────────────────
        "dl_title":        "🌐  Descargar Modelos",
        "dl_save_in":      "Guardar en:",
        "dl_search_ph":    "🔍  Buscar modelo... (ej: llama-3, qwen2.5, mistral)",
        "dl_search_btn":   "Buscar",
        "dl_arch":         "Arquitectura:",
        "dl_author":       "Autor:",
        "dl_searching":    "Buscando...",
        "dl_no_results":   "No se encontraron modelos.\nPrueba con otra búsqueda.",
        "dl_start_hint":   "Busca un modelo para comenzar\no explora los más populares →",
        "dl_select_hint":  "← Selecciona un modelo de la lista",
        "dl_files_title":  "ARCHIVOS DISPONIBLES",
        "dl_quant_filter": "Filtrar cuant:",
        "dl_files_ph":     "Selecciona un modelo para ver sus archivos",
        "dl_loading":      "Cargando archivos...",
        "dl_no_files":     "No hay archivos con este filtro",
        "dl_downloads":    "DESCARGAS ACTIVAS",
        "dl_no_downloads": "Ninguna descarga en curso",
        "dl_view_hf":      "Ver en HF ↗",
        "dl_downloads_lbl":"descargas",
        "dl_updated":      "Actualizado:",
        "dl_caps_warning": "⚠ Las capacidades mostradas pueden ser incompletas",

        # ── MTP ──────────────────────────────────────────────────────
        "mtp_activate":    "Activar MTP",
        "mtp_draft_n_max": "spec-draft-n-max  (tokens drafteados, rec. 6)",
        "mtp_manual_hint": (
            "💡 Si el interruptor MTP no funciona con tu backend, desactívalo y usa "
            "el recuadro de Argumentos Extra (arriba) para introducir los flags manualmente.\n"
            "Ej. AtomicChat: --spec-type nextn --model-draft <ruta_modelo> --draft-max 4 -np 1"
        ),
    },

    "en": {
        # ── Navigation ────────────────────────────────────────────────
        "nav_chat":        "Chat",
        "nav_server":      "Server",
        "nav_logs":        "Logs",
        "nav_info":        "Model Info",
        "nav_download":    "Download",
        "nav_api":         "API Docs",
        "nav_voice":       "Voice",
        "nav_about":       "About",

        # ── Right sidebar ─────────────────────────────────────────────
        "loaded_model":    "LOADED MODEL",
        "no_model":        "None",
        "my_models":       "📂  My models",
        "download_models": "🌐  Download models",
        "server_label":    "SERVER",
        "server_stopped":  "Stopped",
        "server_port":     "port: —",
        "start_server":    "▶  Start server",
        "stop_server":     "■  Stop",
        "backend_label":   "BACKEND",
        "update_llama":    "⬆  Update llama.cpp",
        "controls":        "Controls",

        # ── Left sidebar ──────────────────────────────────────────────
        "app_title":       "⚡ LlamaStation",
        "conversations":   "CONVERSATIONS",
        "new_chat":        "＋  New chat",
        "no_history":      "No conversations yet\nStart chatting →",

        # ── Chat ──────────────────────────────────────────────────────
        "chat_title":      "Chat",
        "clear_chat":      "Clear",
        "thinking_on":         "👁 Razonamiento",
        "thinking_off":        "👁 Razonamiento OFF",
        "enable_thinking_on":  "🧠 Think: ON",
        "enable_thinking_off": "⚡ Think: OFF",
        "web_on":          "🌐 Web ON",
        "web_off":         "🌐 Web OFF",
        "system_label":    "System:",
        "system_ph":       "System prompt...",
        "send":            "Send",
        "sending":         "...",
        "you":             "You",
        "model":           "Model",

        # ── Server tab ────────────────────────────────────────────────
        "server_tab_title":   "Server Configuration",
        "server_exe":         "🔧 llama-server executable",
        "server_network":     "🌐 Network",
        "server_port_lbl":    "Port",
        "server_host_lbl":    "Host",
        "server_save":        "💾 Save",
        "server_cmd":         "📋 Generated command",
        "server_cmd_preview": "Update preview",
        "server_saved_ok":    "Configuration saved.",
        "browse":             "Browse...",
        "server_headless_sec": "🖥️  Headless Mode",
        "server_headless_desc": (
            "Start LlamaStation without the graphical interface.\n"
            "Uses the saved profile of the selected model."
        ),
        "server_headless_cmd_label": "Command (copy and paste into your terminal):",
        "server_headless_no_model":  "← Load a model first",
        "copy":               "📋 Copy",

        # ── Logs tab ──────────────────────────────────────────────────
        "logs_title":      "Server logs",
        "logs_clear":      "Clear",

        # ── Info tab ──────────────────────────────────────────────────
        "info_title":      "Model info",
        "info_no_server":  "⚠ Server is not running.",

        # ── Load model modal ──────────────────────────────────────────
        "modal_vram":      "Estimated VRAM usage",
        "modal_vram_hint": "depends on model and ctx",
        "modal_cancel":    "Cancel",
        "modal_load":      "Load Model  ↵",
        "sec_hardware":    "Hardware",
        "sec_multigpu":    "Multi-GPU  (2x RTX 3060)",
        "sec_cpu":         "CPU / NPU — No dedicated GPU",
        "sec_context":     "Context & Batch",
        "sec_sampling":    "Sampling",
        "sec_flags":       "Advanced Options",
        "sec_rope":        "RoPE Scaling  (context extension)",
        "sec_vision":      "Vision (mmproj)",
        "sec_extra":       "Extra Arguments (free text)",

        # Sliders
        "sl_gpu_layers":   "GPU Layers  — model layers on GPU (-1 = all, reduce if running out of VRAM)",
        "sl_threads":      "Threads  — CPU threads for the non-GPU part",
        "sl_threads_b":    "Threads Batch  — CPU threads for initial prompt processing",
        "sl_max_conc":     "Max Concurrent  — simultaneous requests, experimental, keep at 1",
        "sl_ctx":          "Context Size  — tokens the model remembers, more = more VRAM",
        "sl_batch":        "Batch Size  — tokens processed at once when loading prompt, more = faster",
        "sl_ubatch":       "UBatch Size  — batch subdivision, must be equal or less than Batch Size",
        "sl_max_tok":      "Max Tokens  — generation token limit per response (-1 = unlimited)",
        "sl_temp":         "Temperature  — creativity: 0=deterministic, 0.7=balanced, >1=more random",
        "sl_topk":         "Top K  — limits sampling to the K most probable tokens, 0=disabled",
        "sl_topp":         "Top P  — nucleus sampling: uses tokens until this cumulative probability",
        "sl_minp":         "Min P  — discards tokens with probability below this % of top token",
        "sl_rep_pen":      "Repeat Penalty  — penalizes word repetition, 1.0=no penalty",
        "sl_rep_n":        "Repeat Last N  — how many tokens back to look for Repeat Penalty",
        "sl_seed":         "Seed  — random seed, -1=different each time, set to reproduce results",
        "sl_rope_base":    "RoPE Freq Base  — base frequency for context scaling, 0=Auto",
        "sl_rope_scale":   "RoPE Freq Scale  — frequency scaling factor, 0=Auto",

        # Flags
        "fl_flash":        "Flash Attention",
        "fl_flash_tip":    "Speeds up attention computation on GPU. Always recommended.",
        "fl_mmap":         "mmap()",
        "fl_mmap_tip":     "Memory-map: loads the model by mapping the file into memory. More efficient.",
        "fl_mlock":        "mlock",
        "fl_mlock_tip":    "Locks the model in RAM so the OS won't swap it out.",
        "fl_contbatch":    "Continuous Batching",
        "fl_contbatch_tip":"Handles multiple requests simultaneously without waiting. Recommended.",
        "fl_kvoff":        "KV Cache Offload",
        "fl_kvoff_tip":    "Moves the KV cache to VRAM. Allows longer contexts.",
        "fl_keepmem":      "Keep In Memory",
        "fl_keepmem_tip":  "Doesn't unload the model when stopping the server. Faster next start.",
        "fl_embed":        "Embeddings endpoint",
        "fl_embed_tip":    "Enables /v1/embeddings in the API. Only needed for text vector generation.",
        "kv_label":        "K/V Cache — you can choose different types for K and V (asymmetry recommended with TurboQuant)",
        "kv_zero":         "0 = Auto (recommended). Useful for NTK/YaRN.",
        "sys_prompt":      "System Prompt",
        "sys_prompt_ph":   "You are a helpful assistant...",
        "extra_tip":       "For future flags: TurboQuant, etc.",
        "extra_ph":        "--override-kv ...",
        "split_tip":       "Split Mode — how the model is split across GPUs. 'layer' = layer split (recommended). 'row' = tensor row split.",
        "tensor_tip":      "Tensor Split — VRAM ratio per GPU (e.g. '1,1' = 50/50, '3,1' = 75/25). Empty = auto.",
        "tensor_ph":       "1,1  (leave empty for auto)",
        "cpu_mode_tip":    "Mode: GPU (default), CPU+RAM only, Hybrid, or Vulkan (AMD/Intel/NPU)",
        "cpu_tip2":        "CPU tip: raise Threads (Hardware section) to your CPU's physical core count for best performance.",
        "mmproj_tip":      "mmproj file for vision models. Auto-detected if it's in the same folder as the model.",
        "mmproj_none":     "No mmproj detected — this model has no vision, or set it manually.",
        "mmproj_ph":       "Path to mmproj-*.gguf file  (optional)",
        "mmproj_select":   "Select mmproj",
        "mmproj_disable":  "Disable mmproj (saves VRAM, disables vision)",

        # Buttons modal / browser
        "load_btn":        "Load  ↵",
        "cancel":          "Cancel",
        "delete_model":    "🗑️",
        "delete_model_confirm_title": "Delete model",
        "delete_model_confirm_msg":   "Permanently delete this file?\n\n{name}\n\n({size})",
        "delete_model_ok":  "Model deleted.",
        "delete_model_err": "Error deleting: {err}",
        "cleanup_backups_sec":   "🧹  Clean up old backups",
        "cleanup_backups_desc":  "Backups from previous llama.cpp/TurboQuant installations that are no longer needed.",
        "cleanup_backups_scan":  "Scan for backups",
        "cleanup_backups_none":  "No backups found.",
        "cleanup_backups_found": "{n} backup(s) found  •  {size} total",
        "cleanup_backups_del":   "Delete all",
        "cleanup_backups_done":  "✓ Backups deleted.",
        "change":          "Change",
        "manual_browse":   "📂  Browse file manually...",
        "folder_label":    "Folder:",
        "no_models_dir":   "No models in this folder.\n\nDownload models from the 🌐 Download tab\nor change the folder above.",
        "no_models_dir2":  "No .gguf models in this folder.\n\nDownload from 🌐 Download or change the folder.",
        "my_models_title": "📂  My models",
        "models_folder":   "Models folder",
        "select_gguf":     "Select GGUF model",

        # Error / info messages
        "err_no_model":    "Select a model first.",
        "err_no_server":   "Go to 'Server' and specify the path to llama-server.exe",
        "err_not_found":   "Not found",
        "err_no_model_title":   "No model",
        "err_server_title":     "llama-server not found",
        "err_title":            "Error",
        "warn_stopped":         "Server stopped",
        "warn_stopped_msg":     "Start the server before chatting.",
        "warn_backend_title":   "Backend not found",
        "saved_title":          "Saved",
        "save_file_title":      "Save generated file",
        "saved_msg":            "File saved at:\n",
        "err_save":             "Could not save:\n",
        "err_read":             "Could not read",
        "downloader_missing":   "llamastation_downloader.py not found.\nPlace the file in the same folder.",
        "attach_image":         "Attach image",
        "attach_files":         "Attach file(s)",
        "save_generated":       "Save",
        "describe_image":       "Describe this image.",
        "review_files":         "Review the files.",

        # ── API Docs tab ──────────────────────────────────────────────
        "api_title":       "📡  API Docs",
        "api_subtitle":    "llama.cpp OpenAI-compatible API",
        "api_intro":       (
            "  Your llama.cpp server exposes an OpenAI-compatible API at:\n"
            "  http://127.0.0.1:{port}/v1\n\n"
            "  Any app that uses the OpenAI API works by pointing it to this URL."
        ),
        "api_sec_chat":    "💬  Chat Completions",
        "api_sec_stream":  "⚡  Streaming (real-time response)",
        "api_sec_models":  "📋  Models",
        "api_sec_health":  "🩺  Server status",
        "api_sec_headless":"🖥️  Headless Mode (no window)",
        "api_chat_desc":   "Generate a chat response",
        "api_stream_desc": "SSE Streaming — real-time tokens",
        "api_models_desc": "List loaded models",
        "api_health_desc": "Check if the server is ready",

        # ── Theme / language ──────────────────────────────────────────
        "theme_to_light":  "☀  Light mode",
        "theme_to_dark":   "🌙  Dark mode",
        "lang_btn":        "🌐  Español",

        # ── Update ────────────────────────────────────────────────────
        "update_title":    "⬆  Update llama.cpp",
        "update_subtitle": "official backend only · forks not affected",
        "update_close":    "Close",
        "update_install":  "Install update",
        "update_uptodate": "Already up to date",
        "update_installing":"Installing...",
        "update_done":     "✓ Installed",
        "update_retry":    "Retry",

        # Server — status
        "srv_running":     "Running",
        "srv_running_ext": "Running (external)",
        "srv_starting":    "Starting...",
        "srv_port":        "port: {port}",
        "srv_stopped_log": "Server stopped. VRAM freed.",

        # ── Watchdog ─────────────────────────────────────────────────
        "watchdog_sec":           "🛡️  Server Watchdog",
        "watchdog_auto_relaunch": "Automatically restart server if it crashes",
        "watchdog_crashed_title": "⚠️  Server crashed",
        "watchdog_crashed_msg":   "The server stopped unexpectedly (code {rc}).\n\nLast log lines:\n\n{log}\n\nCheck the Logs tab for details.",
        "watchdog_relaunch_log":  "🔄 Watchdog: restarting server in {delay}s...",
        "watchdog_relaunching":   "Restarting...",

        # ── Sound ────────────────────────────────────────────────────
        "sound_on":        "🔔  Sound ON",
        "sound_off":       "🔕  Sound OFF",

        # ── Downloader ───────────────────────────────────────────────
        "dl_title":        "🌐  Download Models",
        "dl_save_in":      "Save to:",
        "dl_search_ph":    "🔍  Search model... (e.g. llama-3, qwen2.5, mistral)",
        "dl_search_btn":   "Search",
        "dl_arch":         "Architecture:",
        "dl_author":       "Author:",
        "dl_searching":    "Searching...",
        "dl_no_results":   "No models found.\nTry a different search.",
        "dl_start_hint":   "Search for a model to get started\nor browse the most popular →",
        "dl_select_hint":  "← Select a model from the list",
        "dl_files_title":  "AVAILABLE FILES",
        "dl_quant_filter": "Filter quant:",
        "dl_files_ph":     "Select a model to see its files",
        "dl_loading":      "Loading files...",
        "dl_no_files":     "No files match this filter",
        "dl_downloads":    "ACTIVE DOWNLOADS",
        "dl_no_downloads": "No active downloads",
        "dl_view_hf":      "View on HF ↗",
        "dl_downloads_lbl":"downloads",
        "dl_updated":      "Updated:",
        "dl_caps_warning": "⚠ Shown capabilities may be incomplete",

        # ── MTP ──────────────────────────────────────────────────────
        "mtp_activate":    "Enable MTP",
        "mtp_draft_n_max": "spec-draft-n-max  (draft tokens, rec. 6)",
        "mtp_manual_hint": (
            "💡 If the MTP toggle doesn't work with your backend, disable it and use "
            "the Extra Arguments box (above) to set the flags manually.\n"
            "E.g. AtomicChat: --spec-type nextn --model-draft <model_path> --draft-max 4 -np 1"
        ),
    },
}

# Idioma activo — se actualiza desde LlamaStation al cambiar
_LANG = "es"

def set_lang(lang: str):
    global _LANG
    if lang in STRINGS:
        _LANG = lang

def get_lang() -> str:
    return _LANG

def T(key: str, **kwargs) -> str:
    """Devuelve el string traducido. kwargs para interpolación: T('srv_port', port=8080)"""
    s = STRINGS.get(_LANG, STRINGS["es"]).get(key)
    if s is None:
        # Fallback al español
        s = STRINGS["es"].get(key, key)
    if kwargs:
        try:
            return s.format(**kwargs)
        except Exception:
            return s
    return s
