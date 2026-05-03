"""
LlamaStation — Model Downloader
Descargador de modelos GGUF desde Hugging Face, estilo LM Studio.
Usa la API pública de HF (sin autenticación para modelos públicos).

Integración: importa ModelDownloaderFrame y añádelo como tab en LlamaForge.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import requests
import json
import os
import time
from pathlib import Path
from datetime import datetime

try:
    from llamaforge_i18n import T
except ImportError:
    def T(key, **kwargs):
        # Fallback si no está el archivo i18n
        return key


# ── Constantes HuggingFace ─────────────────────────────────────────────────

HF_API_SEARCH   = "https://huggingface.co/api/models"
HF_API_MODEL    = "https://huggingface.co/api/models/{repo_id}"
HF_API_FILES    = "https://huggingface.co/api/models/{repo_id}"   # siblings en la misma respuesta
HF_DOWNLOAD_URL = "https://huggingface.co/{repo_id}/resolve/main/{filename}"
HF_MODEL_PAGE   = "https://huggingface.co/{repo_id}"

# Autores conocidos de compilaciones GGUF de calidad
KNOWN_PROVIDERS = [
    "bartowski", "unsloth", "lmstudio-community", "TheBloke",
    "QuantFactory", "MaziyarPanahi", "second-state", "robotics-diffusion-transformer",
    "mlabonne", "Qwen", "google", "meta-llama", "mistralai", "microsoft",
    "NousResearch", "teknium",
]

# Tags de arquitectura populares para filtrar
ARCH_TAGS = {
    "Todas":        None,
    "Llama":        "llama",
    "Qwen":         "qwen2",
    "Gemma":        "gemma",
    "Mistral":      "mistral",
    "Phi":          "phi",
    "Falcon":       "falcon",
    "DeepSeek":     "deepseek",
    "Command-R":    "cohere",
    "Nemotron":     "nemotron",
    "Mixtral (MoE)":"mixtral",
}

# Capabilities detectables desde los tags de HF
CAPABILITY_BADGES = [
    # (keywords_en_tags,           label,          color)
    (["vision", "image-text-to-text", "image-text", "vl", "visual", "multimodal",
      "llava", "qwen-vl", "gemma.*vision", "minicpm-v", "internvl"],
     "👁 Vision",    "#4a7a6a"),

    (["tool_use", "tool-use", "function-calling", "function_calling",
      "tool_call", "toolcall", "tools"],
     "⚙ Tools",     "#7c6aff"),

    (["reasoning", "thinking", "qwq", "deepseek-r", "r1", "marco-o1",
      "o1", "cot", "chain-of-thought"],
     "◈ Thinking",  "#e8a838"),

    (["code", "coding", "coder", "starcoder", "codellama", "deepseek-coder",
      "qwen.*coder", "wizard.*code"],
     "</> Code",    "#4a9eff"),

    (["math", "mathstral", "deepseek-math", "numina"],
     "∑ Math",      "#e05555"),

    (["multilingual", "multi-lingual", "polyglot"],
     "⊕ Multi-lang","#4caf82"),

    (["audio", "speech", "whisper", "asr"],
     "♪ Audio",     "#c084fc"),

    (["video"],
     "▶ Video",     "#f97316"),
]

# Detección adicional por nombre del modelo (para cuando HF no pone los tags bien)
NAME_CAPABILITIES = [
    (["gemma-4", "gemma4", "qwen2-vl", "qwen2vl", "qwen2.5-vl", "llava",
      "minicpm-v", "internvl", "idefics", "paligemma", "moondream",
      "qwen3.*b-a", "gemma.*e.*b"],   "👁 Vision",   "#4a7a6a"),
    (["qwen3", "qwen2.5", "llama-3.1", "llama-3.2", "llama-3.3",
      "mistral-nemo", "command-r", "hermes", "functionary"],
                                       "⚙ Tools",    "#7c6aff"),
    (["qwq", "deepseek-r", "-r1", "o1", "skywork-o", "light-r1",
      "gemma-3.*glm", "thinking"],     "◈ Thinking", "#e8a838"),
]

def detect_capabilities(tags, model_id=""):
    """
    Extrae badges de capabilities desde los tags de HF y el nombre del modelo.
    Combina ambas fuentes para mayor cobertura.
    """
    import re as _re
    seen  = set()
    result = []

    tags_str = " ".join(t.lower() for t in (tags or []))
    name_low = model_id.lower()

    # 1. Buscar en tags
    for keywords, label, color in CAPABILITY_BADGES:
        if label in seen:
            continue
        for kw in keywords:
            if _re.search(kw, tags_str):
                seen.add(label)
                result.append((label, color))
                break

    # 2. Buscar en nombre del modelo (fallback)
    for keywords, label, color in NAME_CAPABILITIES:
        if label in seen:
            continue
        for kw in keywords:
            if _re.search(kw, name_low):
                seen.add(label)
                result.append((label, color))
                break

    return result

# Cuantizaciones GGUF más comunes
QUANT_FILTERS = [
    "Todas", "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q6_K", "Q8_0",
    "IQ4_XS", "IQ3_M", "IQ2_M", "F16", "BF16",
]


def human_size(n_bytes):
    """Convierte bytes a string legible."""
    if n_bytes is None:
        return "?"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def search_hf_models(query, arch_tag=None, limit=30, provider=None):
    """
    Busca modelos GGUF en HuggingFace.
    Devuelve lista de dicts con info básica.
    """
    params = {
        "search": query,
        "filter":  "gguf",
        "limit":   limit,
        "sort":    "downloads",
        "direction": "-1",
        "full":    "true",
    }
    if arch_tag:
        params["filter"] = f"gguf,{arch_tag}"
    if provider:
        params["author"] = provider

    headers = {"User-Agent": "LlamaStation/2.3"}
    resp = requests.get(HF_API_SEARCH, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _estimate_size_from_name(fname, params_b=None):
    """
    Estima el tamaño en bytes de un GGUF a partir del nombre y cuantización.
    Usado como fallback cuando la API no devuelve el tamaño.
    params_b: número de parámetros en miles de millones (ej: 7 para 7B)
    """
    # Bits por parámetro según cuantización
    bpw = {
        "IQ1_S": 1.56, "IQ1_M": 1.75, "IQ2_XXS": 2.06, "IQ2_XS": 2.31,
        "IQ2_S": 2.50, "IQ2_M": 2.70, "Q2_K": 2.96, "IQ3_XXS": 3.06,
        "IQ3_XS": 3.30, "Q3_K_S": 3.50, "IQ3_S": 3.50, "IQ3_M": 3.66,
        "Q3_K_M": 3.91, "Q3_K_L": 4.27, "IQ4_XS": 4.25, "IQ4_NL": 4.50,
        "Q4_0": 4.55, "Q4_K_S": 4.59, "Q4_K_M": 4.85, "Q4_K_L": 4.90,
        "Q5_0": 5.54, "Q5_K_S": 5.54, "Q5_K_M": 5.69, "Q5_K_L": 5.75,
        "Q6_K": 6.57, "Q8_0": 8.50, "F16": 16.0, "BF16": 16.0,
    }
    fname_up = fname.upper()
    bits = None
    for q, b in bpw.items():
        if q in fname_up:
            bits = b
            break
    if bits is None or params_b is None:
        return None
    # Tamaño ≈ params * 1e9 * bits / 8 bytes + overhead ~10%
    return int(params_b * 1e9 * bits / 8 * 1.10)


def _extract_params_billions(repo_id):
    """Extrae el número de parámetros en B desde el nombre del repo."""
    import re
    m = re.search(r'(\d+(?:\.\d+)?)[-_]?[Bb](?:|[^a-zA-Z])', repo_id)
    if m:
        return float(m.group(1))
    return None


def get_model_files(repo_id):
    """
    Devuelve la lista de archivos de un repositorio HF con tamaños.
    Intenta varias estrategias para obtener el tamaño real.
    """
    url = HF_API_MODEL.format(repo_id=repo_id)
    headers = {"User-Agent": "LlamaStation/2.3"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    siblings = data.get("siblings", [])

    # Intentar obtener tamaños con blobs=true si faltan
    has_sizes = any(s.get("size") for s in siblings if s.get("rfilename","").endswith(".gguf"))
    if not has_sizes:
        try:
            r2 = requests.get(url + "?blobs=true", headers=headers, timeout=15)
            if r2.status_code == 200:
                d2 = r2.json()
                # Merge sizes
                size_map = {s.get("rfilename"): s.get("size") or s.get("blobSize")
                            for s in d2.get("siblings", [])}
                for s in siblings:
                    if not s.get("size") and s.get("rfilename") in size_map:
                        s["size"] = size_map[s["rfilename"]]
        except Exception:
            pass

    params_b = _extract_params_billions(repo_id)
    files = []
    for s in siblings:
        fname = s.get("rfilename", "")
        if not fname.endswith(".gguf"):
            continue
        size = s.get("size") or s.get("blobSize") or None
        # Fallback: estimar desde nombre si la API no da tamaño
        if not size:
            size = _estimate_size_from_name(fname, params_b)
        files.append({
            "name":      fname,
            "size":      size,
            "estimated": size is not None and not (s.get("size") or s.get("blobSize")),
            "is_mmproj": "mmproj" in fname.lower(),
        })
    files.sort(key=lambda x: (x["is_mmproj"], x["name"]))
    return files


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME PRINCIPAL — se embebe como tab en LlamaForge
# ══════════════════════════════════════════════════════════════════════════════

class ModelDownloaderFrame(ctk.CTkFrame):
    """
    Tab completo de descarga de modelos.
    Uso:
        frame = ModelDownloaderFrame(parent, colors=C, models_dir="C:/models")
        frame.pack(fill="both", expand=True)
    """

    def __init__(self, parent, colors: dict, models_dir: str = ""):
        super().__init__(parent, fg_color=colors["bg"], corner_radius=0)
        self.C = colors
        self.models_dir = models_dir or str(Path.home() / "models")
        self._search_results   = []       # lista de modelos del search
        self._selected_model   = None     # dict del modelo seleccionado
        self._model_files      = []       # archivos .gguf del modelo
        self._downloads        = {}       # {filename: DownloadTask}
        self._search_thread    = None
        self._files_thread     = None

        self._build()

    # ── Build UI ──────────────────────────────────────────────────────────

    def _build(self):
        C = self.C

        # ── Top bar (búsqueda) ─────────────────────────────────────────
        top = ctk.CTkFrame(self, fg_color=C["panel"], corner_radius=0, height=64)
        top.pack(fill="x"); top.pack_propagate(False)

        inner_top = ctk.CTkFrame(top, fg_color="transparent")
        inner_top.pack(fill="both", expand=True, padx=16, pady=10)

        # Icono + título
        ctk.CTkLabel(inner_top, text=T("dl_title"),
                     font=ctk.CTkFont("Consolas", 15, "bold"),
                     text_color=C["accent2"]).pack(side="left")

        ctk.CTkLabel(inner_top, text=" Hugging Face · GGUF ",
                     font=ctk.CTkFont("Consolas", 9, "bold"),
                     fg_color=C["accent"], text_color="white",
                     corner_radius=4).pack(side="left", padx=8)

        # Directorio de destino
        dest_frame = ctk.CTkFrame(inner_top, fg_color="transparent")
        dest_frame.pack(side="right")
        ctk.CTkLabel(dest_frame, text=T("dl_save_in"),
                     font=ctk.CTkFont("Consolas", 10),
                     text_color=C["sub"]).pack(side="left")
        self.dest_var = tk.StringVar(value=self.models_dir)
        ctk.CTkEntry(dest_frame, textvariable=self.dest_var, width=240,
                     fg_color=C["input"], text_color=C["text"],
                     font=ctk.CTkFont("Consolas", 10), height=28
                     ).pack(side="left", padx=4)
        ctk.CTkButton(dest_frame, text="📁", width=32, height=28,
                      fg_color=C["card2"], hover_color=C["border"],
                      text_color=C["text"], font=ctk.CTkFont("Consolas", 13),
                      command=self._pick_dest_dir
                      ).pack(side="left")

        # ── Barra de filtros ───────────────────────────────────────────
        filters = ctk.CTkFrame(self, fg_color=C["card"], corner_radius=0, height=48)
        filters.pack(fill="x"); filters.pack_propagate(False)
        fi = ctk.CTkFrame(filters, fg_color="transparent")
        fi.pack(fill="both", expand=True, padx=12, pady=6)

        # Búsqueda texto
        self.search_var = tk.StringVar()
        self.search_entry = ctk.CTkEntry(fi, textvariable=self.search_var,
                                          placeholder_text=T("dl_search_ph"),
                                          width=260, height=34,
                                          fg_color=C["input"], text_color=C["text"],
                                          font=ctk.CTkFont("Consolas", 12))
        self.search_entry.pack(side="left")
        self.search_entry.bind("<Return>", lambda e: self._do_search())

        ctk.CTkButton(fi, text=T("dl_search_btn"), width=80, height=34,
                      fg_color=C["accent"], hover_color="#6457e0",
                      font=ctk.CTkFont("Consolas", 12, "bold"),
                      command=self._do_search).pack(side="left", padx=(6, 16))

        ctk.CTkLabel(fi, text=T("dl_arch"),
                     font=ctk.CTkFont("Consolas", 10), text_color=C["sub"]
                     ).pack(side="left")
        self.arch_var = tk.StringVar(value="Todas")
        ctk.CTkOptionMenu(fi, variable=self.arch_var,
                          values=list(ARCH_TAGS.keys()),
                          width=120, height=30,
                          fg_color=C["card2"], button_color=C["border"],
                          dropdown_fg_color=C["card"],
                          text_color=C["text"],
                          font=ctk.CTkFont("Consolas", 11),
                          command=lambda _: self._do_search()
                          ).pack(side="left", padx=(4, 12))

        # Filtro proveedor
        ctk.CTkLabel(fi, text=T("dl_author"),
                     font=ctk.CTkFont("Consolas", 10), text_color=C["sub"]
                     ).pack(side="left")
        self.provider_var = tk.StringVar(value="Todos")
        ctk.CTkOptionMenu(fi, variable=self.provider_var,
                          values=["Todos"] + KNOWN_PROVIDERS,
                          width=160, height=30,
                          fg_color=C["card2"], button_color=C["border"],
                          dropdown_fg_color=C["card"],
                          text_color=C["text"],
                          font=ctk.CTkFont("Consolas", 11),
                          command=lambda _: self._do_search()
                          ).pack(side="left", padx=4)

        # Resultados: contador
        self.results_count = ctk.CTkLabel(fi, text="",
                                           font=ctk.CTkFont("Consolas", 10),
                                           text_color=C["sub"])
        self.results_count.pack(side="right", padx=8)

        # ── Panel dividido ─────────────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        body.pack(fill="both", expand=True)

        # Lista de modelos (izquierda)
        left = ctk.CTkFrame(body, fg_color=C["panel"], corner_radius=0, width=400)
        left.pack(side="left", fill="y"); left.pack_propagate(False)

        self.model_list_frame = ctk.CTkScrollableFrame(left, fg_color="transparent", corner_radius=0)
        self.model_list_frame.pack(fill="both", expand=True)

        self._empty_label = ctk.CTkLabel(self.model_list_frame,
                                          text=T("dl_start_hint"),
                                          font=ctk.CTkFont("Consolas", 12),
                                          text_color=C["dim"],
                                          justify="center")
        self._empty_label.pack(pady=80)

        # Panel derecho: detalle + archivos + descargas
        right = ctk.CTkFrame(body, fg_color=C["bg"], corner_radius=0)
        right.pack(side="left", fill="both", expand=True)

        # Detalle del modelo seleccionado
        self.detail_card = ctk.CTkFrame(right, fg_color=C["card"], corner_radius=10)
        self.detail_card.pack(fill="x", padx=16, pady=(12, 6))
        self._build_detail_empty()

        # Lista de archivos GGUF
        files_hdr = ctk.CTkFrame(right, fg_color="transparent")
        files_hdr.pack(fill="x", padx=16, pady=(0, 4))
        ctk.CTkLabel(files_hdr, text=T("dl_files_title"),
                     font=ctk.CTkFont("Consolas", 9, "bold"),
                     text_color=C["sub"]).pack(side="left")
        self.quant_var = tk.StringVar(value="Todas")
        ctk.CTkOptionMenu(files_hdr, variable=self.quant_var,
                          values=QUANT_FILTERS,
                          width=110, height=26,
                          fg_color=C["card2"], button_color=C["border"],
                          dropdown_fg_color=C["card"],
                          text_color=C["text"],
                          font=ctk.CTkFont("Consolas", 10),
                          command=lambda _: self._render_files()
                          ).pack(side="right")
        ctk.CTkLabel(files_hdr, text=T("dl_quant_filter"),
                     font=ctk.CTkFont("Consolas", 10), text_color=C["sub"]
                     ).pack(side="right", padx=4)

        self.files_scroll = ctk.CTkScrollableFrame(right, fg_color=C["card"],
                                                    corner_radius=10, height=240)
        self.files_scroll.pack(fill="x", padx=16, pady=(0, 6))
        self._files_placeholder = ctk.CTkLabel(self.files_scroll,
                                                text=T("dl_files_ph"),
                                                font=ctk.CTkFont("Consolas", 11),
                                                text_color=C["dim"])
        self._files_placeholder.pack(pady=24)

        # Aviso etiquetas no fiables
        ctk.CTkLabel(right, text=T("dl_caps_warning"),
                     font=ctk.CTkFont("Consolas", 9),
                     text_color=C["dim"]).pack(anchor="w", padx=16, pady=(0, 2))

        # Cola de descargas activas
        ctk.CTkLabel(right, text=T("dl_downloads"),
                     font=ctk.CTkFont("Consolas", 9, "bold"),
                     text_color=C["sub"]).pack(anchor="w", padx=16, pady=(4, 2))

        self.downloads_scroll = ctk.CTkScrollableFrame(right, fg_color=C["card"],
                                                        corner_radius=10)
        self.downloads_scroll.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        self._no_downloads_label = ctk.CTkLabel(self.downloads_scroll,
                                                 text=T("dl_no_downloads"),
                                                 font=ctk.CTkFont("Consolas", 11),
                                                 text_color=C["dim"])
        self._no_downloads_label.pack(pady=20)

        # Carga automática al abrir
        self.after(300, self._load_popular)

    def _build_detail_empty(self):
        for w in self.detail_card.winfo_children():
            w.destroy()
        ctk.CTkLabel(self.detail_card,
                     text=T("dl_select_hint"),
                     font=ctk.CTkFont("Consolas", 12),
                     text_color=self.C["dim"]).pack(pady=20)

    def _build_detail(self, model):
        C = self.C
        for w in self.detail_card.winfo_children():
            w.destroy()

        di = ctk.CTkFrame(self.detail_card, fg_color="transparent")
        di.pack(fill="x", padx=16, pady=12)

        # Nombre + autor
        top_row = ctk.CTkFrame(di, fg_color="transparent")
        top_row.pack(fill="x")
        ctk.CTkLabel(top_row,
                     text=model.get("modelId", model.get("id", "?")),
                     font=ctk.CTkFont("Consolas", 13, "bold"),
                     text_color=C["text"],
                     wraplength=500, justify="left"
                     ).pack(side="left", anchor="w")

        # Tags: arquitectura, downloads
        tags_row = ctk.CTkFrame(di, fg_color="transparent")
        tags_row.pack(fill="x", pady=(6, 0))

        tags = model.get("tags", [])
        for tag in tags:
            if tag in ("gguf", "text-generation"):
                continue
            if len(tag) > 20:
                continue
            ctk.CTkLabel(tags_row, text=f" {tag} ",
                         font=ctk.CTkFont("Consolas", 9),
                         fg_color=C["card2"], text_color=C["sub"],
                         corner_radius=4).pack(side="left", padx=(0, 4))
            if tags_row.winfo_children().__len__() > 6:
                break

        # Capability badges en el panel de detalle
        repo_id_caps = model.get("modelId", model.get("id", ""))
        caps = detect_capabilities(model.get("tags", []), repo_id_caps)
        if caps:
            cap_row = ctk.CTkFrame(di, fg_color="transparent")
            cap_row.pack(fill="x", pady=(8, 0))
            for label, color in caps:
                ctk.CTkLabel(cap_row, text=f" {label} ",
                             font=ctk.CTkFont("Consolas", 10, "bold"),
                             fg_color=color, text_color="white",
                             corner_radius=5).pack(side="left", padx=(0, 6))

        # Stats
        stats_row = ctk.CTkFrame(di, fg_color="transparent")
        stats_row.pack(fill="x", pady=(8, 0))

        dl = model.get("downloads", 0)
        likes = model.get("likes", 0)
        updated = model.get("lastModified", "")[:10] if model.get("lastModified") else "?"

        for icon, val in [
            ("⬇", f"{dl:,} {T('dl_downloads_lbl')}"),
            ("♥", f"{likes} likes"),
            ("🕒", f"{T('dl_updated')} {updated}"),
        ]:
            ctk.CTkLabel(stats_row, text=f"{icon} {val}",
                         font=ctk.CTkFont("Consolas", 10),
                         text_color=C["sub"]).pack(side="left", padx=(0, 16))

        # Botón HF
        ctk.CTkButton(stats_row, text=T("dl_view_hf"), width=100, height=24,
                      fg_color="transparent", hover_color=C["card2"],
                      text_color=C["accent2"],
                      font=ctk.CTkFont("Consolas", 10),
                      command=lambda: self._open_hf(model.get("modelId", model.get("id", "")))
                      ).pack(side="right")

    # ── Lógica de búsqueda ────────────────────────────────────────────────

    def _load_popular(self):
        """Carga modelos populares al abrir el tab."""
        self.search_var.set("")
        self._do_search()

    def _do_search(self):
        query    = self.search_var.get().strip()
        arch_key = self.arch_var.get()
        arch_tag = ARCH_TAGS.get(arch_key)
        provider = self.provider_var.get()
        if provider == "Todos":
            provider = None

        self._empty_label.configure(text=T("dl_searching"))
        self._clear_model_list()
        self._empty_label.pack(pady=80)
        self.results_count.configure(text="...")

        # Cancelar búsqueda anterior si sigue en marcha
        self._search_thread = threading.Thread(
            target=self._search_worker,
            args=(query, arch_tag, provider),
            daemon=True
        )
        self._search_thread.start()

    def _search_worker(self, query, arch_tag, provider):
        try:
            results = search_hf_models(
                query=query or "gguf",
                arch_tag=arch_tag,
                limit=40,
                provider=provider
            )
            self.after(0, lambda: self._render_results(results))
        except Exception as e:
            self.after(0, lambda: self._search_error(str(e)))

    def _render_results(self, results):
        C = self.C
        self._search_results = results
        self._clear_model_list()
        self.results_count.configure(text=f"{len(results)} modelos")

        if not results:
            self._empty_label.configure(text=T("dl_no_results"))
            self._empty_label.pack(pady=80)
            return

        for model in results:
            self._add_model_card(model)

    def _search_error(self, err):
        self._empty_label.configure(text=f"Error de conexión:\n{err}")
        self._empty_label.pack(pady=80)
        self.results_count.configure(text="Error")

    def _clear_model_list(self):
        for w in self.model_list_frame.winfo_children():
            w.destroy()
        self._empty_label = ctk.CTkLabel(self.model_list_frame, text="",
                                          font=ctk.CTkFont("Consolas", 12),
                                          text_color=self.C["dim"], justify="center")

    def _add_model_card(self, model):
        C = self.C
        repo_id = model.get("modelId", model.get("id", "?"))
        author, _, name = repo_id.partition("/")
        dl = model.get("downloads", 0)

        # Nombre truncado a 1 línea
        display_name = name or repo_id
        if len(display_name) > 36:
            display_name = display_name[:34] + "…"

        # Card con altura fija via pack_propagate
        card = ctk.CTkFrame(self.model_list_frame, fg_color=C["card"],
                             corner_radius=8, cursor="hand2")
        card.pack(fill="x", padx=8, pady=2)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=12, pady=(7, 7))

        # Línea 1: nombre
        ctk.CTkLabel(inner,
                     text=display_name,
                     font=ctk.CTkFont("Consolas", 11, "bold"),
                     text_color=C["text"],
                     anchor="w"
                     ).pack(anchor="w", fill="x")

        # Línea 2: autor + descargas (una sola fila)
        sub = ctk.CTkFrame(inner, fg_color="transparent")
        sub.pack(fill="x", pady=(1, 0))

        ctk.CTkLabel(sub, text=author,
                     font=ctk.CTkFont("Consolas", 10),
                     text_color=C["accent"]).pack(side="left")
        ctk.CTkLabel(sub, text=f"  ⬇ {dl:,}",
                     font=ctk.CTkFont("Consolas", 10),
                     text_color=C["sub"]).pack(side="left")

        # Capability badges
        repo_id_caps = model.get("modelId", model.get("id", ""))
        caps = detect_capabilities(model.get("tags", []), repo_id_caps)
        if caps:
            cap_row = ctk.CTkFrame(inner, fg_color="transparent")
            cap_row.pack(fill="x", pady=(3, 0))
            for label, color in caps[:4]:
                ctk.CTkLabel(cap_row, text=label,
                             font=ctk.CTkFont("Consolas", 8, "bold"),
                             fg_color=color, text_color="white",
                             corner_radius=4).pack(side="left", padx=(0, 4))

        def on_enter(e, c=card): c.configure(fg_color=C["card2"])
        def on_leave(e, c=card): c.configure(fg_color=C["card"])
        def on_click(e, m=model): self._select_model(m)

        for w in [card, inner, sub] + list(sub.winfo_children()) + list(inner.winfo_children()):
            try:
                w.bind("<Enter>", on_enter)
                w.bind("<Leave>", on_leave)
                w.bind("<Button-1>", on_click)
            except Exception:
                pass

    # ── Selección de modelo y carga de archivos ───────────────────────────

    def _select_model(self, model):
        self._selected_model = model
        self._model_files = []
        self._build_detail(model)
        self._render_files_loading()

        repo_id = model.get("modelId", model.get("id", ""))
        self._files_thread = threading.Thread(
            target=self._load_files_worker,
            args=(repo_id,),
            daemon=True
        )
        self._files_thread.start()

    def _load_files_worker(self, repo_id):
        try:
            files = get_model_files(repo_id)
            self.after(0, lambda: self._set_model_files(files))
        except Exception as e:
            self.after(0, lambda: self._files_error(str(e)))

    def _set_model_files(self, files):
        self._model_files = files
        self._render_files()

    def _render_files_loading(self):
        for w in self.files_scroll.winfo_children():
            w.destroy()
        ctk.CTkLabel(self.files_scroll, text=T("dl_loading"),
                     font=ctk.CTkFont("Consolas", 11),
                     text_color=self.C["sub"]).pack(pady=24)

    def _render_files(self):
        C = self.C
        for w in self.files_scroll.winfo_children():
            w.destroy()

        quant_filter = self.quant_var.get()
        files = self._model_files

        if quant_filter != "Todas":
            files = [f for f in files if quant_filter.lower() in f["name"].lower()]

        if not files:
            ctk.CTkLabel(self.files_scroll,
                         text=T("dl_no_files"),
                         font=ctk.CTkFont("Consolas", 11),
                         text_color=C["dim"]).pack(pady=24)
            return

        for finfo in files:
            self._add_file_row(finfo)

    def _add_file_row(self, finfo):
        C = self.C
        fname = finfo["name"]
        fsize = finfo.get("size")
        is_mmproj = finfo.get("is_mmproj", False)

        # Detectar cuantización del nombre
        quant = ""
        for q in ["Q4_K_M","Q4_K_S","Q5_K_M","Q6_K","Q8_0","Q4_0","Q5_0",
                  "IQ4_XS","IQ4_NL","IQ3_M","IQ3_XS","IQ2_M","IQ2_XS","F16","BF16"]:
            if q.lower() in fname.lower():
                quant = q
                break

        # Color del badge según calidad
        quant_colors = {
            "Q4_K_M": "#7c6aff", "Q5_K_M": "#5ba4ff", "Q6_K": "#4caf82",
            "Q8_0": "#e8a838", "F16": "#e05555", "BF16": "#e05555",
            "IQ4_XS": "#9b72ff", "IQ4_NL": "#9b72ff",
            "Q4_K_S": "#8b7aff", "Q4_0": "#aaa",
        }
        badge_color = quant_colors.get(quant, C["card2"])

        row = ctk.CTkFrame(self.files_scroll, fg_color=C["panel"] if is_mmproj else "transparent",
                           corner_radius=6)
        row.pack(fill="x", padx=6, pady=3)

        inner = ctk.CTkFrame(row, fg_color="transparent")
        inner.pack(fill="x", padx=10, pady=8)

        # Izquierda: badge cuantización + nombre limpio + tamaño
        left = ctk.CTkFrame(inner, fg_color="transparent")
        left.pack(side="left", fill="x", expand=True)

        top_line = ctk.CTkFrame(left, fg_color="transparent")
        top_line.pack(fill="x", anchor="w")

        # Badge de cuantización bien visible
        if is_mmproj:
            ctk.CTkLabel(top_line, text=" mmproj ",
                         font=ctk.CTkFont("Consolas", 10, "bold"),
                         fg_color="#4a7a6a", text_color="white",
                         corner_radius=4).pack(side="left", padx=(0, 8))
            ctk.CTkLabel(top_line, text="Proyector de visión",
                         font=ctk.CTkFont("Consolas", 11, "bold"),
                         text_color=C["text"]).pack(side="left")
        else:
            if quant:
                ctk.CTkLabel(top_line, text=f" {quant} ",
                             font=ctk.CTkFont("Consolas", 11, "bold"),
                             fg_color=badge_color, text_color="white",
                             corner_radius=4).pack(side="left", padx=(0, 8))
            # Nombre del archivo (sin la parte de cuantización para no repetir)
            clean = fname
            for q in ["Q4_K_M","Q4_K_S","Q5_K_M","Q6_K","Q8_0","Q4_0","Q5_0",
                      "IQ4_XS","IQ4_NL","IQ3_M","IQ3_XS","IQ2_M","IQ2_XS","F16","BF16"]:
                clean = clean.replace(f"-{q}", "").replace(f"_{q}", "").replace(f".{q}", "")
            # Truncar si es muy largo
            if len(clean) > 44:
                clean = clean[:42] + "…"
            ctk.CTkLabel(top_line, text=clean,
                         font=ctk.CTkFont("Consolas", 11),
                         text_color=C["text"], anchor="w"
                         ).pack(side="left")

        # Tamaño en línea de abajo
        if fsize:
            is_est = finfo.get("estimated", False)
            size_txt = ("~" if is_est else "") + human_size(fsize)
            size_color = C["sub"] if is_est else C["accent2"]
            ctk.CTkLabel(left, text=size_txt,
                         font=ctk.CTkFont("Consolas", 10, "bold"),
                         text_color=size_color, anchor="w"
                         ).pack(anchor="w", pady=(2, 0))
        else:
            ctk.CTkLabel(left, text="Tamaño desconocido",
                         font=ctk.CTkFont("Consolas", 10),
                         text_color=C["dim"], anchor="w"
                         ).pack(anchor="w", pady=(2, 0))

        # Botón descargar
        repo_id = self._selected_model.get("modelId", self._selected_model.get("id", ""))
        ctk.CTkButton(inner, text="⬇ Descargar", width=110, height=32,
                      fg_color=C["accent"], hover_color="#6457e0",
                      text_color="white",
                      font=ctk.CTkFont("Consolas", 11, "bold"),
                      command=lambda r=repo_id, f=fname, s=fsize: self._start_download(r, f, s)
                      ).pack(side="right", padx=(8, 0))

    def _files_error(self, err):
        for w in self.files_scroll.winfo_children():
            w.destroy()
        ctk.CTkLabel(self.files_scroll,
                     text=f"Error al cargar archivos:\n{err}",
                     font=ctk.CTkFont("Consolas", 11),
                     text_color=self.C["red"]).pack(pady=24)

    # ── Descarga ──────────────────────────────────────────────────────────

    def _start_download(self, repo_id, filename, file_size):
        C = self.C
        dest_dir = self.dest_var.get().strip()
        if not dest_dir:
            messagebox.showerror("Error", "Selecciona un directorio de destino.")
            return

        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename.split("/")[-1])

        if dest_path in self._downloads:
            messagebox.showinfo("Ya en curso", f"{filename} ya se está descargando.")
            return

        if os.path.isfile(dest_path):
            if not messagebox.askyesno("Archivo existe",
                                        f"{filename} ya existe en el destino.\n¿Sobreescribir?"):
                return

        # Crear widget de progreso
        self._no_downloads_label.pack_forget()

        dl_card = ctk.CTkFrame(self.downloads_scroll, fg_color=C["card2"], corner_radius=8)
        dl_card.pack(fill="x", padx=6, pady=4)

        dl_inner = ctk.CTkFrame(dl_card, fg_color="transparent")
        dl_inner.pack(fill="x", padx=12, pady=8)

        name_short = filename.split("/")[-1]
        ctk.CTkLabel(dl_inner, text=name_short,
                     font=ctk.CTkFont("Consolas", 11, "bold"),
                     text_color=C["text"], anchor="w",
                     wraplength=400).pack(anchor="w")

        ctk.CTkLabel(dl_inner,
                     text=f"{repo_id}  ·  {human_size(file_size)}",
                     font=ctk.CTkFont("Consolas", 10),
                     text_color=C["sub"], anchor="w").pack(anchor="w")

        progress_bar = ctk.CTkProgressBar(dl_inner, height=8,
                                           fg_color=C["input"],
                                           progress_color=C["accent"],
                                           corner_radius=4)
        progress_bar.pack(fill="x", pady=(6, 2))
        progress_bar.set(0)

        status_lbl = ctk.CTkLabel(dl_inner, text="Iniciando...",
                                   font=ctk.CTkFont("Consolas", 10),
                                   text_color=C["sub"])
        status_lbl.pack(anchor="w")

        btn_row = ctk.CTkFrame(dl_inner, fg_color="transparent")
        btn_row.pack(fill="x", pady=(4, 0))

        task = DownloadTask(
            repo_id=repo_id,
            filename=filename,
            dest_path=dest_path,
            total_size=file_size,
            progress_bar=progress_bar,
            status_label=status_lbl,
            card=dl_card,
            on_done=lambda f=dest_path: self._download_done(f),
            on_fail=lambda e, k=dest_path: self._download_failed(e, k),
        )

        # Botón cancelar
        ctk.CTkButton(btn_row, text="✕ Cancelar", width=90, height=24,
                      fg_color="transparent", hover_color=C["card"],
                      text_color=C["red"],
                      font=ctk.CTkFont("Consolas", 10),
                      command=task.cancel).pack(side="right")

        self._downloads[dest_path] = task
        task.start()

        # Actualizar UI cada segundo
        self._poll_download(task)

    def _poll_download(self, task):
        if task.done:
            return
        try:
            if task.total_size and task.downloaded > 0:
                pct = task.downloaded / task.total_size
                task.progress_bar.set(min(pct, 1.0))
                speed = task.speed_bps
                eta   = task.eta_seconds
                task.status_label.configure(
                    text=f"{human_size(task.downloaded)} / {human_size(task.total_size)}"
                         f"  ·  {human_size(speed)}/s"
                         f"  ·  ETA {int(eta)}s" if eta < 9999 else
                         f"{human_size(task.downloaded)} / {human_size(task.total_size)}"
                         f"  ·  {human_size(speed)}/s"
                )
        except Exception:
            pass
        self.after(1000, lambda: self._poll_download(task))

    def _download_done(self, dest_path):
        task = self._downloads.get(dest_path)
        if task:
            task.progress_bar.set(1.0)
            task.progress_bar.configure(progress_color=self.C["green"])
            task.status_label.configure(text="✓ Descarga completa", text_color=self.C["green"])
            task.done = True
        # Notificar
        fname = os.path.basename(dest_path)
        messagebox.showinfo("Descarga completa", f"✓ {fname}\nGuardado en: {dest_path}")

    def _download_failed(self, error, dest_path):
        task = self._downloads.get(dest_path)
        if task:
            task.status_label.configure(text=f"✗ Error: {error}", text_color=self.C["red"])
            task.progress_bar.configure(progress_color=self.C["red"])
            task.done = True

    # ── Utilidades ────────────────────────────────────────────────────────

    def _pick_dest_dir(self):
        d = filedialog.askdirectory(title="Selecciona carpeta de destino")
        if d:
            self.dest_var.set(d)

    def _open_hf(self, repo_id):
        import webbrowser
        webbrowser.open(HF_MODEL_PAGE.format(repo_id=repo_id))


# ══════════════════════════════════════════════════════════════════════════════
#  TAREA DE DESCARGA (streaming con progreso)
# ══════════════════════════════════════════════════════════════════════════════

class DownloadTask:
    """Descarga en background con seguimiento de progreso y cancelación."""

    CHUNK = 1024 * 256   # 256 KB por chunk

    def __init__(self, repo_id, filename, dest_path, total_size,
                 progress_bar, status_label, card, on_done, on_fail):
        self.repo_id      = repo_id
        self.filename     = filename
        self.dest_path    = dest_path
        self.total_size   = total_size
        self.progress_bar = progress_bar
        self.status_label = status_label
        self.card         = card
        self.on_done      = on_done
        self.on_fail      = on_fail

        self.downloaded   = 0
        self.speed_bps    = 0
        self.eta_seconds  = 9999
        self.done         = False
        self._cancelled   = False
        self._thread      = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self):
        self._cancelled = True
        self.done = True
        try:
            self.status_label.configure(text="Cancelado", text_color="#94a3b8")
        except Exception:
            pass
        # Borrar archivo parcial
        try:
            if os.path.isfile(self.dest_path):
                os.remove(self.dest_path)
        except Exception:
            pass

    def _run(self):
        url = HF_DOWNLOAD_URL.format(repo_id=self.repo_id, filename=self.filename)
        headers = {
            "User-Agent": "LlamaStation/2.3",
        }
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=30)
            resp.raise_for_status()

            # Tamaño real del servidor (puede diferir del metadata HF)
            content_length = resp.headers.get("Content-Length")
            if content_length:
                self.total_size = int(content_length)

            t_start = time.time()
            with open(self.dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=self.CHUNK):
                    if self._cancelled:
                        return
                    if chunk:
                        f.write(chunk)
                        self.downloaded += len(chunk)
                        elapsed = time.time() - t_start
                        if elapsed > 0:
                            self.speed_bps = self.downloaded / elapsed
                            if self.total_size and self.speed_bps > 0:
                                remaining = self.total_size - self.downloaded
                                self.eta_seconds = remaining / self.speed_bps

            if not self._cancelled:
                self.done = True
                self.on_done()

        except Exception as e:
            if not self._cancelled:
                self.done = True
                self.on_fail(str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  INSTRUCCIONES DE INTEGRACIÓN EN llama_gui.py
# ══════════════════════════════════════════════════════════════════════════════
"""
CÓMO INTEGRAR EN llama_gui.py:
================================

1. Al principio del archivo, añade:
   from llamaforge_downloader import ModelDownloaderFrame

2. En _build_tabs(), añade el nuevo frame:
   self.frames = {
       "Chat":        self._build_chat(self.main),
       "Servidor":    self._build_server(self.main),
       "Logs":        self._build_logs(self.main),
       "Info modelo": self._build_info(self.main),
       "Descargar":   self._build_downloader(self.main),   # ← NUEVO
   }

3. Añade el método _build_downloader():
   def _build_downloader(self, parent):
       models_dir = self.settings.get("models_dir", str(Path.home() / "models"))
       frame = ModelDownloaderFrame(parent, colors=C, models_dir=models_dir)
       return frame

4. En _build_sidebar(), en el bucle de botones de nav, añade:
   ("🌐", "Descargar", self._show_download),

5. Añade el método:
   def _show_download(self): self._show_frame("Descargar")

6. Opcional: cuando el usuario descarga un modelo, puedes escuchar el evento
   y actualizar el campo de modelo automáticamente. El on_done del DownloadTask
   recibe la ruta del archivo descargado.
"""
