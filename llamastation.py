"""
LlamaStation v2 - GUI para llama.cpp
Estilo LM Studio: modal de carga con todos los parámetros, guardado por modelo.
Requiere: pip install customtkinter requests
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess, threading, requests, json, os, sys, time, shutil, zipfile, tempfile, re, base64

_DND_AVAILABLE = False  # Desactivado: incompatible con customtkinter sin ventana extra
# try:
#     from tkinterdnd2 import TkinterDnD, DND_FILES
#     _DND_AVAILABLE = True
# except ImportError:
#     _DND_AVAILABLE = False

def _setup_dnd(widget, callback):
    """
    Registra drag & drop en un widget CTk.
    En Windows CTk usa un canvas interno — hay que registrar en el widget real.
    """
    if not _DND_AVAILABLE:
        return
    try:
        # CTkTextbox tiene ._textbox como widget tk real
        target = getattr(widget, "_textbox", widget)
        target.drop_target_register(DND_FILES)
        target.dnd_bind("<<Drop>>", callback)
    except Exception:
        pass
from pathlib import Path
from datetime import datetime
from llamastation_i18n import T, set_lang, get_lang
try:
    from llamastation_voice import VoiceMixin
    _VOICE_OK = True
except ImportError:
    class VoiceMixin: pass
    _VOICE_OK = False

# Flag para ocultar ventanas de consola en Windows al lanzar subprocesos
_NOWIN = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}

ctk.set_default_color_theme("blue")

# ── Adaptación a resolución de pantalla ──────────────────────────────────────
def _get_screen_scale():
    """
    Devuelve un factor de escala basado en la resolución real del monitor.
    - 1080p o menos  → 0.85  (pantallas pequeñas / scaling Windows 100–125%)
    - 1440p           → 1.0   (diseño base)
    - 4K+             → 1.2
    Usa un root temporal para no interferir con la app real.
    """
    try:
        _tmp = tk.Tk()
        _tmp.withdraw()
        sw = _tmp.winfo_screenwidth()
        sh = _tmp.winfo_screenheight()
        _tmp.destroy()
    except Exception:
        return 1.0
    if sh <= 1080:
        return 0.85
    elif sh <= 1440:
        return 1.0
    else:
        return 1.2

def _scale(value: int, factor: float = None) -> int:
    """Escala un valor entero de píxeles según el factor de pantalla."""
    if factor is None:
        factor = _SCREEN_SCALE
    return max(1, int(round(value * factor)))

_SCREEN_SCALE = _get_screen_scale()

THEMES = {
    "dark": {
        "bg":      "#0f0f13",
        "panel":   "#16161e",
        "card":    "#1e1e2a",
        "card2":   "#252533",
        "border":  "#2a2a3a",
        "accent":  "#7c6af7",
        "accent2": "#a78bfa",
        "green":   "#4ade80",
        "red":     "#f87171",
        "yellow":  "#fbbf24",
        "text":    "#e2e8f0",
        "sub":     "#94a3b8",
        "dim":     "#4a5568",
        "input":   "#1a1a26",
        "ctk_mode": "dark",
    },
    "light": {
        "bg":      "#f5f4f0",
        "panel":   "#ebe9e3",
        "card":    "#ffffff",
        "card2":   "#f0eeea",
        "border":  "#d8d5cc",
        "accent":  "#6457e0",
        "accent2": "#7c6af7",
        "green":   "#16a34a",
        "red":     "#dc2626",
        "yellow":  "#b45309",
        "text":    "#1c1917",
        "sub":     "#57534e",
        "dim":     "#a8a29e",
        "input":   "#fafaf9",
        "ctk_mode": "light",
    },
}

# Se actualiza al iniciar segun settings
C = dict(THEMES["dark"])

PROFILES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llamastation_profiles.json")
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llamastation_settings.json")

APP_VERSION = "v0.9"

DEFAULT_PROFILE = {
    "gpu_layers": -1, "threads": 8, "threads_batch": 8,
    "ctx_size": 4096, "batch_size": 512, "ubatch_size": 512, "max_concurrent": 1,
    "temperature": 0.7, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
    "repeat_penalty": 1.1, "repeat_last_n": 64, "max_tokens": 2048, "seed": -1,
    "flash_attn": True, "mmap": False, "mlock": False, "cont_batching": True,
    "kv_cache_offload": True, "keep_in_memory": False, "embeddings": False,
    "kv_type": "f16", "kv_type_v": "f16",
    "rope_freq_base": 0.0, "rope_freq_scale": 0.0,
    "extra_args": "", "system_prompt": "", "mmproj": "", "mmproj_disable": False,
    "split_mode": "layer", "main_gpu": 0, "tensor_split": "", "draft_model": "",
    "draft_spec_type": "draft-simple",
    "mtp_enabled": False, "mtp_draft_n_max": 6,
}

BACKENDS = {
    "⚡ Oficial  (llama.cpp)": r"C:\llama.cpp\llama-server.exe",
    "🔬 TurboQuant  (TheTom fork)": r"C:\llama-turboquant\llama-server.exe",
    "🚀 MTP  (llama.cpp + PR#22673)": r"C:\llama-mtp\llama-server.exe",
    "🐝 BeeLlama  (DFlash + TurboQuant)": r"C:\llama-bee\llama-server.exe",
    "⚛️ AtomicChat  (TurboQuant + MTP)": r"C:\llama-atomic\llama-server.exe",
}


def find_llama_server():
    # Prefer AtomicChat > BeeLlama > TurboQuant > MTP > official
    for c in [r"C:\llama-atomic\llama-server.exe",
              r"C:\llama-bee\llama-server.exe",
              r"C:\llama-turboquant\llama-server.exe",
              r"C:\llama-mtp\llama-server.exe",
              r"C:\llama.cpp\llama-server.exe",
              "llama-server", "llama-server.exe",
              r"C:\llama.cpp\build\bin\Release\llama-server.exe"]:
        f = shutil.which(c)
        if f: return f
        if os.path.isfile(c): return c
    return ""

def load_profiles():
    if os.path.isfile(PROFILES_FILE):
        try:
            with open(PROFILES_FILE) as f: return json.load(f)
        except: pass
    return {}

def save_profiles(p):
    with open(PROFILES_FILE, "w") as f: json.dump(p, f, indent=2)

def load_settings():
    if os.path.isfile(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f: return json.load(f)
        except: pass
    return {"server_path": find_llama_server(), "port": "8080", "host": "127.0.0.1", "theme": "dark", "models_dir": str(Path.home() / "models")}

def save_settings(d):
    with open(SETTINGS_FILE, "w") as f: json.dump(d, f, indent=2)

def apply_theme(name):
    """Actualiza el dict global C y el modo de apariencia de customtkinter."""
    t = THEMES.get(name, THEMES["dark"])
    C.update(t)
    ctk.set_appearance_mode(t["ctk_mode"])


# ══════════════════════════════════════════════════════════════════════════
#  MODAL DE CARGA (estilo LM Studio)
# ══════════════════════════════════════════════════════════════════════════

class LoadModelDialog(ctk.CTkToplevel):
    def __init__(self, parent, model_path, profiles):
        super().__init__(parent)
        self.model_path = model_path
        self.profiles   = profiles
        self.result     = None
        self._vars      = {}

        key  = model_path or "__default__"
        saved = profiles.get(key, {})
        self.prof = {**DEFAULT_PROFILE, **saved}

        self.title("Configurar modelo")
        self.geometry(f"{_scale(700)}x{_scale(860)}")
        self.resizable(True, True)
        self.configure(fg_color=C["bg"])
        self.grab_set()
        self.focus_set()

        model_name = Path(model_path).name if model_path else "Sin modelo"
        self._build(model_name)
        self._load_vars()
        self._autodetect_mmproj()

    def _build(self, model_name):
        # Header
        hdr = ctk.CTkFrame(self, fg_color=C["panel"], corner_radius=0, height=64)
        hdr.pack(fill="x"); hdr.pack_propagate(False)

        ctk.CTkButton(hdr, text="←", width=36, height=36,
                       fg_color="transparent", hover_color=C["card"],
                       text_color=C["sub"], font=ctk.CTkFont("Consolas", 18),
                       command=self.destroy).pack(side="left", padx=12, pady=14)

        ctk.CTkLabel(hdr, text=model_name,
                     font=ctk.CTkFont("Consolas", 14, "bold"),
                     text_color=C["text"]).pack(side="left", padx=4)

        ctk.CTkLabel(hdr, text=" GGUF ",
                     font=ctk.CTkFont("Consolas", 10, "bold"),
                     fg_color=C["accent"], text_color="white",
                     corner_radius=4).pack(side="left", padx=8)

        # Scroll
        s = ctk.CTkScrollableFrame(self, fg_color=C["bg"], corner_radius=0)
        s.pack(fill="both", expand=True)

        # VRAM estimada
        mem = ctk.CTkFrame(s, fg_color=C["card"], corner_radius=10)
        mem.pack(fill="x", padx=20, pady=(16, 8))
        mi = ctk.CTkFrame(mem, fg_color="transparent")
        mi.pack(fill="x", padx=16, pady=10)
        ctk.CTkLabel(mi, text=T("modal_vram"),
                     font=ctk.CTkFont("Consolas", 11), text_color=C["sub"]).pack(side="left")
        self.mem_label = ctk.CTkLabel(mi, text=T("modal_vram_hint"),
                                       font=ctk.CTkFont("Consolas", 11),
                                       text_color=C["accent2"])
        self.mem_label.pack(side="right")

        # Secciones
        self._sec_hardware(s)
        self._sec_multigpu(s)
        self._sec_cpu_ram(s)
        self._sec_context(s)
        self._sec_sampling(s)
        self._sec_flags(s)
        self._sec_rope(s)
        self._sec_vision(s)
        self._sec_extra(s)

        # Footer
        footer = ctk.CTkFrame(self, fg_color=C["panel"], corner_radius=0, height=64)
        footer.pack(fill="x", side="bottom"); footer.pack_propagate(False)

        rem_var = tk.BooleanVar(value=True)
        self._vars["remember"] = rem_var
        ctk.CTkCheckBox(footer,
                         text=f"Recordar configuración para este modelo",
                         variable=rem_var,
                         fg_color=C["accent"], hover_color=C["accent2"],
                         font=ctk.CTkFont("Consolas", 11),
                         text_color=C["sub"]).pack(side="left", padx=20, pady=20)

        ctk.CTkButton(footer, text=T("modal_cancel"),
                       fg_color="transparent", hover_color=C["card"],
                       text_color=C["sub"], font=ctk.CTkFont("Consolas", 12),
                       width=100, height=38, command=self.destroy
                       ).pack(side="right", padx=8, pady=13)

        ctk.CTkButton(footer, text=T("modal_load"),
                       fg_color=C["accent"], hover_color="#6457e0",
                       font=ctk.CTkFont("Consolas", 13, "bold"),
                       width=160, height=38, command=self._confirm
                       ).pack(side="right", padx=(0, 8), pady=13)

    # ── Secciones ────────────────────────────────────────────────────────

    def _autodetect_mmproj(self):
        """
        Busca automáticamente un archivo mmproj en la misma carpeta que el modelo.
        Solo lo asigna si el usuario no tenía ya uno guardado.
        """
        if not self.model_path:
            return
        # Si el usuario desactivó mmproj explícitamente, no autodetectar
        if self._vars.get("mmproj_disable") and self._vars["mmproj_disable"].get():
            return
        # Si ya hay uno guardado en el perfil, no sobreescribir
        if self._vars.get("mmproj") and self._vars["mmproj"].get().strip():
            return
        model_dir = Path(self.model_path).parent
        # Buscar cualquier archivo que contenga "mmproj" y sea .gguf
        candidates = list(model_dir.glob("*mmproj*.gguf")) +                      list(model_dir.glob("*mmproj*"))
        candidates = [p for p in candidates if p.suffix in (".gguf", ".bin")]
        if candidates:
            mmproj_path = str(candidates[0])
            if "mmproj" in self._vars:
                self._vars["mmproj"].set(mmproj_path)
            # Mostrar badge verde en la sección vision
            self.after(100, lambda: self._show_mmproj_badge(mmproj_path))

    def _show_mmproj_badge(self, path):
        """Muestra un mensaje verde indicando que se detectó mmproj automáticamente."""
        try:
            # Buscar el label de la sección vision y actualizarlo
            fname = Path(path).name
            if hasattr(self, "_mmproj_status_label"):
                self._mmproj_status_label.configure(
                    text=f"✓ Detectado automáticamente: {fname}",
                    text_color=C["green"]
                )
        except Exception:
            pass

    def _sec_hardware(self, s):
        self._title(s, T("sec_hardware"))
        c = self._card(s)
        self._slider(c, T("sl_gpu_layers"),   "gpu_layers",    -1,    200, 1,    int)
        self._slider(c, T("sl_threads"),       "threads",        1,     64, 1,    int)
        self._slider(c, T("sl_threads_b"),     "threads_batch",  1,     64, 1,    int)
        self._slider(c, T("sl_max_conc"),      "max_concurrent", 1,     16, 1,    int)

    def _sec_multigpu(self, s):
        self._title(s, T("sec_multigpu"))
        c = self._card(s)
        ctk.CTkLabel(c,
            text=T("split_tip"),
            font=ctk.CTkFont("Consolas", 11), text_color=C["sub"],
            wraplength=_scale(560), justify="left"
        ).pack(anchor="w", padx=16, pady=(10, 4))

        sm_var = tk.StringVar(value="layer")
        self._vars["split_mode"] = sm_var
        sm_row = ctk.CTkFrame(c, fg_color="transparent")
        sm_row.pack(fill="x", padx=16, pady=(0, 6))
        for opt, tip in [("none","1 GPU"), ("layer","Layer split (rec.)"), ("row","Row split")]:
            ctk.CTkRadioButton(sm_row, text=tip, variable=sm_var, value=opt,
                               fg_color=C["accent"], hover_color=C["accent2"],
                               font=ctk.CTkFont("Consolas", 11),
                               text_color=C["text"]).pack(side="left", padx=(0, 14))

        # ── Selector de GPU principal (solo visible en modo 1 GPU) ──────────
        gpu_sel_frame = ctk.CTkFrame(c, fg_color=C["card2"], corner_radius=8)
        mg_var = tk.IntVar(value=0)
        self._vars["main_gpu"] = mg_var

        ctk.CTkLabel(gpu_sel_frame,
            text="GPU a usar:",
            font=ctk.CTkFont("Consolas", 11), text_color=C["sub"]
        ).pack(side="left", padx=(12, 8), pady=8)

        for idx, label in [(0, "GPU 0  (primera)"), (1, "GPU 1  (segunda)")]:
            ctk.CTkRadioButton(gpu_sel_frame, text=label, variable=mg_var, value=idx,
                               fg_color=C["accent"], hover_color=C["accent2"],
                               font=ctk.CTkFont("Consolas", 11),
                               text_color=C["text"]).pack(side="left", padx=(0, 14), pady=8)

        def _on_split_change(*_):
            if sm_var.get() == "none":
                gpu_sel_frame.pack(fill="x", padx=16, pady=(0, 8))
                ts_frame.pack_forget()
            else:
                gpu_sel_frame.pack_forget()
                ts_frame.pack(fill="x", padx=16, pady=(0, 12))

        sm_var.trace_add("write", _on_split_change)

        # ── Tensor split (oculto en modo 1 GPU) ────────────────────────────
        ts_frame = ctk.CTkFrame(c, fg_color="transparent")
        ctk.CTkLabel(ts_frame,
            text=T("tensor_tip"),
            font=ctk.CTkFont("Consolas", 11), text_color=C["sub"],
            wraplength=_scale(560), justify="left"
        ).pack(anchor="w", pady=(0, 4))
        ts_var = tk.StringVar(value="")
        self._vars["tensor_split"] = ts_var
        ctk.CTkEntry(ts_frame, textvariable=ts_var,
                      fg_color=C["input"], text_color=C["accent2"],
                      font=ctk.CTkFont("Consolas", 12),
                      placeholder_text=T("tensor_ph"),
                      height=32).pack(fill="x")

        # Estado inicial según perfil guardado
        if self.prof.get("split_mode", "layer") == "none":
            gpu_sel_frame.pack(fill="x", padx=16, pady=(0, 8))
        else:
            ts_frame.pack(fill="x", padx=16, pady=(0, 12))

        ctk.CTkFrame(c, height=4, fg_color="transparent").pack()

    def _sec_cpu_ram(self, s):
        self._title(s, T("sec_cpu"))
        c = self._card(s)
        ctk.CTkLabel(c,
            text=T("cpu_mode_tip"),
            font=ctk.CTkFont("Consolas", 11), text_color=C["sub"],
            wraplength=_scale(560), justify="left"
        ).pack(anchor="w", padx=16, pady=(10, 8))

        mode_var = tk.StringVar(value="gpu")
        self._vars["exec_mode"] = mode_var

        mode_row = ctk.CTkFrame(c, fg_color="transparent")
        mode_row.pack(fill="x", padx=16, pady=(0, 8))

        for val, label, tip in [
            ("gpu",    "GPU (CUDA)",     "Todo en GPU — configuración por defecto"),
            ("cpu",    "Solo CPU + RAM", "ngl=0, usa RAM, va en cualquier PC"),
            ("hybrid", "Híbrido",        "Tú controlas cuántas capas van a GPU con el slider de arriba"),
            ("vulkan", "Vulkan — iGPU/NPU (experimental)", "AMD, Intel Arc, iGPU Intel UHD, AMD Radeon integrada, NPU — sin CUDA"),
        ]:
            ctk.CTkRadioButton(mode_row, text=label, variable=mode_var, value=val,
                               fg_color=C["accent"], hover_color=C["accent2"],
                               font=ctk.CTkFont("Consolas", 11),
                               text_color=C["text"],
                               command=lambda v=val: self._on_exec_mode(v)
                               ).pack(side="left", padx=(0, 14))

        # Sugerencia de threads para CPU
        ctk.CTkLabel(c,
            text=T("cpu_tip2"),
            font=ctk.CTkFont("Consolas", 10), text_color=C["dim"],
            wraplength=_scale(560), justify="left"
        ).pack(anchor="w", padx=16, pady=(0, 12))

    def _on_exec_mode(self, mode):
        """Ajusta gpu_layers automáticamente según el modo elegido."""
        if mode == "cpu":
            # ngl=0 → todo en CPU
            if "gpu_layers" in self._vars:
                self._vars["gpu_layers"].set(0)
        elif mode == "gpu":
            # ngl=-1 → todo en GPU
            if "gpu_layers" in self._vars:
                self._vars["gpu_layers"].set(-1)
        elif mode == "vulkan":
            # Vulkan también usa ngl=-1, llama.cpp lo gestiona solo si se lanzó con Vulkan
            if "gpu_layers" in self._vars:
                self._vars["gpu_layers"].set(-1)
        # hybrid: el usuario controla el slider manualmente

    def _sec_context(self, s):
        self._title(s, T("sec_context"))
        c = self._card(s)
        self._slider(c, T("sl_ctx"),      "ctx_size",   512, 1048576, 512, int)
        self._slider(c, T("sl_batch"),    "batch_size",  64,   4096,  64, int)
        self._slider(c, T("sl_ubatch"),   "ubatch_size", 64,   4096,  64, int)
        self._slider(c, T("sl_max_tok"),  "max_tokens",  -1, 131072,  64, int)

    def _sec_sampling(self, s):
        self._title(s, T("sec_sampling"))
        c = self._card(s)
        self._slider(c, T("sl_temp"),     "temperature",    0.0, 2.0,  0.01, float)
        self._slider(c, T("sl_topk"),     "top_k",          0,   200,  1,    int)
        self._slider(c, T("sl_topp"),     "top_p",          0.0, 1.0,  0.01, float)
        self._slider(c, T("sl_minp"),     "min_p",          0.0, 1.0,  0.01, float)
        self._slider(c, T("sl_rep_pen"),  "repeat_penalty", 1.0, 2.0,  0.01, float)
        self._slider(c, T("sl_rep_n"),    "repeat_last_n",  0,   512,  1,    int)
        self._slider(c, T("sl_seed"),     "seed",          -1, 99999,  1,    int)

        ctk.CTkLabel(c, text=T("sys_prompt"),
                     font=ctk.CTkFont("Consolas", 12), text_color=C["text"]
                     ).pack(anchor="w", padx=16, pady=(10, 2))
        sp = tk.StringVar()
        self._vars["system_prompt"] = sp
        ctk.CTkEntry(c, textvariable=sp,
                      fg_color=C["input"], text_color=C["text"],
                      font=ctk.CTkFont("Consolas", 12),
                      placeholder_text=T("sys_prompt_ph"),
                      height=32).pack(fill="x", padx=16, pady=(0, 12))

    def _sec_flags(self, s):
        self._title(s, T("sec_flags"))
        c = self._card(s)
        for key, lbl_k, tip_k in [
            ("flash_attn",       "fl_flash",    "fl_flash_tip"),
            ("mmap",             "fl_mmap",     "fl_mmap_tip"),
            ("mlock",            "fl_mlock",    "fl_mlock_tip"),
            ("cont_batching",    "fl_contbatch","fl_contbatch_tip"),
            ("kv_cache_offload", "fl_kvoff",    "fl_kvoff_tip"),
            ("keep_in_memory",   "fl_keepmem",  "fl_keepmem_tip"),
            ("embeddings",       "fl_embed",    "fl_embed_tip"),
        ]:
            self._toggle(c, T(lbl_k), key, T(tip_k))

        ctk.CTkLabel(c, text=T("kv_label"),
                     font=ctk.CTkFont("Consolas", 11), text_color=C["sub"],
                     wraplength=_scale(560), justify="left"
                     ).pack(anchor="w", padx=16, pady=(10, 2))

        for cache_key, cache_lbl, tip_txt in [
            ("kv_type",   "Cache-K:",   "K-cache: usa q8_0 para máxima calidad, turbo3/4 para máximo ahorro VRAM"),
            ("kv_type_v", "Cache-V:",   "V-cache: turbo3 tiene mínima pérdida de calidad — ideal para comprimir"),
        ]:
            ctk.CTkLabel(c, text=f"  {cache_lbl}  {tip_txt}",
                         font=ctk.CTkFont("Consolas", 10), text_color=C["dim"],
                         wraplength=_scale(560), justify="left"
                         ).pack(anchor="w", padx=16, pady=(6, 0))
            kv_var = tk.StringVar(value="f16")
            self._vars[cache_key] = kv_var
            # Fila 1: clásicos
            kv_row1 = ctk.CTkFrame(c, fg_color="transparent")
            kv_row1.pack(fill="x", padx=16, pady=(2, 0))
            for opt in ["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"]:
                ctk.CTkRadioButton(kv_row1, text=opt, variable=kv_var, value=opt,
                                    fg_color=C["accent"], hover_color=C["accent2"],
                                    font=ctk.CTkFont("Consolas", 11),
                                    text_color=C["text"]).pack(side="left", padx=(0, 8))
            # Fila 2: TurboQuant
            kv_row2 = ctk.CTkFrame(c, fg_color="transparent")
            kv_row2.pack(fill="x", padx=16, pady=(2, 6))
            ctk.CTkLabel(kv_row2, text="TurboQuant:",
                         font=ctk.CTkFont("Consolas", 10), text_color=C["yellow"]).pack(side="left")
            for opt, tip in [("turbo2","2bit"), ("turbo3","3bit (rec.V)"), ("turbo4","4bit")]:
                ctk.CTkRadioButton(kv_row2, text=tip, variable=kv_var, value=opt,
                                    fg_color=C["yellow"], hover_color="#d97706",
                                    font=ctk.CTkFont("Consolas", 11),
                                    text_color=C["yellow"]).pack(side="left", padx=(8, 0))

    def _sec_rope(self, s):
        self._title(s, T("sec_rope"))
        c = self._card(s)
        ctk.CTkLabel(c, text=T("kv_zero"),
                     font=ctk.CTkFont("Consolas", 11), text_color=C["sub"]
                     ).pack(anchor="w", padx=16, pady=(8, 4))
        self._slider(c, T("sl_rope_base"),  "rope_freq_base",  0.0, 1000000.0, 1000.0, float)
        self._slider(c, T("sl_rope_scale"), "rope_freq_scale", 0.0, 4.0, 0.01, float)

    def _sec_vision(self, s):
        self._title(s, T("sec_vision"))
        c = self._card(s)
        ctk.CTkLabel(c,
                     text=T("mmproj_tip"),
                     font=ctk.CTkFont("Consolas", 11), text_color=C["sub"],
                     wraplength=_scale(560), justify="left"
                     ).pack(anchor="w", padx=16, pady=(8, 4))

        self._mmproj_status_label = ctk.CTkLabel(c,
            text=T("mmproj_none"),
            font=ctk.CTkFont("Consolas", 10), text_color=C["dim"],
            wraplength=_scale(560), justify="left"
        )
        self._mmproj_status_label.pack(anchor="w", padx=16, pady=(0, 6))

        mm_var = tk.StringVar()
        self._vars["mmproj"] = mm_var
        row = ctk.CTkFrame(c, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkEntry(row, textvariable=mm_var,
                      fg_color=C["input"], text_color=C["text"],
                      font=ctk.CTkFont("Consolas", 12),
                      placeholder_text=T("mmproj_ph"),
                      height=34).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(row, text=T("browse"), width=90, height=34,
                       fg_color=C["card2"], hover_color=C["border"],
                       font=ctk.CTkFont("Consolas", 12),
                       command=lambda: mm_var.set(
                           filedialog.askopenfilename(
                               title=T("mmproj_select"),
                               filetypes=[("GGUF", "*.gguf"), ("Todos", "*.*")]
                           ) or mm_var.get()
                       )).pack(side="left", padx=(8, 0))
        ctk.CTkButton(row, text="✕", width=32, height=34,
                       fg_color="transparent", hover_color=C["card2"],
                       text_color=C["dim"], font=ctk.CTkFont("Consolas", 13),
                       command=lambda: mm_var.set("")
                       ).pack(side="left", padx=(4, 0))
        # Switch: deshabilitar mmproj
        dis_row = ctk.CTkFrame(c, fg_color="transparent")
        dis_row.pack(fill="x", padx=16, pady=(0, 10))
        ctk.CTkLabel(dis_row, text=T("mmproj_disable"),
                     font=ctk.CTkFont("Consolas", 11), text_color=C["sub"]
                     ).pack(side="left")
        _mmproj_dis_var = tk.BooleanVar(value=False)
        self._vars["mmproj_disable"] = _mmproj_dis_var
        ctk.CTkSwitch(dis_row, variable=_mmproj_dis_var, text="",
                       fg_color=C["input"], progress_color=C["accent"],
                       button_color=C["accent2"]).pack(side="right")

    def _sec_extra(self, s):
        self._title(s, T("sec_extra"))
        c = self._card(s)
        ctk.CTkLabel(c, text=T("extra_tip"),
                     font=ctk.CTkFont("Consolas", 11), text_color=C["sub"]
                     ).pack(anchor="w", padx=16, pady=(8, 4))
        ev = tk.StringVar()
        self._vars["extra_args"] = ev
        ctk.CTkEntry(c, textvariable=ev,
                      fg_color=C["input"], text_color=C["text"],
                      font=ctk.CTkFont("Consolas", 12),
                      placeholder_text=T("extra_ph"),
                      height=36).pack(fill="x", padx=16, pady=(0, 12))

        # Draft Model (especulación clásica con modelo pequeño)
        ctk.CTkLabel(c, text="Draft Model  (opcional — modelo drafter pequeño)",
                     font=ctk.CTkFont("Consolas", 11), text_color=C["yellow"]
                     ).pack(anchor="w", padx=16, pady=(0, 4))
        ctk.CTkLabel(c, text="Ruta al GGUF del modelo drafter. Usa draft-simple para backends estándar (rec.), dflash solo para BeeLlama.",
                     font=ctk.CTkFont("Consolas", 10), text_color=C["dim"],
                     wraplength=_scale(560), justify="left"
                     ).pack(anchor="w", padx=16, pady=(0, 4))

        # Selector de tipo de especulación
        spec_type_var = tk.StringVar(value="draft-simple")
        self._vars["draft_spec_type"] = spec_type_var
        spec_row = ctk.CTkFrame(c, fg_color="transparent")
        spec_row.pack(fill="x", padx=16, pady=(0, 6))
        ctk.CTkLabel(spec_row, text="Tipo:",
                     font=ctk.CTkFont("Consolas", 11), text_color=C["sub"]
                     ).pack(side="left", padx=(0, 8))
        for val, tip, col in [
            ("draft-simple", "draft-simple  (Oficial · TurboQuant · MTP)", C["accent"]),
            ("dflash",       "dflash  (solo BeeLlama)",                    C["yellow"]),
        ]:
            ctk.CTkRadioButton(spec_row, text=tip, variable=spec_type_var, value=val,
                               fg_color=col, hover_color=C["accent2"],
                               font=ctk.CTkFont("Consolas", 11),
                               text_color=C["text"]).pack(side="left", padx=(0, 14))

        draft_var = tk.StringVar()
        self._vars["draft_model"] = draft_var
        draft_row = ctk.CTkFrame(c, fg_color="transparent")
        draft_row.pack(fill="x", padx=16, pady=(0, 16))
        ctk.CTkEntry(draft_row, textvariable=draft_var,
                      fg_color=C["input"], text_color=C["text"],
                      font=ctk.CTkFont("Consolas", 12),
                      placeholder_text="p.ej. C:\\modelos_llamaforge\\Qwen3.5-0.8B-Q8_0.gguf",
                      height=34).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(draft_row, text="Browse", width=90, height=34,
                       fg_color=C["card2"], hover_color=C["border"],
                       font=ctk.CTkFont("Consolas", 12),
                       command=lambda: draft_var.set(
                           filedialog.askopenfilename(
                               title="Seleccionar GGUF del drafter",
                               filetypes=[("GGUF", "*.gguf"), ("Todos", "*.*")]
                           ) or draft_var.get()
                       )).pack(side="left", padx=(8, 0))
        ctk.CTkButton(draft_row, text="✕", width=32, height=34,
                       fg_color="transparent", hover_color=C["card2"],
                       text_color=C["dim"], font=ctk.CTkFont("Consolas", 13),
                       command=lambda: draft_var.set("")
                       ).pack(side="left", padx=(4, 0))

        # ── MTP (Multi-Token Prediction) ─────────────────────────────────
        ctk.CTkFrame(c, height=1, fg_color=C["border"]).pack(fill="x", padx=16, pady=(10, 0))
        mtp_hdr = ctk.CTkFrame(c, fg_color="transparent")
        mtp_hdr.pack(fill="x", padx=16, pady=(8, 2))
        ctk.CTkLabel(mtp_hdr, text="🚀 MTP — Multi-Token Prediction",
                     font=ctk.CTkFont("Consolas", 11, "bold"),
                     text_color=C["accent2"]).pack(side="left")
        ctk.CTkLabel(mtp_hdr,
                     text="  ✓ Oficial · MTP · AtomicChat",
                     font=ctk.CTkFont("Consolas", 9),
                     text_color=C["green"]).pack(side="left", padx=(8, 0))
        ctk.CTkLabel(mtp_hdr,
                     text="  ✗ TurboQuant · BeeLlama",
                     font=ctk.CTkFont("Consolas", 9),
                     text_color=C["red"]).pack(side="left", padx=(4, 0))
        ctk.CTkLabel(c,
                     text="Requiere GGUF MTP (p.ej. Qwen3.6-27B-MTP-UD-Q4_K_XL de Unsloth).\n"
                          "Fuerza -np 1 automáticamente. Incompatible con --mmproj.",
                     font=ctk.CTkFont("Consolas", 10), text_color=C["dim"],
                     wraplength=_scale(560), justify="left"
                     ).pack(anchor="w", padx=16, pady=(0, 6))
        mtp_sw_row = ctk.CTkFrame(c, fg_color="transparent")
        mtp_sw_row.pack(fill="x", padx=16, pady=(0, 4))
        mtp_en_var = tk.BooleanVar(value=False)
        self._vars["mtp_enabled"] = mtp_en_var
        ctk.CTkSwitch(mtp_sw_row, variable=mtp_en_var, text=T("mtp_activate"),
                       fg_color=C["input"], progress_color=C["accent2"],
                       button_color=C["accent"],
                       font=ctk.CTkFont("Consolas", 11),
                       text_color=C["text"]).pack(side="left")
        self._slider(c, T("mtp_draft_n_max"),
                     "mtp_draft_n_max", 1, 12, 1, int)
        ctk.CTkLabel(c,
                     text=T("mtp_manual_hint"),
                     font=ctk.CTkFont("Consolas", 10), text_color=C["dim"],
                     wraplength=_scale(560), justify="left"
                     ).pack(anchor="w", padx=16, pady=(8, 0))
        ctk.CTkFrame(c, height=8, fg_color="transparent").pack()

    # ── Widget helpers ────────────────────────────────────────────────────

    def _title(self, p, t):
        ctk.CTkLabel(p, text=t, font=ctk.CTkFont("Consolas", 12, "bold"),
                     text_color=C["accent2"]).pack(anchor="w", padx=24, pady=(16, 4))

    def _card(self, p):
        f = ctk.CTkFrame(p, fg_color=C["card"], corner_radius=10)
        f.pack(fill="x", padx=20, pady=(0, 4))
        return f

    def _slider(self, parent, label, key, mn, mx, step, typ):
        var_n = tk.DoubleVar() if typ == float else tk.IntVar()
        var_t = tk.StringVar()
        self._vars[key] = var_n

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(8, 0))
        ctk.CTkLabel(row, text=label, font=ctk.CTkFont("Consolas", 12),
                     text_color=C["text"]).pack(side="left")

        entry = ctk.CTkEntry(row, textvariable=var_t,
                              fg_color=C["input"], text_color=C["accent2"],
                              font=ctk.CTkFont("Consolas", 12, "bold"),
                              width=96, height=26, justify="right")
        entry.pack(side="right")

        steps = max(1, int((mx - mn) / step)) if step > 0 else 1000
        sl = ctk.CTkSlider(parent, from_=mn, to=mx, number_of_steps=steps,
                            variable=var_n,
                            fg_color=C["input"], progress_color=C["accent"],
                            button_color=C["accent2"], button_hover_color=C["accent"])
        sl.pack(fill="x", padx=16, pady=(2, 8))

        updating = [False]

        def on_slide(*_):
            if updating[0]: return
            updating[0] = True
            v = float(var_n.get())
            if step >= 1:
                v = typ(round(v / step) * step)
            else:
                v = typ(round(v / step) * step)
            fmt = f"{v:.2f}" if typ == float else str(int(v))
            var_t.set(fmt)
            updating[0] = False

        def on_entry(*_):
            if updating[0]: return
            try:
                v = typ(var_t.get())
                v = max(mn, min(mx, v))
                updating[0] = True
                var_n.set(v)
                updating[0] = False
            except: pass

        var_n.trace_add("write", on_slide)
        var_t.trace_add("write", on_entry)

    def _toggle(self, parent, label, key, tip=""):
        var = tk.BooleanVar()
        self._vars[key] = var
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=5)
        ctk.CTkLabel(row, text=label, font=ctk.CTkFont("Consolas", 12),
                     text_color=C["text"]).pack(side="left")
        if tip:
            ctk.CTkLabel(row, text=f"  {tip}", font=ctk.CTkFont("Consolas", 10),
                         text_color=C["dim"]).pack(side="left")
        ctk.CTkSwitch(row, variable=var, text="",
                       fg_color=C["input"], progress_color=C["accent"],
                       button_color=C["accent2"]).pack(side="right")

    # ── Confirm ───────────────────────────────────────────────────────────

    def _load_vars(self):
        for k, v in self.prof.items():
            if k in self._vars:
                try: self._vars[k].set(v)
                except: pass

    def _confirm(self):
        prof = {}
        for k, v in self._vars.items():
            if k == "remember": continue
            try: prof[k] = v.get()
            except: pass
        if self._vars.get("remember") and self._vars["remember"].get():
            key = self.model_path or "__default__"
            self.profiles[key] = prof
            save_profiles(self.profiles)
        self.result = prof
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════
#  MODAL DE ACTUALIZACIÓN DE llama.cpp (solo backend oficial)
# ══════════════════════════════════════════════════════════════════════════

LLAMA_CPP_OFFICIAL_DIR   = r"C:\llama.cpp"
LLAMA_CPP_TURBOQUANT_DIR = r"C:\llama-turboquant"
LLAMA_CPP_MTP_DIR        = r"C:\llama-mtp"
LLAMA_CPP_BEE_DIR        = r"C:\llama-bee"
LLAMA_CPP_ATOMIC_DIR     = r"C:\llama-atomic"
GITHUB_API_LATEST        = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

def _tq_find_asset(assets, cuda_mm, cuda_maj):
    """Busca el zip de TurboQuant en los releases de TheTom.
    El asset se llama turboquant-plus-tqp-vX.Y.Z-windows-x64-cudaXX.X.zip
    Nunca coger assets del oficial llama.cpp (llama-bXXXX-...).
    """
    # Primero: buscar específicamente "turboquant" en el nombre
    for a in assets:
        n = a["name"].lower()
        if "turboquant" in n and n.endswith(".zip") and "windows" in n:
            return a
    # Fallback: cualquier zip que NO sea del oficial (no empiece por "llama-b")
    for a in assets:
        n = a["name"].lower()
        if n.endswith(".zip") and not n.startswith("llama-b") and not n.startswith("cudart-llama"):
            return a
    return None

BACKEND_META = {
    "⚡ Oficial  (llama.cpp)": {
        "label":    "llama.cpp oficial",
        "dir":      LLAMA_CPP_OFFICIAL_DIR,
        "api":      "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
        "asset_fn": None,
    },
    "🔬 TurboQuant  (TheTom fork)": {
        "label":    "TurboQuant (TheTom)",
        "dir":      LLAMA_CPP_TURBOQUANT_DIR,
        "api":      "https://api.github.com/repos/TheTom/llama-cpp-turboquant/releases/latest",
        "asset_fn": _tq_find_asset,
    },
    "🚀 MTP  (llama.cpp + PR#22673)": {
        "label":    "MTP (PR#22673)",
        "dir":      LLAMA_CPP_MTP_DIR,
        "api":      "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
        "asset_fn": None,  # Mismo updater que el oficial
    },
    "🐝 BeeLlama  (DFlash + TurboQuant)": {
        "label":    "BeeLlama (DFlash + TurboQuant)",
        "dir":      LLAMA_CPP_BEE_DIR,
        "api":      "https://api.github.com/repos/Anbeeld/beellama.cpp/releases/latest",
        "asset_fn": _tq_find_asset,
    },
    "⚛️ AtomicChat  (TurboQuant + MTP)": {
        "label":    "AtomicChat (TurboQuant + MTP)",
        "dir":      LLAMA_CPP_ATOMIC_DIR,
        "api":      "https://api.github.com/repos/AtomicBot-ai/atomic-llama-cpp-turboquant/releases/latest",
        "asset_fn": _tq_find_asset,
    },
}

def _extract_build_number(raw_output):
    """
    Extrae el número de build de la salida de llama-server --version.
    llama.cpp puede imprimir logs de CUDA antes de la versión, así que
    buscamos en todo el output. Formatos conocidos:
      - "build: 9006"          (versiones recientes)
      - "version: b9006 (...)" (versiones anteriores)
      - "llama.cpp build 9006"
    Devuelve int o None.
    """
    # "build: 9006" o "build 9006"
    m = re.search(r"build[:\s]+(\d{4,})", raw_output, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # "b9006" standalone
    m = re.search(r"\bb(\d{4,})\b", raw_output)
    if m:
        return int(m.group(1))
    return None

def _extract_version_label(raw_output):
    """Devuelve un string legible para mostrar en el ver_label."""
    # Buscar línea con "build" o "version" que no sea de CUDA/ggml
    for line in raw_output.splitlines():
        low = line.lower()
        if ("build" in low or "version" in low) and "cuda" not in low and "ggml" not in low and "found" not in low:
            return line.strip()[:60]
    # Fallback: primer número de build que encontremos
    n = _extract_build_number(raw_output)
    if n:
        return f"build {n}"
    return "versión desconocida"

def _detect_cuda_version():
    """Devuelve (major_minor, major) como strings, o ('12.4', '12') si no detecta."""
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10, **_NOWIN)
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", r.stdout + r.stderr)
        if m:
            return f"{m.group(1)}.{m.group(2)}", m.group(1)
    except Exception:
        pass
    return "12.4", "12"

def _find_best_asset(assets, cuda_mm, cuda_maj):
    """
    Replica la lógica de búsqueda del .ps1:
    1. Exacto: cuda-X.Y + win + x64
    2. Major: cuda-X.  + win + x64
    3. Cualquier cuda + win + x64
    Devuelve el dict del asset o None.
    """
    patterns = [
        lambda n: (re.search(r"^llama-b", n) and "win" in n and f"cuda-{cuda_mm}" in n and "x64" in n and n.endswith(".zip")),
        lambda n: (re.search(r"^llama-b", n) and "win" in n and f"cuda-{cuda_maj}." in n and "x64" in n and n.endswith(".zip")),
        lambda n: (re.search(r"^llama-b", n) and "win" in n and "cuda" in n and "x64" in n and n.endswith(".zip")),
    ]
    for pat in patterns:
        for a in assets:
            if pat(a["name"]):
                return a
    return None

def _find_release_with_asset(api_latest_url, cuda_mm, cuda_maj, asset_fn=None, max_releases=5, log_fn=None):
    """
    Busca un asset compatible empezando por /releases/latest y, si esa release
    no tiene ningún build compatible (por ejemplo porque el CI aún no ha subido
    todos los assets), va probando las releases anteriores una a una.

    Devuelve (release_dict, asset_dict, cudart_dict_or_None) o (None, None, None)
    si tras revisar max_releases no se encuentra nada.
    log_fn, si se pasa, se llama con cada línea de log (pensado para volcar a la GUI).
    """
    def _log(msg):
        if log_fn:
            log_fn(msg)

    # /releases/latest -> /releases (lista paginada, más recientes primero)
    list_url = api_latest_url.rsplit("/latest", 1)[0]

    try:
        resp = requests.get(f"{list_url}?per_page={max_releases}",
                            headers={"User-Agent": "LlamaStation-Updater"},
                            timeout=20)
        resp.raise_for_status()
        releases = resp.json()
    except Exception as e:
        _log(f"✗ Error al listar releases: {e}")
        return None, None, None

    if not isinstance(releases, list) or not releases:
        return None, None, None

    for i, release in enumerate(releases):
        tag = release.get("tag_name", "desconocida")
        assets = release.get("assets", [])

        if asset_fn:
            asset  = asset_fn(assets, cuda_mm, cuda_maj)
            cudart = None
        else:
            asset  = _find_best_asset(assets, cuda_mm, cuda_maj)
            cudart = _find_cudart_asset(assets, cuda_mm, cuda_maj)

        if asset:
            if i > 0:
                _log(f"  ⚠ La release más reciente no tenía asset compatible — usando {tag} ({i} release(s) atrás)")
            return release, asset, cudart

        # Sin match en esta release: loguear qué assets sí había, para depurar
        nombres = [a.get("name", "?") for a in assets]
        _log(f"  ✗ {tag}: sin asset compatible. Assets disponibles: {', '.join(nombres) if nombres else '(ninguno)'}")

    return None, None, None

def _find_cudart_asset(assets, cuda_mm, cuda_maj):
    patterns = [
        lambda n: (n.startswith("cudart-llama") and "win" in n and f"cuda-{cuda_mm}" in n and "x64" in n and n.endswith(".zip")),
        lambda n: (n.startswith("cudart-llama") and "win" in n and f"cuda-{cuda_maj}." in n and "x64" in n and n.endswith(".zip")),
        lambda n: (n.startswith("cudart-llama") and "win" in n and "cuda" in n and "x64" in n and n.endswith(".zip")),
    ]
    for pat in patterns:
        for a in assets:
            if pat(a["name"]):
                return a
    return None


class UpdateDialog(ctk.CTkToplevel):
    """Modal de actualización — soporta cualquier backend definido en BACKEND_META."""

    def __init__(self, parent, backend_key=None):
        super().__init__(parent)
        if backend_key is None or backend_key not in BACKEND_META:
            backend_key = "⚡ Oficial  (llama.cpp)"
        self._meta        = BACKEND_META[backend_key]
        self._backend_key = backend_key

        self.title(f"Actualizar {self._meta['label']}")
        self.geometry(f"{_scale(620)}x{_scale(500)}")
        self.resizable(False, False)
        self.configure(fg_color=C["bg"])
        self.grab_set()
        self.focus_set()

        self._cancelled = False
        self._thread    = None

        self._build()
        self.after(200, self._start_check)

    def _build(self):
        # ── Header ────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color=C["panel"], corner_radius=0, height=56)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text=f"⬆  Actualizar {self._meta['label']}",
                     font=ctk.CTkFont("Consolas", 15, "bold"),
                     text_color=C["accent2"]).pack(side="left", padx=20, pady=14)
        ctk.CTkLabel(hdr, text=self._backend_key,
                     font=ctk.CTkFont("Consolas", 10),
                     text_color=C["dim"]).pack(side="right", padx=16)

        # ── Info cards ───────────────────────────────────────────────
        cards = ctk.CTkFrame(self, fg_color="transparent")
        cards.pack(fill="x", padx=20, pady=(16, 0))

        def _info_card(parent, label, attr):
            f = ctk.CTkFrame(parent, fg_color=C["card"], corner_radius=8)
            f.pack(side="left", fill="x", expand=True, padx=(0, 8))
            ctk.CTkLabel(f, text=label, font=ctk.CTkFont("Consolas", 9, "bold"),
                         text_color=C["sub"]).pack(anchor="w", padx=12, pady=(8, 2))
            lbl = ctk.CTkLabel(f, text="—", font=ctk.CTkFont("Consolas", 12),
                               text_color=C["text"])
            lbl.pack(anchor="w", padx=12, pady=(0, 8))
            setattr(self, attr, lbl)

        _info_card(cards, "VERSIÓN INSTALADA", "lbl_current")
        _info_card(cards, "ÚLTIMA RELEASE",    "lbl_latest")
        _info_card(cards, "CUDA DETECTADA",    "lbl_cuda")

        # ── Log interno ───────────────────────────────────────────────
        self.log_box = ctk.CTkTextbox(self, fg_color=C["panel"],
                                       text_color=C["sub"],
                                       font=ctk.CTkFont("Consolas", 11),
                                       wrap="word", state="disabled",
                                       corner_radius=8)
        self.log_box.pack(fill="both", expand=True, padx=20, pady=12)

        # ── Barra de progreso ────────────────────────────────────────
        self.progress = ctk.CTkProgressBar(self, fg_color=C["card"],
                                            progress_color=C["accent"],
                                            height=6, corner_radius=3)
        self.progress.pack(fill="x", padx=20, pady=(0, 4))
        self.progress.set(0)

        self.progress_label = ctk.CTkLabel(self, text="",
                                            font=ctk.CTkFont("Consolas", 10),
                                            text_color=C["sub"])
        self.progress_label.pack(anchor="w", padx=22, pady=(0, 8))

        # ── Footer ───────────────────────────────────────────────────
        footer = ctk.CTkFrame(self, fg_color=C["panel"], corner_radius=0, height=56)
        footer.pack(fill="x", side="bottom"); footer.pack_propagate(False)

        self.btn_cancel = ctk.CTkButton(footer, text="Cerrar",
                                         fg_color="transparent", hover_color=C["card"],
                                         text_color=C["sub"], font=ctk.CTkFont("Consolas", 12),
                                         width=100, height=36,
                                         command=self._on_cancel)
        self.btn_cancel.pack(side="right", padx=8, pady=10)

        self.btn_update = ctk.CTkButton(footer, text="Instalar actualización",
                                         fg_color=C["accent"], hover_color="#6457e0",
                                         font=ctk.CTkFont("Consolas", 13, "bold"),
                                         width=200, height=36, state="disabled",
                                         command=self._start_install)
        self.btn_update.pack(side="right", padx=(0, 8), pady=10)

    # ── Helpers de log ────────────────────────────────────────────────

    def _log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _set_progress(self, value, label=""):
        self.progress.set(value)
        self.progress_label.configure(text=label)

    # ── Paso 1: Comprobar versión ─────────────────────────────────────

    def _start_check(self):
        self._thread = threading.Thread(target=self._check_thread, daemon=True)
        self._thread.start()

    def _check_thread(self):
        self.after(0, lambda: self._log("→ Detectando CUDA..."))
        cuda_mm, cuda_maj = _detect_cuda_version()
        self.after(0, lambda: self.lbl_cuda.configure(text=cuda_mm))
        self.after(0, lambda: self._log(f"  CUDA: {cuda_mm}"))

        # Versión instalada
        exe = os.path.join(LLAMA_CPP_OFFICIAL_DIR, "llama-server.exe")
        current_ver = "no instalado"
        current_build = None
        if os.path.isfile(exe):
            try:
                r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=5, **_NOWIN)
                raw_ver = r.stdout + r.stderr
                current_build = _extract_build_number(raw_ver)
                current_ver = _extract_version_label(raw_ver)
            except Exception:
                current_ver = "instalado (ver desconocida)"
        self.after(0, lambda: self.lbl_current.configure(text=current_ver))
        self.after(0, lambda: self._log(f"  Instalado: {current_ver}"))

        # Consultar GitHub
        api_url = self._meta["api"]
        repo = "/".join(api_url.split("/")[4:6])
        self.after(0, lambda: self._log(f"→ Consultando GitHub ({repo})..."))
        self.after(0, lambda: self._set_progress(0.1, "Consultando GitHub..."))

        asset_fn = self._meta.get("asset_fn")

        # Busca en /releases/latest y, si no hay asset compatible (por ejemplo
        # porque el CI aún no ha terminado de subir los builds), cae hacia
        # releases anteriores automáticamente.
        release, asset, cudart = _find_release_with_asset(
            api_url, cuda_mm, cuda_maj, asset_fn=asset_fn,
            log_fn=lambda msg: self.after(0, lambda m=msg: self._log(m)),
        )

        if release is None:
            self.after(0, lambda: self._set_progress(0, "Error de conexión"))
            return

        tag = release.get("tag_name", "desconocida")
        self.after(0, lambda: self.lbl_latest.configure(text=tag))
        self.after(0, lambda: self._log(f"  Última release: {tag}"))

        if not asset:
            self.after(0, lambda: self._log("✗ No se encontró asset compatible para tu CUDA/Windows/x64 en las últimas releases"))
            self.after(0, lambda: self._set_progress(0, "Asset no encontrado"))
            return

        size_mb = round(asset["size"] / 1_048_576, 1)
        cudart_mb = round(cudart["size"] / 1_048_576, 1) if cudart else 0
        total_mb = size_mb + cudart_mb

        self.after(0, lambda: self._log(f"  Asset: {asset['name']}  ({size_mb} MB)"))
        if cudart:
            self.after(0, lambda: self._log(f"  Cudart: {cudart['name']}  ({cudart_mb} MB)"))
        else:
            self.after(0, lambda: self._log("  Cudart: no disponible (se omite)"))
        self.after(0, lambda: self._log(f"  Descarga total estimada: {total_mb:.1f} MB"))

        # ¿Ya está al día?
        # Para backends con tag tipo "b9019" comparamos número de build.
        # Para backends con tag tipo "tqp-v0.1.1" usamos un version.txt
        # que guardamos nosotros al instalar.
        up_to_date = False
        install_dir = self._meta["dir"]
        version_file = os.path.join(install_dir, "llamastation_version.txt")
        m_tag = re.search(r"b(\d+)", tag)
        if m_tag and current_build is not None:
            # Tag con número de build (oficial llama.cpp)
            up_to_date = current_build >= int(m_tag.group(1))
        elif os.path.isfile(version_file):
            # Tag semántico (TheTom tqp-v0.x.x) — comparar con version.txt
            try:
                saved_tag = open(version_file).read().strip()
                up_to_date = (saved_tag == tag)
                if not up_to_date:
                    self.after(0, lambda s=saved_tag: self._log(f"  Versión instalada (LlamaStation): {s}"))
            except Exception:
                up_to_date = False
        else:
            # Sin version.txt y sin número de build → no podemos confirmar → ofrecer instalar
            up_to_date = False

        if up_to_date:
            self.after(0, lambda: self._log(f"\n✓ Ya tienes la última versión ({tag})."))
            self.after(0, lambda: self._set_progress(1.0, f"Al día · {tag}"))
            self.after(0, lambda: self.btn_update.configure(
                text="Ya estás al día", state="disabled",
                fg_color=C["green"], text_color="#0f0f13"))
            return

        # Hay actualización disponible
        self._pending_asset  = asset
        self._pending_cudart = cudart
        self._pending_tag    = tag
        self._cuda_mm        = cuda_mm
        self._cuda_maj       = cuda_maj

        self.after(0, lambda: self._log(f"\n★ Actualización disponible: {tag}"))
        self.after(0, lambda: self._set_progress(0.15, f"Listo para instalar {tag}"))
        self.after(0, lambda: self.btn_update.configure(
            state="normal", text=f"Instalar {tag}"))

    # ── Paso 2: Instalar ──────────────────────────────────────────────

    def _start_install(self):
        self.btn_update.configure(state="disabled", text="Instalando...")
        self.btn_cancel.configure(state="disabled")
        self._thread = threading.Thread(target=self._install_thread, daemon=True)
        self._thread.start()

    def _install_thread(self):
        asset       = self._pending_asset
        cudart      = self._pending_cudart
        tag         = self._pending_tag
        install_dir = self._meta["dir"]
        tmp_dir     = tempfile.mkdtemp(prefix="llamastation_upd_")
        main_zip    = os.path.join(tmp_dir, "llama_main.zip")
        cudart_zip  = os.path.join(tmp_dir, "llama_cudart.zip") if cudart else None

        try:
            # ─ Descarga principal ────────────────────────────────────
            self.after(0, lambda: self._log(f"\n→ Descargando {asset['name']}..."))
            self._download_file(asset["browser_download_url"], main_zip,
                                label_prefix=asset["name"])
            if self._cancelled: return

            # ─ Descarga cudart ───────────────────────────────────────
            if cudart and cudart_zip:
                self.after(0, lambda: self._log(f"→ Descargando {cudart['name']}..."))
                self._download_file(cudart["browser_download_url"], cudart_zip,
                                    label_prefix=cudart["name"], prog_start=0.5)
                if self._cancelled: return

            # ─ Backup ────────────────────────────────────────────────
            self.after(0, lambda: self._log(f"→ Haciendo backup de {install_dir}..."))
            self.after(0, lambda: self._set_progress(0.82, "Haciendo backup..."))
            if os.path.isdir(install_dir):
                backup_name = f"{install_dir}_backup_{datetime.now():%Y%m%d_%H%M}"
                try:
                    os.rename(install_dir, backup_name)
                    self.after(0, lambda: self._log(f"  Backup: {backup_name}"))
                except Exception as e:
                    self.after(0, lambda err=str(e): self._log(f"  ⚠ Backup falló: {err} (continuando de todos modos)"))

            # ─ Extraer principal ────────────────────────────────────
            self.after(0, lambda: self._log(f"→ Extrayendo en {install_dir}..."))
            self.after(0, lambda: self._set_progress(0.86, "Extrayendo..."))
            os.makedirs(install_dir, exist_ok=True)
            with zipfile.ZipFile(main_zip, "r") as zf:
                zf.extractall(install_dir)

            # Aplanar subcarpeta si el zip la mete dentro de una carpeta
            subdirs = [d for d in os.scandir(install_dir) if d.is_dir()]
            if len(subdirs) == 1:
                sub = subdirs[0].path
                server_in_sub  = os.path.isfile(os.path.join(sub, "llama-server.exe"))
                server_in_root = os.path.isfile(os.path.join(install_dir, "llama-server.exe"))
                if server_in_sub and not server_in_root:
                    for item in os.listdir(sub):
                        shutil.move(os.path.join(sub, item), install_dir)
                    shutil.rmtree(sub, ignore_errors=True)

            # ─ Extraer cudart encima ─────────────────────────────────
            if cudart and cudart_zip and os.path.isfile(cudart_zip):
                self.after(0, lambda: self._log("→ Instalando DLLs de CUDA runtime..."))
                self.after(0, lambda: self._set_progress(0.92, "Instalando cudart DLLs..."))
                cudart_tmp = os.path.join(tmp_dir, "cudart_ext")
                os.makedirs(cudart_tmp, exist_ok=True)
                with zipfile.ZipFile(cudart_zip, "r") as zf:
                    zf.extractall(cudart_tmp)
                for root_d, _, files in os.walk(cudart_tmp):
                    for fname in files:
                        shutil.copy2(os.path.join(root_d, fname), install_dir)
                shutil.rmtree(cudart_tmp, ignore_errors=True)

            # ─ Verificar ────────────────────────────────────────────
            self.after(0, lambda: self._set_progress(0.97, "Verificando..."))
            server_exe = os.path.join(install_dir, "llama-server.exe")
            has_server = os.path.isfile(server_exe)
            has_cuda   = os.path.isfile(os.path.join(install_dir, "ggml-cuda.dll"))
            has_cudart = any(f.name.startswith("cudart") and f.name.endswith(".dll")
                             for f in os.scandir(install_dir) if f.is_file())

            self.after(0, lambda: self._log(
                f"\n{'✓' if has_server else '✗'} llama-server.exe\n"
                f"{'✓' if has_cuda   else '✗'} ggml-cuda.dll (CUDA backend)\n"
                f"{'✓' if has_cudart else '–'} cudart DLLs"
            ))

            if has_server:
                # Guardar tag instalado para comparaciones futuras (backends con versionado semántico)
                try:
                    with open(os.path.join(install_dir, "llamastation_version.txt"), "w") as vf:
                        vf.write(tag)
                except Exception:
                    pass
                self.after(0, lambda: self._log(f"\n✓ {self._meta['label']} {tag} instalado correctamente."))
                self.after(0, lambda: self._set_progress(1.0, f"✓ {tag} instalado"))
                self.after(0, lambda: self.btn_update.configure(
                    text="✓ Instalado", state="disabled",
                    fg_color=C["green"], text_color="#0f0f13"))
                self.after(0, lambda: self.btn_cancel.configure(
                    state="normal", text="Cerrar"))
            else:
                raise RuntimeError("llama-server.exe no encontrado tras la instalación")

        except Exception as e:
            self.after(0, lambda err=str(e): self._log(f"\n✗ Error durante la instalación: {err}"))
            self.after(0, lambda: self._set_progress(0, "Error de instalación"))
            self.after(0, lambda: self.btn_cancel.configure(state="normal"))
            self.after(0, lambda: self.btn_update.configure(
                state="normal", text="Reintentar"))
        finally:
            # Limpieza temporal
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _download_file(self, url, dest, label_prefix="", prog_start=0.15):
        """Descarga con barra de progreso real. prog_start..prog_start+0.35 del rango."""
        prog_range = 0.35
        with requests.get(url, stream=True, timeout=60,
                          headers={"User-Agent": "LlamaStation-Updater"}) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done  = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=131072):
                    if self._cancelled:
                        return
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            frac = prog_start + (done / total) * prog_range
                            mb_done = done / 1_048_576
                            mb_total = total / 1_048_576
                            lbl = f"{label_prefix}  {mb_done:.1f} / {mb_total:.1f} MB"
                            self.after(0, lambda p=frac, l=lbl: self._set_progress(p, l))

    def _on_cancel(self):
        self._cancelled = True
        self.destroy()



# ══════════════════════════════════════════════════════════════════════════
#  MODAL EXPLORADOR DE MODELOS — selector estilo LM Studio
# ══════════════════════════════════════════════════════════════════════════

def _scan_models_dir(models_dir):
    """
    Escanea models_dir buscando archivos .gguf.
    Estructura soportada:
      models_dir/modelo.gguf                  (plano)
      models_dir/NombreModelo/modelo.gguf     (subcarpeta por modelo, como LM Studio)
    Devuelve lista de dicts: {name, path, size, folder}
    """
    results = []
    base = Path(models_dir)
    if not base.exists():
        return results
    # Buscar .gguf en raíz y en subcarpetas de 1 nivel
    for p in sorted(base.rglob("*.gguf")):
        rel = p.relative_to(base)
        parts = rel.parts
        if len(parts) == 1:
            folder = ""
        else:
            folder = parts[0]
        results.append({
            "name":   p.name,
            "path":   str(p),
            "size":   p.stat().st_size if p.exists() else 0,
            "folder": folder,
        })
    return results


class ModelBrowserDialog(ctk.CTkToplevel):
    """
    Modal explorador de modelos: muestra los .gguf organizados por carpeta,
    permite cambiar el directorio base y seleccionar el modelo.
    """

    def __init__(self, parent, models_dir, profiles, settings):
        super().__init__(parent)
        self.profiles    = profiles
        self.settings    = settings
        self.result_path = None
        self.result_dir  = None
        self._models     = []

        self.title("Seleccionar modelo")
        self.geometry(f"{_scale(680)}x{_scale(580)}")
        self.resizable(True, True)
        self.configure(fg_color=C["bg"])
        self.grab_set()
        self.focus_set()

        self.dir_var = tk.StringVar(value=models_dir)
        self._build()
        self._scan()

    def _build(self):
        # ── Header ────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color=C["panel"], corner_radius=0, height=58)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text=T("my_models_title"),
                     font=ctk.CTkFont("Consolas", 15, "bold"),
                     text_color=C["accent2"]).pack(side="left", padx=16, pady=14)

        # ── Barra de directorio ───────────────────────────────────────
        dir_bar = ctk.CTkFrame(self, fg_color=C["card"], corner_radius=0, height=44)
        dir_bar.pack(fill="x"); dir_bar.pack_propagate(False)
        di = ctk.CTkFrame(dir_bar, fg_color="transparent")
        di.pack(fill="both", expand=True, padx=12, pady=6)

        ctk.CTkLabel(di, text=T("folder_label"),
                     font=ctk.CTkFont("Consolas", 11),
                     text_color=C["sub"]).pack(side="left")

        ctk.CTkEntry(di, textvariable=self.dir_var, width=380,
                     fg_color=C["input"], text_color=C["text"],
                     font=ctk.CTkFont("Consolas", 11), height=28
                     ).pack(side="left", padx=6)

        ctk.CTkButton(di, text=T("change"), width=80, height=28,
                      fg_color=C["card2"], hover_color=C["border"],
                      text_color=C["text"], font=ctk.CTkFont("Consolas", 11),
                      command=self._pick_dir).pack(side="left", padx=(0, 6))

        ctk.CTkButton(di, text="↻", width=32, height=28,
                      fg_color=C["card2"], hover_color=C["border"],
                      text_color=C["accent2"], font=ctk.CTkFont("Consolas", 14),
                      command=self._scan).pack(side="left")

        self.count_label = ctk.CTkLabel(di, text="",
                                         font=ctk.CTkFont("Consolas", 10),
                                         text_color=C["sub"])
        self.count_label.pack(side="right")

        # ── Lista de modelos ──────────────────────────────────────────
        self.list_frame = ctk.CTkScrollableFrame(self, fg_color=C["bg"], corner_radius=0)
        self.list_frame.pack(fill="both", expand=True)

        self._empty_lbl = ctk.CTkLabel(self.list_frame,
                                        text=T("no_models_dir"),
                                        font=ctk.CTkFont("Consolas", 12),
                                        text_color=C["dim"], justify="center")
        self._empty_lbl.pack(pady=80)

        # ── Footer ────────────────────────────────────────────────────
        footer = ctk.CTkFrame(self, fg_color=C["panel"], corner_radius=0, height=52)
        footer.pack(fill="x", side="bottom"); footer.pack_propagate(False)

        ctk.CTkButton(footer, text=T("cancel"),
                      fg_color="transparent", hover_color=C["card"],
                      text_color=C["sub"], font=ctk.CTkFont("Consolas", 12),
                      width=100, height=36, command=self.destroy
                      ).pack(side="right", padx=8, pady=8)

        ctk.CTkButton(footer, text=T("manual_browse"),
                      fg_color="transparent", hover_color=C["card"],
                      text_color=C["sub"], font=ctk.CTkFont("Consolas", 11),
                      height=36, command=self._pick_file_manual
                      ).pack(side="left", padx=12, pady=8)

    def _pick_dir(self):
        d = filedialog.askdirectory(title=T("models_folder"))
        if d:
            self.dir_var.set(d)
            self._scan()

    def _scan(self):
        models_dir = self.dir_var.get().strip()
        self._models = _scan_models_dir(models_dir)
        self._render()

    def _render(self):
        for w in self.list_frame.winfo_children():
            w.destroy()

        models = self._models
        self.count_label.configure(text=f"{len(models)} modelos")

        if not models:
            lbl = ctk.CTkLabel(self.list_frame,
                               text=T("no_models_dir2"),
                               font=ctk.CTkFont("Consolas", 12),
                               text_color=C["dim"], justify="center")
            lbl.pack(pady=80)
            return

        # Agrupar por carpeta
        folders = {}
        for m in models:
            key = m["folder"] or "— sin carpeta —"
            folders.setdefault(key, []).append(m)

        for folder_name, folder_models in sorted(folders.items()):
            # Cabecera de carpeta
            fhdr = ctk.CTkFrame(self.list_frame, fg_color="transparent")
            fhdr.pack(fill="x", padx=8, pady=(10, 2))
            ctk.CTkLabel(fhdr,
                         text=f"  📁  {folder_name}",
                         font=ctk.CTkFont("Consolas", 11, "bold"),
                         text_color=C["accent"]).pack(side="left")
            ctk.CTkLabel(fhdr,
                         text=f"{len(folder_models)} archivo{'s' if len(folder_models)!=1 else ''}",
                         font=ctk.CTkFont("Consolas", 10),
                         text_color=C["dim"]).pack(side="right")
            ctk.CTkFrame(self.list_frame, height=1, fg_color=C["border"]).pack(fill="x", padx=8, pady=(0, 4))

            for m in folder_models:
                self._add_model_row(m)

    def _add_model_row(self, m):
        # Detectar cuantización del nombre
        name = m["name"]
        quant = ""
        for q in ["Q4_K_M","Q4_K_S","Q5_K_M","Q6_K","Q8_0","IQ4_XS","IQ3_M","IQ2_M","F16","BF16","Q4_0","Q5_0"]:
            if q.lower() in name.lower():
                quant = q
                break

        size_str = ""
        s = m["size"]
        if s:
            if s > 1_073_741_824:
                size_str = f"{s/1_073_741_824:.1f} GB"
            else:
                size_str = f"{s/1_048_576:.0f} MB"

        row = ctk.CTkFrame(self.list_frame, fg_color=C["card"], corner_radius=8, cursor="hand2")
        row.pack(fill="x", padx=8, pady=2)

        inner = ctk.CTkFrame(row, fg_color="transparent")
        inner.pack(fill="x", padx=14, pady=10)

        left_col = ctk.CTkFrame(inner, fg_color="transparent")
        left_col.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(left_col, text=name,
                     font=ctk.CTkFont("Consolas", 12, "bold"),
                     text_color=C["text"], anchor="w",
                     wraplength=_scale(420), justify="left").pack(anchor="w")

        meta_row = ctk.CTkFrame(left_col, fg_color="transparent")
        meta_row.pack(anchor="w", pady=(2, 0))

        if quant:
            ctk.CTkLabel(meta_row, text=f" {quant} ",
                         font=ctk.CTkFont("Consolas", 9, "bold"),
                         fg_color=C["accent"], text_color="white",
                         corner_radius=3).pack(side="left", padx=(0, 6))

        if size_str:
            ctk.CTkLabel(meta_row, text=size_str,
                         font=ctk.CTkFont("Consolas", 10),
                         text_color=C["sub"]).pack(side="left")

        ctk.CTkButton(inner, text=T("load_btn"), width=90, height=32,
                      fg_color=C["accent"], hover_color="#6457e0",
                      text_color="white", font=ctk.CTkFont("Consolas", 11, "bold"),
                      command=lambda path=m["path"]: self._select(path)
                      ).pack(side="right")

        ctk.CTkButton(inner, text=T("delete_model"), width=32, height=32,
                      fg_color="transparent", hover_color="#5a1a1a",
                      text_color=C["dim"], font=ctk.CTkFont("Consolas", 13),
                      command=lambda path=m["path"], r=row, sz=size_str: self._delete_model(path, r, sz)
                      ).pack(side="right", padx=(0, 4))

        # Click en toda la fila también selecciona
        for w in [row, inner, left_col]:
            w.bind("<Button-1>", lambda e, path=m["path"]: self._select(path))
            w.bind("<Enter>", lambda e, r=row: r.configure(fg_color=C["card2"]))
            w.bind("<Leave>", lambda e, r=row: r.configure(fg_color=C["card"]))

    def _delete_model(self, path, row_widget, size_str):
        name = os.path.basename(path)
        if not messagebox.askyesno(
            T("delete_model_confirm_title"),
            T("delete_model_confirm_msg", name=name, size=size_str or "?")
        ):
            return
        try:
            os.remove(path)
            row_widget.destroy()
            self._models = [m for m in self._models if m["path"] != path]
            self.count_label.configure(text=f"{len(self._models)} modelos")
            messagebox.showinfo(T("delete_model_confirm_title"), T("delete_model_ok"))
        except Exception as e:
            messagebox.showerror(T("delete_model_confirm_title"), T("delete_model_err", err=str(e)))

    def _select(self, path):
        self.result_path = path
        self.result_dir  = self.dir_var.get().strip()
        self.destroy()

    def _pick_file_manual(self):
        path = filedialog.askopenfilename(
            title=T("select_gguf"),
            filetypes=[("GGUF", "*.gguf"), ("Todos", "*.*")]
        )
        if path:
            self.result_path = path
            self.result_dir  = None
            self.destroy()


# ══════════════════════════════════════════════════════════════════════════
#  APP PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
#  PROXY ANTHROPIC MESSAGES API → OpenAI (para Claude Code y otros clientes)
# ══════════════════════════════════════════════════════════════════════════

class AnthropicProxyServer:
    """
    Mini proxy HTTP que expone /v1/messages (formato Anthropic)
    y lo traduce a /v1/chat/completions (formato OpenAI/llama.cpp).
    Permite usar Claude Code y cualquier cliente Anthropic-compatible
    apuntando a LlamaStation sin necesidad de instalaciones extra.
    Puerto: llama-server port + 1 (ej: 8080 → proxy en 8081)
    """

    def __init__(self, openai_base_url: str, proxy_port: int):
        self.openai_base_url = openai_base_url.rstrip("/")
        self.proxy_port      = proxy_port
        self._server         = None
        self._thread         = None

    # ── Conversión de formatos ──────────────────────────────────────────

    @staticmethod
    def _anthropic_to_openai(body: dict) -> dict:
        """Convierte un body Anthropic Messages a formato OpenAI chat/completions."""
        messages = []

        # System prompt
        system = body.get("system", "")
        if isinstance(system, list):
            system = " ".join(
                b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"
            )
        if system:
            messages.append({"role": "system", "content": system})

        # Mensajes
        for msg in body.get("messages", []):
            role    = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extraer texto de bloques de contenido
                content = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            messages.append({"role": role, "content": content})

        oai = {
            "model":       body.get("model", "local"),
            "messages":    messages,
            "max_tokens":  body.get("max_tokens", 4096),
            "stream":      body.get("stream", False),
        }
        if "temperature" in body:
            oai["temperature"] = body["temperature"]
        if "top_p" in body:
            oai["top_p"] = body["top_p"]
        return oai

    @staticmethod
    def _openai_to_anthropic(oai_resp: dict, model: str) -> dict:
        """Convierte respuesta OpenAI a formato Anthropic Messages."""
        choice  = oai_resp.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        usage   = oai_resp.get("usage", {})
        return {
            "id":      oai_resp.get("id", "msg_proxy"),
            "type":    "message",
            "role":    "assistant",
            "model":   model,
            "content": [{"type": "text", "text": content}],
            "stop_reason":    "end_turn",
            "stop_sequence":  None,
            "usage": {
                "input_tokens":  usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }

    @staticmethod
    def _stream_openai_to_anthropic(line: str, model: str) -> str:
        """Convierte una línea SSE de OpenAI a SSE de Anthropic."""
        if not line.startswith("data: "):
            return ""
        data = line[6:].strip()
        if data == "[DONE]":
            return "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        try:
            chunk  = json.loads(data)
            choice = chunk.get("choices", [{}])[0]
            delta  = choice.get("delta", {})
            text   = delta.get("content", "")
            if text:
                payload = json.dumps({
                    "type":  "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text}
                })
                return f"event: content_block_delta\ndata: {payload}\n\n"
        except Exception:
            pass
        return ""

    # ── HTTP Handler ────────────────────────────────────────────────────

    def _make_handler(self):
        proxy = self

        from http.server import BaseHTTPRequestHandler

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass  # silenciar logs de acceso

            def _send_json(self, status: int, body: dict):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)

            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.end_headers()

            def do_GET(self):
                # Health / models pass-through
                try:
                    resp = requests.get(
                        proxy.openai_base_url + self.path, timeout=5
                    )
                    self._send_json(resp.status_code, resp.json())
                except Exception as e:
                    self._send_json(503, {"error": str(e)})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw    = self.rfile.read(length)
                try:
                    ant_body = json.loads(raw)
                except Exception:
                    self._send_json(400, {"error": "Invalid JSON"})
                    return

                model    = ant_body.get("model", "local")
                oai_body = proxy._anthropic_to_openai(ant_body)
                stream   = oai_body.get("stream", False)

                try:
                    resp = requests.post(
                        proxy.openai_base_url + "/v1/chat/completions",
                        json=oai_body,
                        stream=stream,
                        timeout=300,
                    )
                except Exception as e:
                    self._send_json(503, {"error": str(e)})
                    return

                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    # Enviar mensaje de inicio Anthropic
                    start = json.dumps({
                        "type": "message_start",
                        "message": {
                            "id": "msg_proxy", "type": "message",
                            "role": "assistant", "model": model,
                            "content": [], "stop_reason": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0}
                        }
                    })
                    self.wfile.write(f"event: message_start\ndata: {start}\n\n".encode())
                    block_start = json.dumps({"type": "content_block_start", "index": 0,
                                              "content_block": {"type": "text", "text": ""}})
                    self.wfile.write(f"event: content_block_start\ndata: {block_start}\n\n".encode())
                    try:
                        for line in resp.iter_lines():
                            if line:
                                converted = proxy._stream_openai_to_anthropic(
                                    line.decode() if isinstance(line, bytes) else line,
                                    model
                                )
                                if converted:
                                    self.wfile.write(converted.encode())
                                    self.wfile.flush()
                    except Exception:
                        pass
                    block_stop = json.dumps({"type": "content_block_stop", "index": 0})
                    self.wfile.write(f"event: content_block_stop\ndata: {block_stop}\n\n".encode())
                    msg_delta = json.dumps({"type": "message_delta",
                                            "delta": {"stop_reason": "end_turn"},
                                            "usage": {"output_tokens": 0}})
                    self.wfile.write(f"event: message_delta\ndata: {msg_delta}\n\n".encode())
                    self.wfile.write(b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
                else:
                    try:
                        oai_resp  = resp.json()
                        ant_resp  = proxy._openai_to_anthropic(oai_resp, model)
                        self._send_json(200, ant_resp)
                    except Exception as e:
                        self._send_json(502, {"error": str(e)})

        return Handler

    # ── Start / Stop ────────────────────────────────────────────────────

    def start(self):
        from http.server import HTTPServer
        handler = self._make_handler()
        try:
            self._server = HTTPServer(("127.0.0.1", self.proxy_port), handler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            return True
        except Exception:
            return False

    def stop(self):
        if self._server:
            try:
                self._server.shutdown()
            except Exception:
                pass
            self._server = None

class LlamaStation(VoiceMixin, ctk.CTk):
    def __init__(self):
        self.profiles         = load_profiles()
        self.settings         = load_settings()
        # Aplicar tema e idioma guardados ANTES de construir la UI
        apply_theme(self.settings.get("theme", "dark"))
        set_lang(self.settings.get("lang", "es"))

        super().__init__()
        self.title("LlamaStation")
        _w = _scale(1300)
        _h = _scale(840)
        self.geometry(f"{_w}x{_h}")
        self.minsize(_scale(900), _scale(620))

        # Icono de la ventana
        try:
            _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llamastation_icon.ico")
            if os.path.isfile(_icon_path):
                self.iconbitmap(_icon_path)
            else:
                # Generar icono al vuelo si no existe
                from PIL import Image, ImageDraw, ImageTk
                _s = 64
                _img = Image.new("RGBA", (_s, _s), (0, 0, 0, 0))
                _d = ImageDraw.Draw(_img)
                _d.rounded_rectangle([0, 0, 63, 63], radius=12, fill="#1a3a5c")
                _d.ellipse([8, 8, 56, 56], outline="#4ac8d0", width=3)
                _ico = _img.resize((32, 32))
                _ico_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llamastation_icon.ico")
                _ico.save(_ico_path, format="ICO", sizes=[(32,32),(16,16)])
                self.iconbitmap(_ico_path)
        except Exception:
            pass  # Si falla, queda el icono por defecto
        self.configure(fg_color=C["bg"])

        self.server_process   = None
        self.server_running   = False
        self._proxy_server    = None
        self.chat_history     = []
        self.current_model    = ""
        self.current_prof     = dict(DEFAULT_PROFILE)
        self._stop_generation = False   # flag para abortar generación
        self._reasoning_control_supported = True  # se desactiva solo si el backend responde 404
        self._current_gen_id = None     # id de la generación en curso (para /control)
        self._session_tokens  = 0       # tokens de contexto acumulados en sesión
        self._stopping        = False   # flag de parada manual del servidor
        self._attached_image  = None    # ruta de imagen adjunta para vision
        self._attached_files  = []      # lista de archivos de texto adjuntos (py, html, etc.)
        self._wd_var          = tk.BooleanVar(value=self.settings.get("watchdog_auto_relaunch", False))

        self._init_voice_state() if _VOICE_OK else None
        self._build_ui()
        self._check_server_on_start()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._load_sessions_from_disk()

        # ── Anti-flicker al restaurar ventana ─────────────────────────
        # Arrancamos oculta, esperamos a que tkinter pinte todo y luego mostramos
        self.withdraw()
        self.after(120, self._show_ready)
        # Bind al evento de restaurar desde minimizado
        self.bind("<Map>", self._on_map)

    # ── Layout ────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Estado de colapso de sidebars
        self._left_collapsed  = False
        self._right_collapsed = False

        # Historial de chats: lista de {id, title, history, model, profile}
        self._chat_sessions   = []
        self._current_session_id = None

        # ── Sidebar izquierda (historial) ──────────────────────────────
        self.left_sidebar = ctk.CTkFrame(self, width=_scale(240), fg_color=C["panel"], corner_radius=0)
        self.left_sidebar.pack(side="left", fill="y")
        self.left_sidebar.pack_propagate(False)

        # ── Panel central (chat / tabs) ────────────────────────────────
        self.main = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        self.main.pack(side="left", fill="both", expand=True)

        # ── Sidebar derecha (controles) ────────────────────────────────
        self.sidebar = ctk.CTkFrame(self, width=_scale(260), fg_color=C["panel"], corner_radius=0)
        self.sidebar.pack(side="right", fill="y")
        self.sidebar.pack_propagate(False)

        self._build_left_sidebar()
        self._build_sidebar()       # sidebar derecha (controles existentes)
        self._build_tabs()

    def _build_left_sidebar(self):
        ls = self.left_sidebar

        # ── Header ────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(ls, fg_color="transparent")
        hdr.pack(fill="x", padx=12, pady=(16, 0))

        ctk.CTkLabel(hdr, text=T("app_title"),
                     font=ctk.CTkFont("Consolas", 16, "bold"),
                     text_color=C["accent2"]).pack(side="left", anchor="w")

        # Versión
        ctk.CTkLabel(ls, text=APP_VERSION,
                     font=ctk.CTkFont("Consolas", 9),
                     text_color=C["dim"]).pack(anchor="w", padx=16, pady=(2, 0))

        # Botón colapsar sidebar izquierda
        self.btn_collapse_left = ctk.CTkButton(
            hdr, text="‹", width=28, height=28,
            fg_color="transparent", hover_color=C["card"],
            text_color=C["sub"], font=ctk.CTkFont("Consolas", 16, "bold"),
            command=self._toggle_left_sidebar
        )
        self.btn_collapse_left.pack(side="right")

        ctk.CTkFrame(ls, height=1, fg_color=C["border"]).pack(fill="x", padx=12, pady=(10, 4))

        # Botón nuevo chat
        ctk.CTkButton(ls, text=T("new_chat"),
                      fg_color=C["accent"], hover_color="#6457e0",
                      text_color="white", font=ctk.CTkFont("Consolas", 12, "bold"),
                      height=36, corner_radius=8,
                      command=self._new_chat
                      ).pack(fill="x", padx=12, pady=(4, 8))

        ctk.CTkLabel(ls, text=T("conversations"),
                     font=ctk.CTkFont("Consolas", 9, "bold"),
                     text_color=C["dim"]).pack(anchor="w", padx=16, pady=(0, 4))

        # Lista scrollable de sesiones
        self.history_list = ctk.CTkScrollableFrame(ls, fg_color="transparent", corner_radius=0)
        self.history_list.pack(fill="both", expand=True, padx=4)

        self._no_history_label = ctk.CTkLabel(
            self.history_list,
            text=T("no_history"),
            font=ctk.CTkFont("Consolas", 11),
            text_color=C["dim"], justify="center"
        )
        self._no_history_label.pack(pady=40)

        # Botón colapsar pegado al borde derecho de la sidebar — toggle externo
        self._build_left_toggle_tab()

    def _build_left_toggle_tab(self):
        pass  # Los botones de toggle están en el header del chat

    def _toggle_left_sidebar(self):
        if self._left_collapsed:
            self.left_sidebar.pack(side="left", fill="y", before=self.main)
            self.left_sidebar.pack_propagate(False)
            self._left_collapsed = False
        else:
            self.left_sidebar.pack_forget()
            self._left_collapsed = True
        # Actualizar icono en el header del chat
        if hasattr(self, "btn_toggle_left"):
            self.btn_toggle_left.configure(text="☰" if self._left_collapsed else "☰")

    def _build_right_toggle_tab(self):
        pass  # Los botones de toggle están en el header del chat

    def _toggle_right_sidebar(self):
        if self._right_collapsed:
            self.sidebar.pack(side="right", fill="y")
            self.sidebar.pack_propagate(False)
            self._right_collapsed = False
        else:
            self.sidebar.pack_forget()
            self._right_collapsed = True

    # ── Gestión del historial de chats ────────────────────────────────────

    def _new_chat(self):
        """Guarda la sesión actual (si tiene mensajes) e inicia una nueva."""
        self._save_current_session()
        self.chat_history    = []
        self._session_tokens = 0
        self._current_session_id = None
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")
        # Deseleccionar en la lista
        self._refresh_history_list()

    def _save_current_session(self):
        """Guarda el chat actual en la lista de sesiones."""
        if not self.chat_history:
            return
        # Título: primer mensaje del usuario, truncado
        title = ""
        for msg in self.chat_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # mensaje con imagen
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            content = part.get("text", "")
                            break
                title = str(content)[:45]
                if len(str(msg.get("content",""))) > 45:
                    title += "..."
                break
        if not title:
            title = "Chat sin título"

        if self._current_session_id is not None:
            # Actualizar sesión existente
            for s in self._chat_sessions:
                if s["id"] == self._current_session_id:
                    s["history"] = list(self.chat_history)
                    s["title"]   = title
                    break
        else:
            # Nueva sesión
            import time as _time
            sid = int(_time.time() * 1000)
            self._current_session_id = sid
            self._chat_sessions.insert(0, {
                "id":      sid,
                "title":   title,
                "history": list(self.chat_history),
                "model":   self.current_model,
            })
        self._save_sessions_to_disk()
        self._refresh_history_list()

    def _load_session(self, session_id):
        """Carga una sesión del historial al chat."""
        self._save_current_session()
        for s in self._chat_sessions:
            if s["id"] == session_id:
                self._current_session_id = session_id
                self.chat_history = list(s["history"])
                # Reconstruir display
                self.chat_display.configure(state="normal")
                self.chat_display.delete("1.0", "end")
                for msg in self.chat_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = "[imagen adjunta]"
                    if role == "user":
                        self.chat_display.insert("end", f"\nTú\n{content}\n")
                    elif role == "assistant":
                        self.chat_display.insert("end", f"\nModelo\n{content}\n")
                self.chat_display.see("end")
                self.chat_display.configure(state="disabled")
                self._refresh_history_list()
                break

    def _delete_session(self, session_id):
        """Elimina una sesión del historial."""
        self._chat_sessions = [s for s in self._chat_sessions if s["id"] != session_id]
        if self._current_session_id == session_id:
            self._current_session_id = None
            self.chat_history = []
            self._session_tokens = 0
            self.chat_display.configure(state="normal")
            self.chat_display.delete("1.0", "end")
            self.chat_display.configure(state="disabled")
        self._save_sessions_to_disk()
        self._refresh_history_list()

    def _refresh_history_list(self):
        """Redibuja la lista de sesiones en la sidebar."""
        for w in self.history_list.winfo_children():
            w.destroy()

        if not self._chat_sessions:
            self._no_history_label = ctk.CTkLabel(
                self.history_list,
                text=T("no_history"),
                font=ctk.CTkFont("Consolas", 11),
                text_color=C["dim"], justify="center"
            )
            self._no_history_label.pack(pady=40)
            return

        for s in self._chat_sessions:
            sid   = s["id"]
            title = s["title"]
            is_active = (sid == self._current_session_id)

            row = ctk.CTkFrame(
                self.history_list,
                fg_color=C["card"] if is_active else "transparent",
                corner_radius=8, cursor="hand2"
            )
            row.pack(fill="x", pady=2, padx=4)

            inner = ctk.CTkFrame(row, fg_color="transparent")
            inner.pack(fill="x", padx=8, pady=6)

            ctk.CTkLabel(inner, text=title,
                         font=ctk.CTkFont("Consolas", 11),
                         text_color=C["accent2"] if is_active else C["text"],
                         anchor="w", wraplength=_scale(170), justify="left"
                         ).pack(side="left", fill="x", expand=True)

            # Botón eliminar (aparece en hover)
            del_btn = ctk.CTkButton(
                inner, text="✕", width=20, height=20,
                fg_color="transparent", hover_color=C["card2"],
                text_color=C["dim"], font=ctk.CTkFont("Consolas", 11),
                command=lambda i=sid: self._delete_session(i)
            )
            del_btn.pack(side="right")

            # Click en la fila carga la sesión
            for w in [row, inner]:
                w.bind("<Button-1>", lambda e, i=sid: self._load_session(i))
                w.bind("<Enter>",    lambda e, r=row, a=is_active: r.configure(
                    fg_color=C["card"] if not a else C["card"]))
                w.bind("<Leave>",    lambda e, r=row, a=is_active: r.configure(
                    fg_color=C["card"] if a else "transparent"))

    # ── Persistencia del historial ────────────────────────────────────────

    def _sessions_file(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "llamastation_sessions.json")

    def _save_sessions_to_disk(self):
        try:
            with open(self._sessions_file(), "w", encoding="utf-8") as f:
                json.dump(self._chat_sessions, f, ensure_ascii=False, indent=2)
        except Exception: pass

    def _load_sessions_from_disk(self):
        try:
            p = self._sessions_file()
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    self._chat_sessions = json.load(f)
                self._refresh_history_list()
        except Exception: pass

    def _build_sidebar(self):
        sb = self.sidebar

        # Header derecha con botón colapso
        rhdr = ctk.CTkFrame(sb, fg_color="transparent")
        rhdr.pack(fill="x", padx=12, pady=(14, 0))
        ctk.CTkLabel(rhdr, text=T("controls"),
                     font=ctk.CTkFont("Consolas", 13, "bold"),
                     text_color=C["sub"]).pack(side="left")
        ctk.CTkButton(rhdr, text="›", width=28, height=28,
                      fg_color="transparent", hover_color=C["card"],
                      text_color=C["sub"], font=ctk.CTkFont("Consolas", 16, "bold"),
                      command=self._toggle_right_sidebar
                      ).pack(side="right")

        ctk.CTkFrame(sb, height=1, fg_color=C["border"]).pack(fill="x", padx=16, pady=(10, 10))
        self._build_right_toggle_tab()

        # ── Área scrollable (todo el contenido central) ────────────────
        sb_scroll = ctk.CTkScrollableFrame(sb, fg_color="transparent", corner_radius=0)
        sb_scroll.pack(fill="both", expand=True)
        # A partir de aquí usamos sb_scroll como padre en lugar de sb
        sb = sb_scroll

        # Modelo
        mc = ctk.CTkFrame(sb, fg_color=C["card"], corner_radius=10)
        mc.pack(fill="x", padx=12, pady=(0, 6))
        mi = ctk.CTkFrame(mc, fg_color="transparent")
        mi.pack(fill="x", padx=12, pady=10)
        ctk.CTkLabel(mi, text=T("loaded_model"),
                     font=ctk.CTkFont("Consolas", 9, "bold"),
                     text_color=C["sub"]).pack(anchor="w")
        self.model_label = ctk.CTkLabel(mi, text=T("no_model"),
                                         font=ctk.CTkFont("Consolas", 11),
                                         text_color=C["text"],
                                         wraplength=_scale(210), justify="left")
        self.model_label.pack(anchor="w", pady=(4, 0))

        ctk.CTkButton(sb, text=T("my_models"),
                       fg_color=C["card2"], hover_color=C["border"],
                       text_color=C["text"],
                       font=ctk.CTkFont("Consolas", 12), height=38,
                       corner_radius=8, command=self._open_model_browser
                       ).pack(fill="x", padx=12, pady=(0, 3))

        ctk.CTkButton(sb, text=T("download_models"),
                       fg_color="transparent", hover_color=C["card"],
                       text_color=C["accent2"],
                       font=ctk.CTkFont("Consolas", 11), height=32,
                       corner_radius=8, command=self._show_download
                       ).pack(fill="x", padx=12, pady=(0, 6))

        # Estado
        sc = ctk.CTkFrame(sb, fg_color=C["card"], corner_radius=10)
        sc.pack(fill="x", padx=12, pady=(0, 6))
        si = ctk.CTkFrame(sc, fg_color="transparent")
        si.pack(fill="x", padx=12, pady=10)
        ctk.CTkLabel(si, text=T("server_label"),
                     font=ctk.CTkFont("Consolas", 9, "bold"),
                     text_color=C["sub"]).pack(anchor="w")
        sr = ctk.CTkFrame(si, fg_color="transparent")
        sr.pack(fill="x", pady=(4, 0))
        self.status_dot   = ctk.CTkLabel(sr, text="●", width=16,
                                          font=ctk.CTkFont(size=14), text_color=C["red"])
        self.status_dot.pack(side="left")
        self.status_label = ctk.CTkLabel(sr, text=T("server_stopped"),
                                          font=ctk.CTkFont("Consolas", 12),
                                          text_color=C["text"])
        self.status_label.pack(side="left", padx=4)
        self.port_label   = ctk.CTkLabel(si, text=T("server_port"),
                                          font=ctk.CTkFont("Consolas", 10),
                                          text_color=C["sub"])
        self.port_label.pack(anchor="w", pady=(2, 0))

        # ── VRAM meter ────────────────────────────────────────────────
        self.vram_frame = ctk.CTkFrame(sc, fg_color="transparent")
        self.vram_frame.pack(fill="x", padx=12, pady=(0, 10))
        self._vram_bars  = []   # lista de (label, progressbar, label_val)
        self._vram_timer = None
        self._build_vram_bars()

        bf = ctk.CTkFrame(sb, fg_color="transparent")
        bf.pack(fill="x", padx=12, pady=(0, 6))
        self.btn_start = ctk.CTkButton(bf, text=T("start_server"),
                                        fg_color=C["accent"], hover_color="#6457e0",
                                        font=ctk.CTkFont("Consolas", 12, "bold"),
                                        height=38, command=self.start_server)
        self.btn_start.pack(fill="x", pady=(0, 5))
        self.btn_stop = ctk.CTkButton(bf, text=T("stop_server"),
                                       fg_color="#3a1a1a", hover_color="#5a2020",
                                       text_color=C["red"],
                                       font=ctk.CTkFont("Consolas", 12),
                                       height=38, state="disabled",
                                       command=self.stop_server)
        self.btn_stop.pack(fill="x")

        ctk.CTkFrame(sb, height=1, fg_color=C["border"]).pack(fill="x", padx=16, pady=10)

        # ── Selector de backend ─────────────────────────────────────────
        ctk.CTkFrame(sb, height=1, fg_color=C["border"]).pack(fill="x", padx=16, pady=(4, 6))
        ctk.CTkLabel(sb, text=T("backend_label"),
                     font=ctk.CTkFont("Consolas", 9, "bold"),
                     text_color=C["sub"]).pack(anchor="w", padx=20)

        self.backend_var = tk.StringVar(value=list(BACKENDS.keys())[0])
        # Detectar cuál está en settings actualmente
        cur_path = self.settings.get("server_path", "")
        for bname, bpath in BACKENDS.items():
            if cur_path == bpath:
                self.backend_var.set(bname)
                break

        self.backend_menu = ctk.CTkOptionMenu(
            sb,
            variable=self.backend_var,
            values=list(BACKENDS.keys()),
            fg_color=C["card2"],
            button_color=C["accent"],
            button_hover_color=C["accent2"],
            dropdown_fg_color=C["card"],
            dropdown_hover_color=C["card2"],
            text_color=C["text"],
            dropdown_text_color=C["text"],
            font=ctk.CTkFont("Consolas", 11),
            dropdown_font=ctk.CTkFont("Consolas", 11),
            anchor="w",
            command=lambda _: self._on_backend_change()
        )
        self.backend_menu.pack(fill="x", padx=14, pady=(4, 6))

        ctk.CTkFrame(sb, height=1, fg_color=C["border"]).pack(fill="x", padx=16, pady=(2, 4))
        # ────────────────────────────────────────────────────────────────

        self.nav_btns = {}
        for icon, label_key, cmd in [
            ("💬", "nav_chat",     self._show_chat),
            ("⚙️", "nav_server",  self._show_server),
            ("📋", "nav_logs",     self._show_logs),
            ("ℹ️", "nav_info",    self._show_info),
            ("🌐", "nav_download", self._show_download),
            ("📡", "nav_api",      self._show_api_docs),
            ("🎤", "nav_voice",    self._show_voice),
            ("⚖️", "nav_about",   self._show_about),
        ]:
            b = ctk.CTkButton(sb, text=f"  {icon}  {T(label_key)}",
                               fg_color="transparent", hover_color=C["card"],
                               text_color=C["sub"], font=ctk.CTkFont("Consolas", 13),
                               anchor="w", height=40, command=cmd)
            b.pack(fill="x", padx=8, pady=2)
            self.nav_btns[label_key] = b

        ctk.CTkFrame(sb, height=1, fg_color=C["border"]).pack(fill="x", padx=16, pady=(8, 6))

        # Botón de sonido
        sound_on = self.settings.get("sound", True)
        self.btn_sound = ctk.CTkButton(
            sb, text=T("sound_on") if sound_on else T("sound_off"),
            fg_color=C["card2"], hover_color=C["border"],
            text_color=C["sub"] if sound_on else C["dim"],
            font=ctk.CTkFont("Consolas", 11),
            height=32, corner_radius=8,
            command=self._toggle_sound
        )
        self.btn_sound.pack(fill="x", padx=12, pady=(0, 4))

        # Botón de idioma
        self.btn_lang = ctk.CTkButton(
            sb, text=T("lang_btn"),
            fg_color=C["card2"], hover_color=C["border"],
            text_color=C["sub"], font=ctk.CTkFont("Consolas", 11),
            height=32, corner_radius=8,
            command=self._toggle_lang
        )
        self.btn_lang.pack(fill="x", padx=12, pady=(0, 4))

        # Botón de tema claro/oscuro
        cur_theme = self.settings.get("theme", "dark")
        theme_icon = T("theme_to_light") if cur_theme == "dark" else T("theme_to_dark")
        self.btn_theme = ctk.CTkButton(
            sb, text=theme_icon,
            fg_color=C["card2"], hover_color=C["border"],
            text_color=C["sub"], font=ctk.CTkFont("Consolas", 11),
            height=32, corner_radius=8,
            command=self._toggle_theme
        )
        self.btn_theme.pack(fill="x", padx=12, pady=(0, 4))

        # Botón de actualización
        self._update_btns = {}
        self.btn_update_backend = ctk.CTkButton(
            sb, text=T("update_llama"),
            fg_color=C["card2"], hover_color=C["border"],
            text_color=C["sub"], font=ctk.CTkFont("Consolas", 11),
            height=32, corner_radius=8,
            command=lambda: self._open_update_dialog(self.backend_var.get())
        )
        self.btn_update_backend.pack(fill="x", padx=12, pady=(0, 4))
        for bkey in BACKEND_META:
            self._update_btns[bkey] = self.btn_update_backend

        self.ver_label = ctk.CTkLabel(sb, text="llama.cpp: —",
                                       font=ctk.CTkFont("Consolas", 10),
                                       text_color=C["sub"])
        self.ver_label.pack(anchor="w", padx=14, pady=(4, 12))

        self._detect_llama_version()
        threading.Thread(target=self._silent_update_check, daemon=True).start()

    def _build_tabs(self):
        self.frames = {
            "Chat":        self._build_chat(self.main),
            "Servidor":    self._build_server(self.main),
            "Logs":        self._build_logs(self.main),
            "Info modelo": self._build_info(self.main),
            "Descargar":   self._build_downloader(self.main),
            "API Docs":    self._build_api_docs(self.main),
            "Voz":         self._build_voice(self.main) if _VOICE_OK else ctk.CTkFrame(self.main),
            "Acerca de":   self._build_about(self.main),
        }
        self._show_chat()

    def _show_frame(self, name):
        for f in self.frames.values(): f.pack_forget()
        self.frames[name].pack(fill="both", expand=True)
        # nav_btns ahora usa label_key como clave
        key_map = {
            "Chat": "nav_chat", "Servidor": "nav_server", "Logs": "nav_logs",
            "Info modelo": "nav_info", "Descargar": "nav_download", "API Docs": "nav_api",
            "Voz": "nav_voice", "Acerca de": "nav_about",
        }
        active_key = key_map.get(name, name)
        for k, b in self.nav_btns.items():
            b.configure(fg_color=C["card"] if k == active_key else "transparent",
                        text_color=C["accent2"] if k == active_key else C["sub"])

    def _toggle_sound(self):
        cur = self.settings.get("sound", True)
        new_val = not cur
        self.settings["sound"] = new_val
        save_settings(self.settings)
        if hasattr(self, "btn_sound"):
            self.btn_sound.configure(
                text=T("sound_on") if new_val else T("sound_off"),
                text_color=C["sub"] if new_val else C["dim"]
            )

    def _toggle_lang(self):
        """Cambia entre español e inglés, guarda preferencia y reconstruye la UI."""
        cur = self.settings.get("lang", "es")
        new_lang = "en" if cur == "es" else "es"
        self.settings["lang"] = new_lang
        save_settings(self.settings)
        set_lang(new_lang)
        # Reconstruir UI igual que el cambio de tema
        sessions_backup = list(getattr(self, "_chat_sessions", []))
        cur_sid_backup  = getattr(self, "_current_session_id", None)
        self.configure(fg_color=C["bg"])
        for widget in self.winfo_children():
            widget.destroy()
        self.frames   = {}
        self.nav_btns = {}
        self._build_ui()
        self._chat_sessions = sessions_backup
        self._current_session_id = cur_sid_backup
        self._refresh_history_list()
        self._check_server_on_start()
        self._detect_llama_version()

    def _show_chat(self):   self._show_frame("Chat")
    def _show_server(self): self._show_frame("Servidor")
    def _show_logs(self):   self._show_frame("Logs")
    def _show_info(self):   self._show_frame("Info modelo"); self._refresh_info()
    def _show_voice(self):  self._show_frame("Voz")

    # ── Modal de modelo ───────────────────────────────────────────────────

    def _open_model_dialog(self):
        path = filedialog.askopenfilename(
            title=T("select_gguf"),
            filetypes=[("GGUF", "*.gguf"), ("Todos", "*.*")]
        )
        if not path: return
        dlg = LoadModelDialog(self, path, self.profiles)
        self.wait_window(dlg)
        if dlg.result is not None:
            self.current_model = path
            self.current_prof  = dlg.result
            self.model_label.configure(text=Path(path).name)
            self._log(f"[{datetime.now():%H:%M:%S}] Modelo: {Path(path).name}")
            self.settings["last_model"] = path
            save_settings(self.settings)
            # Sincronizar system prompt al chat
            sp = self.current_prof.get("system_prompt", "")
            if sp and hasattr(self, "sys_entry"):
                self.sys_entry.delete(0, "end")
                self.sys_entry.insert(0, sp)

    # ── Chat ──────────────────────────────────────────────────────────────

    def _build_chat(self, p):
        f = ctk.CTkFrame(p, fg_color=C["bg"], corner_radius=0)

        hdr = ctk.CTkFrame(f, fg_color=C["panel"], height=52, corner_radius=0)
        hdr.pack(fill="x"); hdr.pack_propagate(False)

        # Botón toggle sidebar izquierda — siempre visible
        self.btn_toggle_left = ctk.CTkButton(
            hdr, text="☰", width=36, height=36,
            fg_color="transparent", hover_color=C["card"],
            text_color=C["sub"], font=ctk.CTkFont(size=16),
            command=self._toggle_left_sidebar
        )
        self.btn_toggle_left.pack(side="left", padx=(8, 0), pady=8)

        ctk.CTkLabel(hdr, text=T("chat_title"),
                     font=ctk.CTkFont("Consolas", 15, "bold"),
                     text_color=C["text"]).pack(side="left", padx=12, pady=14)

        # Botón toggle sidebar derecha — siempre visible
        self.btn_toggle_right = ctk.CTkButton(
            hdr, text="⊟", width=36, height=36,
            fg_color="transparent", hover_color=C["card"],
            text_color=C["sub"], font=ctk.CTkFont(size=16),
            command=self._toggle_right_sidebar
        )
        self.btn_toggle_right.pack(side="right", padx=(0, 4), pady=8)

        ctk.CTkButton(hdr, text=T("clear_chat"), fg_color="transparent",
                       hover_color=C["card"], text_color=C["sub"],
                       font=ctk.CTkFont("Consolas", 11), width=70,
                       command=self._clear_chat).pack(side="right", padx=4)

        # Toggle thinking (mostrar/ocultar thinking en UI)
        _think_show = self.settings.get("toggle_thinking_show", True)
        self.thinking_var = tk.BooleanVar(value=_think_show)
        self.btn_thinking = ctk.CTkButton(
            hdr, text=T("thinking_on") if _think_show else T("thinking_off"),
            width=130, height=30,
            fg_color=C["accent"] if _think_show else C["card2"],
            hover_color="#6457e0",
            text_color="white" if _think_show else C["sub"],
            font=ctk.CTkFont("Consolas", 11, "bold"),
            command=self._toggle_thinking
        )
        self.btn_thinking.pack(side="right", padx=(0, 8), pady=10)

        # Toggle enable_thinking
        _think_en = self.settings.get("toggle_thinking_enable", True)
        self.enable_thinking_var = tk.BooleanVar(value=_think_en)
        self.btn_enable_thinking = ctk.CTkButton(
            hdr, text=T("enable_thinking_on") if _think_en else T("enable_thinking_off"),
            width=110, height=30,
            fg_color=C["card2"], hover_color=C["border"],
            text_color=C["accent2"] if _think_en else C["yellow"],
            font=ctk.CTkFont("Consolas", 11, "bold"),
            command=self._toggle_enable_thinking
        )
        self.btn_enable_thinking.pack(side="right", padx=(0, 4), pady=10)

        # Reasoning budget slider (visible solo cuando thinking está ON)
        _budget = self.settings.get("reasoning_budget", 8000)
        self.reasoning_budget_var = tk.IntVar(value=_budget)
        self.reasoning_budget_frame = ctk.CTkFrame(hdr, fg_color="transparent")
        self.reasoning_budget_frame.pack(side="right", padx=(0, 4), pady=10)
        self.reasoning_budget_label = ctk.CTkLabel(
            self.reasoning_budget_frame,
            text=f"Budget: {_budget//1000}k",
            font=ctk.CTkFont("Consolas", 10),
            text_color=C["sub"], width=60
        )
        self.reasoning_budget_label.pack(side="left", padx=(0, 2))
        self.reasoning_budget_slider = ctk.CTkSlider(
            self.reasoning_budget_frame,
            from_=1000, to=32000, number_of_steps=31,
            variable=self.reasoning_budget_var,
            width=90, height=16,
            command=self._on_reasoning_budget_change
        )
        self.reasoning_budget_slider.pack(side="left")
        # Mostrar/ocultar según estado inicial del thinking
        if not _think_en:
            self.reasoning_budget_frame.pack_forget()

        # Toggle web search
        _web = self.settings.get("toggle_websearch", False)
        self.websearch_var = tk.BooleanVar(value=_web)
        self.btn_websearch = ctk.CTkButton(
            hdr, text=T("web_on") if _web else T("web_off"),
            width=110, height=30,
            fg_color=C["green"] if _web else C["card2"],
            hover_color=C["border"],
            text_color="#0f0f13" if _web else C["sub"],
            font=ctk.CTkFont("Consolas", 11, "bold"),
            command=self._toggle_websearch
        )
        self.btn_websearch.pack(side="right", padx=(0, 6), pady=10)

        # Toggle auto-continue (continuation loop cuando finish_reason == "length")
        _autocont = self.settings.get("toggle_autocont", False)
        self.autocont_var = tk.BooleanVar(value=_autocont)
        self.btn_autocont = ctk.CTkButton(
            hdr, text="🔁 AutoCont ON" if _autocont else "🔁 AutoCont",
            width=110, height=30,
            fg_color=C["yellow"] if _autocont else C["card2"],
            hover_color=C["border"],
            text_color="#0f0f13" if _autocont else C["sub"],
            font=ctk.CTkFont("Consolas", 11, "bold"),
            command=self._toggle_autocont
        )
        self.btn_autocont.pack(side="right", padx=(0, 4), pady=10)

        # Toggle file tools (acceso a carpeta local: listar, ver/analizar imagenes, crear carpetas, mover archivos)
        _fs = self.settings.get("toggle_fstools", False)
        self.fstools_var = tk.BooleanVar(value=_fs)
        self.btn_fstools = ctk.CTkButton(
            hdr, text="\U0001f4c1 Files ON" if _fs else "\U0001f4c1 Files",
            width=100, height=30,
            fg_color=C["green"] if _fs else C["card2"],
            hover_color=C["border"],
            text_color="#0f0f13" if _fs else C["sub"],
            font=ctk.CTkFont("Consolas", 11, "bold"),
            command=self._toggle_fstools
        )
        self.btn_fstools.pack(side="right", padx=(0, 4), pady=10)

        self.chat_display = ctk.CTkTextbox(f, fg_color=C["panel"],
                                            text_color=C["text"],
                                            font=ctk.CTkFont("Segoe UI", 14),
                                            wrap="word", state="disabled", corner_radius=0)
        self.chat_display.pack(fill="both", expand=True)
        # Configurar tags de color para el thinking
        tb = self.chat_display._textbox
        tb.tag_config("thinking", foreground="#8b7cf8",
                      font=("Segoe UI", 13, "italic"))
        tb.tag_config("think_hdr", foreground="#6457e0",
                      font=("Segoe UI", 12, "bold italic"))

        sp_bar = ctk.CTkFrame(f, fg_color=C["card2"], height=38, corner_radius=0)
        sp_bar.pack(fill="x"); sp_bar.pack_propagate(False)
        sp_in = ctk.CTkFrame(sp_bar, fg_color="transparent")
        sp_in.pack(fill="both", expand=True, padx=16, pady=4)
        ctk.CTkLabel(sp_in, text=T("system_label"), font=ctk.CTkFont("Consolas", 11),
                     text_color=C["sub"]).pack(side="left")
        self.sys_entry = ctk.CTkEntry(sp_in, fg_color="transparent",
                                       text_color=C["text"],
                                       font=ctk.CTkFont("Consolas", 11),
                                       placeholder_text=T("system_ph"),
                                       border_width=0, height=28)
        self.sys_entry.pack(side="left", fill="x", expand=True, padx=8)

        # Barra de chips para archivos adjuntos
        self.chips_bar = ctk.CTkFrame(f, fg_color=C["card2"], height=0, corner_radius=0)
        self.chips_bar.pack(fill="x")
        self.chips_bar.pack_propagate(False)
        self.chips_inner = ctk.CTkFrame(self.chips_bar, fg_color="transparent")
        self.chips_inner.pack(fill="x", padx=12, pady=4)

        inp = ctk.CTkFrame(f, fg_color=C["card"], height=80, corner_radius=0)
        inp.pack(fill="x"); inp.pack_propagate(False)
        ii = ctk.CTkFrame(inp, fg_color="transparent")
        ii.pack(fill="both", expand=True, padx=16, pady=12)
        self.chat_input = ctk.CTkTextbox(ii, height=44,
                                          fg_color=C["input"], text_color=C["text"],
                                          font=ctk.CTkFont("Segoe UI", 14), corner_radius=8)
        self.chat_input.pack(side="left", fill="both", expand=True)
        self.chat_input.bind("<Return>", self._enter_key)

        # Drag & drop de imágenes sobre el input
        _setup_dnd(self.chat_input, self._on_image_drop)

        # Botón adjuntar imagen (vision)
        self.btn_attach = ctk.CTkButton(ii, text="🖼", width=44, height=44,
                                         fg_color=C["card2"], hover_color=C["border"],
                                         text_color=C["sub"],
                                         font=ctk.CTkFont(size=18),
                                         corner_radius=8,
                                         command=self._attach_image)
        self.btn_attach.pack(side="left", padx=(10, 0))

        # Botón adjuntar archivo de texto (py, html, bat, etc.) - siempre visible
        self.btn_attach_file = ctk.CTkButton(ii, text="📁", width=44, height=44,
                                              fg_color=C["card2"], hover_color=C["border"],
                                              text_color=C["sub"],
                                              font=ctk.CTkFont(size=18),
                                              corner_radius=8,
                                              command=self._attach_file)
        self.btn_attach_file.pack(side="left", padx=(6, 0))

        if _VOICE_OK:
            self._build_voice_chat_buttons(ii)
        self.btn_send = ctk.CTkButton(ii, text=T("send"), width=90, height=44,
                                       fg_color=C["accent"], hover_color="#6457e0",
                                       font=ctk.CTkFont("Consolas", 13, "bold"),
                                       corner_radius=8, command=self._send)
        self.btn_send.pack(side="left", padx=(6, 0))
        self.btn_stop_gen = ctk.CTkButton(ii, text="⏹", width=44, height=44,
                                          fg_color="#3a1a1a", hover_color="#5a2020",
                                          text_color=C["red"],
                                          font=ctk.CTkFont("Consolas", 16, "bold"),
                                          corner_radius=8, state="disabled",
                                          command=self._stop_gen)
        self.btn_stop_gen.pack(side="left", padx=(6, 0))

        # Botón "Responder ya" (reasoning_end) — solo visible mientras el modelo piensa
        self.btn_reasoning_end = ctk.CTkButton(
            ii, text="⚡ Responder ya", width=130, height=44,
            fg_color=C["card2"], hover_color=C["accent"],
            text_color=C["accent2"], font=ctk.CTkFont("Segoe UI", 12, "bold"),
            corner_radius=8, command=self._force_reasoning_end
        )
        # No se hace .pack() aquí: se muestra/oculta dinámicamente desde _api()
        return f

    def _toggle_thinking(self):
        self.thinking_var.set(not self.thinking_var.get())
        if self.thinking_var.get():
            self.btn_thinking.configure(text=T("thinking_on"),
                                        fg_color=C["accent"],
                                        text_color="white")
        else:
            self.btn_thinking.configure(text=T("thinking_off"),
                                        fg_color=C["card2"],
                                        text_color=C["sub"])
        self.settings["toggle_thinking_show"] = self.thinking_var.get()
        save_settings(self.settings)

    def _toggle_enable_thinking(self):
        self.enable_thinking_var.set(not self.enable_thinking_var.get())
        if self.enable_thinking_var.get():
            self.btn_enable_thinking.configure(
                text=T("enable_thinking_on"),
                fg_color=C["card2"],
                text_color=C["accent2"],
            )
        else:
            self.btn_enable_thinking.configure(
                text=T("enable_thinking_off"),
                fg_color=C["card2"],
                text_color=C["yellow"],
            )
        self.settings["toggle_thinking_enable"] = self.enable_thinking_var.get()
        save_settings(self.settings)
        # Mostrar/ocultar slider de budget según estado del thinking
        if self.enable_thinking_var.get():
            self.reasoning_budget_frame.pack(side="right", padx=(0, 4), pady=10,
                                             before=self.btn_enable_thinking)
        else:
            self.reasoning_budget_frame.pack_forget()
        # Si el servidor esta corriendo, reiniciarlo para que el cambio
        # afecte a todos los clientes (chat, OpenClaw, etc.)
        if self.server_running:
            self._log(f"[{datetime.now():%H:%M:%S}] Reiniciando servidor (cambio thinking mode)...")
            self._restart_for_thinking()

    def _on_reasoning_budget_change(self, value):
        """Callback del slider de reasoning budget."""
        budget = int(value)
        self.reasoning_budget_var.set(budget)
        self.reasoning_budget_label.configure(text=f"Budget: {budget//1000}k")
        self.settings["reasoning_budget"] = budget
        save_settings(self.settings)

    def _restart_for_thinking(self):
        """Para el servidor y lo vuelve a arrancar con el nuevo chat_template_kwargs."""
        def _do():
            self.after(0, lambda: self.stop_server())
            for _ in range(40):
                time.sleep(0.25)
                if not self.server_running:
                    break
            time.sleep(0.5)
            self.after(0, lambda: self.start_server())
        threading.Thread(target=_do, daemon=True).start()

    def _toggle_websearch(self):
        self.websearch_var.set(not self.websearch_var.get())
        if self.websearch_var.get():
            self.btn_websearch.configure(
                text=T("web_on"),
                fg_color=C["green"] if C["green"] else "#4ade80",
                text_color="#0f0f13",
            )
        else:
            self.btn_websearch.configure(
                text=T("web_off"),
                fg_color=C["card2"],
                text_color=C["sub"],
            )
        self.settings["toggle_websearch"] = self.websearch_var.get()
        save_settings(self.settings)

    def _toggle_fstools(self):
        # Si se activa y no hay carpeta base configurada, pedirla primero
        if not self.fstools_var.get():
            base = self.settings.get("fs_base_dir", "")
            if not base or not os.path.isdir(base):
                chosen = filedialog.askdirectory(title="Elige la carpeta a la que el modelo tendra acceso")
                if not chosen:
                    return  # cancelado, no se activa
                self.settings["fs_base_dir"] = chosen
                save_settings(self.settings)

        self.fstools_var.set(not self.fstools_var.get())
        if self.fstools_var.get():
            self.btn_fstools.configure(
                text="\U0001f4c1 Files ON",
                fg_color=C["green"],
                text_color="#0f0f13",
            )
        else:
            self.btn_fstools.configure(
                text="\U0001f4c1 Files",
                fg_color=C["card2"],
                text_color=C["sub"],
            )
        self.settings["toggle_fstools"] = self.fstools_var.get()
        save_settings(self.settings)

    def _toggle_autocont(self):
        self.autocont_var.set(not self.autocont_var.get())
        if self.autocont_var.get():
            self.btn_autocont.configure(
                text="🔁 AutoCont ON",
                fg_color=C["yellow"],
                text_color="#0f0f13",
            )
        else:
            self.btn_autocont.configure(
                text="🔁 AutoCont",
                fg_color=C["card2"],
                text_color=C["sub"],
            )
        self.settings["toggle_autocont"] = self.autocont_var.get()
        save_settings(self.settings)

    def _stop_gen(self):
        self._stop_generation = True

    # ── Control de reasoning_end (botón "Responder ya") ──────────────────

    def _show_reasoning_end_btn(self):
        """Muestra el botón junto a ⏹ solo si el backend soporta el control."""
        if not self._reasoning_control_supported:
            return
        self.btn_reasoning_end.configure(state="normal", text="⚡ Responder ya")
        self.btn_reasoning_end.pack(side="left", padx=(6, 0))

    def _hide_reasoning_end_btn(self):
        self.btn_reasoning_end.pack_forget()

    def _force_reasoning_end(self):
        """Pide al servidor que corte el bloque de pensamiento en curso
        y pase a generar la respuesta final, sin abortar la generación."""
        gen_id = self._current_gen_id
        if not gen_id:
            self._log("[!] reasoning_end: no hay id de generación capturado todavía (¿backend no lo devuelve en el stream?)")
            return
        self.btn_reasoning_end.configure(state="disabled", text="Cortando...")
        port = self.settings.get("port", "8080")

        def _do():
            try:
                r = requests.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions/control",
                    json={"id": gen_id, "action": "reasoning_end"},
                    timeout=10
                )
                body_preview = (r.text or "")[:200]
                self._log(f"[reasoning_end] id={gen_id}  status={r.status_code}  body={body_preview!r}")
                if r.status_code == 404:
                    # El backend no tiene este endpoint (build antiguo / fork sin este PR)
                    self._reasoning_control_supported = False
                    self.after(0, lambda: (
                        self._log("[!] Tu backend no soporta reasoning_end "
                                   "(necesitas un llama-server con soporte de reasoning_control)."),
                        self._hide_reasoning_end_btn()
                    ))
                    return
                if not r.ok:
                    self._log(f"[!] reasoning_end devolvió {r.status_code}: {body_preview}")
            except Exception as e:
                self._log(f"[!] Error llamando a reasoning_end: {e}")
            finally:
                self.after(0, lambda: self.btn_reasoning_end.configure(
                    state="normal", text="⚡ Responder ya"))

        threading.Thread(target=_do, daemon=True).start()

    def _clear_chat(self):
        self.chat_history = []
        self._session_tokens = 0
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")

    def _enter_key(self, e):
        if not (e.state & 0x1):
            self._send(); return "break"

    def _on_image_drop(self, event):
        """Maneja el drop de un archivo de imagen sobre el input."""
        # event.data puede venir con llaves: {C:/ruta/imagen.png}
        path = event.data.strip().strip("{}")
        # En Windows a veces vienen múltiples archivos separados por espacio
        # Cogemos solo el primero
        if " " in path and not os.path.isfile(path):
            path = path.split()[0].strip("{}")
        ext = os.path.splitext(path)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
            self._attached_image = path
            fname = Path(path).name
            self.btn_attach.configure(text="🖼✓", fg_color=C["accent"], text_color="white")
            self._log(f"Imagen adjunta (drop): {fname}")
            # Feedback visual en el input
            current = self.chat_input.get("1.0", "end").strip()
            if not current:
                self.chat_input.insert("1.0", f"[{fname}] ")
        else:
            self._log(f"Formato no soportado para vision: {ext}")

    def _attach_image(self):
        """Abre selector de imagen para adjuntar al siguiente mensaje."""
        path = filedialog.askopenfilename(
            title=T("attach_image"),
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.gif *.webp *.bmp"),
                ("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"),
                ("Todos", "*.*")
            ]
        )
        if not path:
            return
        self._attached_image = path
        fname = Path(path).name
        # Feedback visual en el botón
        self.btn_attach.configure(
            text="🖼✓",
            fg_color=C["accent"],
            text_color="white",
        )
        self._log(f"Imagen adjunta: {fname}")

    def _clear_attached_image(self):
        """Limpia la imagen adjunta y restaura el botón."""
        self._attached_image = None
        self.btn_attach.configure(
            text="🖼",
            fg_color=C["card2"],
            text_color=C["sub"],
        )

    def _attach_file(self):
        """Abre selector — permite adjuntar varios archivos acumulándolos."""
        paths = filedialog.askopenfilenames(
            title=T("attach_files"),
            filetypes=[
                ("Archivos de código", "*.py *.js *.ts *.html *.css *.json *.bat *.sh *.txt *.md *.yaml *.yml *.toml *.ini *.cfg *.xml *.sql *.cpp *.c *.h *.rs *.go *.java *.cs *.php *.rb *.swift *.kt"),
                ("Python", "*.py"),
                ("HTML/CSS/JS", "*.html *.css *.js *.ts"),
                ("Texto", "*.txt *.md"),
                ("Todos", "*.*"),
            ]
        )
        if not paths:
            return
        for path in paths:
            # Evitar duplicados
            if any(f["path"] == path for f in self._attached_files):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    fcontent = fh.read()
            except Exception as e:
                messagebox.showerror(T("err_title"), f"{T('err_read')} {Path(path).name}:\n{e}")
                continue
            self._attached_files.append({
                "path": path, "name": Path(path).name,
                "ext": Path(path).suffix.lower(), "content": fcontent
            })
            self._log(f"Archivo adjunto: {Path(path).name} ({len(fcontent)} chars)")
        self._refresh_chips()

    def _refresh_chips(self):
        """Redibuja los chips de archivos adjuntos encima del input."""
        for w in self.chips_inner.winfo_children():
            w.destroy()
        if not self._attached_files:
            self.chips_bar.configure(height=0)
            return
        self.chips_bar.configure(height=34)
        for info in list(self._attached_files):
            fname = info["name"]
            short = fname if len(fname) <= 18 else fname[:15] + "…"
            chip = ctk.CTkFrame(self.chips_inner, fg_color=C["accent"],
                                corner_radius=6, height=24)
            chip.pack(side="left", padx=(0, 6))
            chip.pack_propagate(False)
            ctk.CTkLabel(chip, text=f"📎 {short}",
                         font=ctk.CTkFont("Consolas", 11),
                         text_color="white").pack(side="left", padx=(8, 2), pady=2)
            # X para quitar este archivo
            _info = info  # capture
            ctk.CTkButton(chip, text="×", width=20, height=20,
                          fg_color="transparent", hover_color="#6457e0",
                          text_color="white", font=ctk.CTkFont("Consolas", 13, "bold"),
                          command=lambda i=_info: self._remove_chip(i)
                          ).pack(side="left", padx=(0, 4))

    def _remove_chip(self, info):
        """Quita un archivo adjunto concreto."""
        self._attached_files = [f for f in self._attached_files if f["path"] != info["path"]]
        self._refresh_chips()

    def _clear_attached_file(self):
        """Limpia todos los archivos adjuntos."""
        self._attached_files = []
        self._refresh_chips()

    def _offer_download(self, code: str, suggested_name: str):
        """Añade un botón de descarga en el chat para el código generado."""
        # Necesitamos un frame embebido en el CTkTextbox — usamos una window embed de tk
        tb = self.chat_display._textbox
        ext = Path(suggested_name).suffix or ".txt"
        # Mapa de extensión a color de botón
        ext_colors = {
            ".py": "#3572A5", ".js": "#f1e05a", ".ts": "#3178c6",
            ".html": "#e34c26", ".css": "#563d7c", ".bat": "#4a4a4a",
            ".sh": "#89e051", ".json": "#292929", ".md": "#083fa1",
        }
        btn_color = ext_colors.get(ext, C["accent"])

        def _save(c=code, n=suggested_name):
            ext_filter = f"*{Path(n).suffix}" if Path(n).suffix else "*.*"
            save_path = filedialog.asksaveasfilename(
                initialfile=n,
                defaultextension=Path(n).suffix or ".txt",
                filetypes=[(f"Archivo {Path(n).suffix}", ext_filter), ("Todos", "*.*")],
                title=T("save_file_title"),
            )
            if save_path:
                try:
                    with open(save_path, "w", encoding="utf-8") as fh:
                        fh.write(c)
                    self._log(f"Archivo guardado: {save_path}")
                    messagebox.showinfo(T("saved_title"), T("saved_msg") + save_path)
                except Exception as e:
                    messagebox.showerror(T("err_title"), T("err_save") + str(e))

        btn = tk.Button(
            tb,
            text=f"  💾  Guardar  {suggested_name}  ",
            command=_save,
            bg=btn_color, fg="white",
            font=("Consolas", 11, "bold"),
            relief="flat", cursor="hand2",
            padx=8, pady=4,
            bd=0, highlightthickness=0,
        )
        self.chat_display.configure(state="normal")
        tb.insert("end", "\n")
        tb.window_create("end", window=btn)
        tb.insert("end", "\n\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def _send(self):
        if not self.server_running:
            messagebox.showwarning(T("warn_stopped"), T("warn_stopped_msg"))
            return
        txt = self.chat_input.get("1.0", "end").strip()
        if not txt and not self._attached_image and not self._attached_files:
            return
        self.chat_input.delete("1.0", "end")

        img_path   = self._attached_image
        files_list = list(self._attached_files)
        self._clear_attached_image()
        self._clear_attached_file()

        sp = self.sys_entry.get().strip() or self.current_prof.get("system_prompt", "")

        # Construir contenido del mensaje
        if img_path and os.path.isfile(img_path):
            ext = Path(img_path).suffix.lower().lstrip(".")
            mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                        "png": "image/png", "gif": "image/gif",
                        "webp": "image/webp", "bmp": "image/bmp"}
            mime = mime_map.get(ext, "image/jpeg")
            with open(img_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            user_content = [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": txt or "Describe esta imagen."},
            ]
            display_txt = "[\U0001f5bc " + Path(img_path).name + "]\n" + (txt or "Describe esta imagen.")

        elif files_list:
            # Uno o varios archivos adjuntos
            parts = []
            display_names = []
            for fi in files_list:
                fname  = fi["name"]
                fext   = fi["ext"]
                fcont  = fi["content"]
                lang   = fext.lstrip(".")
                n_lines = len(fcont.splitlines())
                display_names.append(fname)
                parts.append(
                    f"### Archivo: `{fname}` ({n_lines} lineas)\n"
                    f"```{lang}\n{fcont}\n```"
                )
            files_block = "\n\n".join(parts)
            multi = len(files_list) > 1
            instruction = (
                f"Se {'te han enviado los siguientes archivos' if multi else 'te ha enviado el siguiente archivo'}:\n\n"
                f"{files_block}\n\n"
                f"PROCESO QUE DEBES SEGUIR (obligatorio):\n"
                f"1. Lee cada archivo completo y comprende su estructura.\n"
                f"2. Localiza EXACTAMENTE las lineas o funciones que hay que tocar para cumplir la peticion.\n"
                f"3. Explica brevemente (1-3 lineas) que vas a cambiar y donde.\n"
                f"4. Devuelve {'cada archivo modificado en su propio bloque de codigo' if multi else 'el archivo COMPLETO con los cambios integrados en UN SOLO bloque de codigo'}.\n"
                f"\n"
                f"REGLAS ESTRICTAS:\n"
                f"- Modifica SOLO lo que la peticion requiere. No toques nada mas.\n"
                f"- No cambies nombres de variables, estilo, indentacion ni estructura del resto.\n"
                f"- Primera linea de cada bloque de codigo: # nombre_del_archivo\n"
                f"- NO truncques con '# ... resto igual ...' — devuelve SIEMPRE el archivo completo.\n"
                f"\n"
                f"Peticion: {txt or 'Revisa los archivos, detecta posibles errores o mejoras y aplicalos.'}"
            )
            user_content = instruction
            joined = ", ".join(display_names)
            display_txt = f"[\U0001f4ce {joined}]\n{txt or 'Revisa los archivos.'}"

        else:
            user_content = txt
            display_txt  = txt

        self._append("user", display_txt)

        msgs = ([{"role": "system", "content": sp}] if sp else []) + \
               self.chat_history + [{"role": "user", "content": user_content}]
        hist_txt = display_txt if files_list else (txt or "Describe esta imagen.")
        self.chat_history.append({"role": "user", "content": hist_txt})

        self._stop_generation = False
        self._current_gen_id = None
        self._hide_reasoning_end_btn()
        self.btn_send.configure(state="disabled", text=T("sending"))
        self.btn_stop_gen.configure(state="normal")
        use_web = self.websearch_var.get()
        use_fs = self.fstools_var.get()
        threading.Thread(target=self._api, args=(msgs, use_web, use_fs), daemon=True).start()

    def _check_code_blocks(self, response_text: str):
        """
        Busca bloques de codigo en la respuesta del modelo.
        Si encuentra alguno, muestra un boton de descarga en el chat.
        Detecta el nombre del archivo desde un comentario en la primera linea (# nombre.ext)
        o infiere la extension desde el lenguaje del bloque.
        """
        # Extensiones por lenguaje
        lang_ext = {
            "python": ".py", "py": ".py",
            "javascript": ".js", "js": ".js",
            "typescript": ".ts", "ts": ".ts",
            "html": ".html", "css": ".css",
            "bash": ".sh", "sh": ".sh", "shell": ".sh",
            "batch": ".bat", "bat": ".bat",
            "json": ".json", "yaml": ".yaml", "yml": ".yaml",
            "sql": ".sql", "xml": ".xml", "markdown": ".md", "md": ".md",
            "cpp": ".cpp", "c": ".c", "rust": ".rs", "go": ".go",
            "java": ".java", "csharp": ".cs", "cs": ".cs",
            "php": ".php", "ruby": ".rb", "swift": ".swift",
            "kotlin": ".kt", "toml": ".toml", "ini": ".ini",
        }
        pattern = r"```([a-zA-Z0-9_+-]*)\n(.*?)```"
        matches = list(re.finditer(pattern, response_text, re.DOTALL))
        offered = set()
        for m in matches:
            lang  = m.group(1).strip().lower()
            code  = m.group(2)
            if not code.strip() or lang in ("", "text", "plaintext"):
                continue
            ext = lang_ext.get(lang, f".{lang}" if lang else ".txt")
            # Intentar leer nombre desde primera linea: # nombre.ext o // nombre.ext
            first_line = code.split("\n")[0].strip()
            fname = None
            for prefix in ("#", "//", "--", ";"):
                if first_line.startswith(prefix):
                    candidate = first_line[len(prefix):].strip()
                    if "." in candidate and len(candidate) < 60 and " " not in candidate:
                        fname = candidate
                        # Quitar esa linea del codigo guardado
                        code = "\n".join(code.split("\n")[1:]).lstrip("\n")
                        break
            if not fname:
                fname = f"archivo_{len(offered)+1}{ext}"
            if fname in offered:
                continue
            offered.add(fname)
            self._offer_download(code, fname)

    def _append(self, role, text):
        self.chat_display.configure(state="normal")
        ts = datetime.now().strftime("%H:%M")
        self.chat_display.insert("end", f"\n[{ts}] {T('you') if role=='user' else T('model')}\n{text}\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def _insert_with_thinking(self, text, is_thinking):
        """Inserta texto en el chat con el tag correcto según si es thinking o respuesta."""
        tb = self.chat_display._textbox
        if is_thinking and self.thinking_var.get():
            tb.insert("end", text, "thinking")
        elif not is_thinking:
            tb.insert("end", text)
        # Si is_thinking pero thinking OFF → no insertar nada
        tb.see("end")

    # ── Web search helper (DuckDuckGo, sin API key) ──────────────────────

    def _web_search(self, query):
        """
        Busca en DuckDuckGo usando su endpoint HTML público.
        Devuelve string con los resultados para meter en el contexto.
        """
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            # DuckDuckGo instant answer API
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
                headers=headers, timeout=10
            )
            data = resp.json()
            results = []

            # Abstract (respuesta directa)
            if data.get("AbstractText"):
                results.append(f"Resumen: {data['AbstractText']}")
                if data.get("AbstractURL"):
                    results.append(f"Fuente: {data['AbstractURL']}")

            # RelatedTopics
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(f"- {topic['Text']}")

            if results:
                return "\n".join(results)

            # Fallback: DuckDuckGo HTML scrape ligero
            resp2 = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers=headers, timeout=10
            )
            import re as _re
            snippets = _re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', resp2.text, _re.DOTALL)
            snippets = [_re.sub(r'<[^>]+>', '', s).strip() for s in snippets[:5]]
            if snippets:
                return "\n".join(f"- {s}" for s in snippets if s)

            return "No se encontraron resultados para: " + query

        except Exception as e:
            return f"Error en búsqueda web: {e}"

    # ── File tools (acceso a carpeta local, confinado a fs_base_dir) ─────

    def _fs_safe_path(self, rel_path):
        """
        Resuelve rel_path dentro de la carpeta base configurada.
        Lanza ValueError si el resultado se sale de esa carpeta (path traversal).
        """
        base = Path(self.settings.get("fs_base_dir", "")).resolve()
        candidate = (base / rel_path).resolve() if rel_path not in (None, "", ".") else base
        if base not in candidate.parents and candidate != base:
            raise ValueError(f"Ruta fuera de la carpeta permitida: {rel_path}")
        return candidate

    def _fs_list_dir(self, rel_path=""):
        try:
            target = self._fs_safe_path(rel_path)
            if not target.exists():
                return f"No existe: {rel_path or '.'}"
            if not target.is_dir():
                return f"No es una carpeta: {rel_path or '.'}"
            entries = []
            for item in sorted(target.iterdir()):
                kind = "DIR " if item.is_dir() else "FILE"
                size = "" if item.is_dir() else f" ({item.stat().st_size} bytes)"
                entries.append(f"[{kind}] {item.name}{size}")
            return "\n".join(entries) if entries else "(carpeta vacia)"
        except Exception as e:
            return f"Error al listar: {e}"

    def _fs_create_dir(self, rel_path):
        try:
            target = self._fs_safe_path(rel_path)
            target.mkdir(parents=True, exist_ok=True)
            return f"Carpeta creada: {rel_path}"
        except Exception as e:
            return f"Error al crear carpeta: {e}"

    def _fs_move_file(self, src_rel, dst_rel):
        try:
            src = self._fs_safe_path(src_rel)
            dst = self._fs_safe_path(dst_rel)
            if not src.exists():
                return f"No existe el origen: {src_rel}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            return f"Movido: {src_rel} -> {dst_rel}"
        except Exception as e:
            return f"Error al mover: {e}"

    def _fs_write_file(self, rel_path, content):
        try:
            target = self._fs_safe_path(rel_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content or "", encoding="utf-8")
            return f"Archivo escrito: {rel_path} ({len(content or '')} caracteres)"
        except Exception as e:
            return f"Error al escribir archivo: {e}"

    def _fs_describe_image(self, rel_path, question=""):
        """
        Codifica la imagen en base64 y hace una llamada de vision aparte
        (single-turn, no streaming) contra el propio llama-server local.
        Devuelve la descripcion como texto, para usarla como resultado de tool call.
        """
        try:
            target = self._fs_safe_path(rel_path)
            if not target.is_file():
                return f"No existe el archivo: {rel_path}"
            ext = target.suffix.lower().lstrip(".")
            mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                        "png": "image/png", "gif": "image/gif",
                        "webp": "image/webp", "bmp": "image/bmp"}
            if ext not in mime_map:
                return f"Formato de imagen no soportado: {ext}"
            mime = mime_map[ext]
            with open(target, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()

            port = self.settings.get("port", "8080")
            prompt = question or "Describe esta imagen con detalle: contenido, personas, escena, colores, estilo."
            payload = {
                "model": "local",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                "temperature": 0.3,
                "max_tokens": 400,
                "stream": False,
            }
            resp = requests.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json=payload, timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error al analizar imagen: {e}"

    def _api(self, messages, use_web=False, use_fs=False):
        port = self.settings.get("port", "8080")
        p = self.current_prof
        full = ""
        full_raw = ""
        t_start = time.time()
        native_in_think = False
        in_think = False
        in_channel = False
        think_buf = ""
        timings = {}

        WEB_TOOL = [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Busca informacion actualizada en internet. Usala cuando el usuario pregunte sobre noticias recientes, eventos actuales, precios, o cualquier informacion que pueda haber cambiado desde tu entrenamiento.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "La consulta de busqueda"}
                    },
                    "required": ["query"]
                }
            }
        }]

        FS_BASE_LABEL = self.settings.get("fs_base_dir", "(sin configurar)")
        FS_TOOLS = [
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": f"Lista archivos y subcarpetas dentro de la carpeta permitida ({FS_BASE_LABEL}). Usa una ruta relativa a esa carpeta, vacia ('') para la raiz.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Ruta relativa a la carpeta base. Vacia para listar la raiz."}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "describe_image",
                    "description": "Analiza una imagen dentro de la carpeta permitida usando el modelo de vision y devuelve una descripcion en texto (contenido, escena, personas, colores, etc). Usala antes de decidir donde mover o como clasificar una imagen.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Ruta relativa de la imagen dentro de la carpeta base"},
                            "question": {"type": "string", "description": "Opcional: que quieres saber sobre la imagen en concreto"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_dir",
                    "description": "Crea una subcarpeta (y las carpetas intermedias necesarias) dentro de la carpeta permitida.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Ruta relativa de la carpeta a crear"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_file",
                    "description": "Mueve o renombra un archivo dentro de la carpeta permitida. Crea la carpeta destino si no existe.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "src": {"type": "string", "description": "Ruta relativa del archivo de origen"},
                            "dst": {"type": "string", "description": "Ruta relativa de destino (incluye nombre de archivo)"}
                        },
                        "required": ["src", "dst"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Crea o sobrescribe un archivo de texto (.txt, .md, etc) dentro de la carpeta permitida con el contenido indicado.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Ruta relativa del archivo a crear/sobrescribir, incluyendo extension"},
                            "content": {"type": "string", "description": "Contenido de texto a escribir en el archivo"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
        ]

        def _dispatch_fs_tool(name, args):
            if name == "list_dir":
                return self._fs_list_dir(args.get("path", ""))
            if name == "describe_image":
                return self._fs_describe_image(args.get("path", ""), args.get("question", ""))
            if name == "create_dir":
                return self._fs_create_dir(args.get("path", ""))
            if name == "move_file":
                return self._fs_move_file(args.get("src", ""), args.get("dst", ""))
            if name == "write_file":
                return self._fs_write_file(args.get("path", ""), args.get("content", ""))
            return f"Tool desconocida: {name}"

        try:
            current_messages = list(messages)
            max_tool_rounds = 5
            max_autocont_rounds = 10  # máximo de continuaciones automáticas
            _autocont_count = 0

            for tool_round in range(max_tool_rounds + max_autocont_rounds + 1):
                payload = {
                    "model": "local",
                    "messages": current_messages,
                    "temperature":    float(p.get("temperature", 0.7)),
                    "max_tokens":     int(p.get("max_tokens", 2048)) if int(p.get("max_tokens", 2048)) > 0 else -1,
                    "top_k":          int(p.get("top_k", 40)),
                    "top_p":          float(p.get("top_p", 0.95)),
                    "min_p":          float(p.get("min_p", 0.05)),
                    "repeat_penalty": float(p.get("repeat_penalty", 1.1)),
                    "seed":           int(p.get("seed", -1)),
                    "stream": True,
                    "chat_template_kwargs": {
                        "enable_thinking": bool(self.enable_thinking_var.get()),
                        **({"thinking_budget": self.reasoning_budget_var.get()} if self.enable_thinking_var.get() else {}),
                    },
                }
                if self.enable_thinking_var.get() and self._reasoning_control_supported:
                    payload["reasoning_control"] = True
                self._current_gen_id = None  # se rellena con el primer chunk de esta ronda
                if tool_round < max_tool_rounds:
                    active_tools = []
                    if use_web:
                        active_tools += WEB_TOOL
                    if use_fs and self.settings.get("fs_base_dir"):
                        active_tools += FS_TOOLS
                    if active_tools:
                        payload["tools"] = active_tools
                        payload["tool_choice"] = "auto"

                resp = requests.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json=payload, stream=True, timeout=180
                )

                ts = datetime.now().strftime("%H:%M")
                tool_calls_acc = {}
                finish_reason = None

                if tool_round == 0:
                    self.after(0, lambda t=ts: (
                        self.chat_display.configure(state="normal"),
                        self.chat_display.insert("end", f"\n[{t}] Modelo\n"),
                        self.chat_display.configure(state="disabled")
                    ))
                    time.sleep(0.05)

                for line in resp.iter_lines():
                    if self._stop_generation:
                        resp.close()
                        self.after(0, lambda: (
                            self.chat_display.configure(state="normal"),
                            self.chat_display.insert("end", "\n[generacion detenida]\n"),
                            self.chat_display.see("end"),
                            self.chat_display.configure(state="disabled")
                        ))
                        self.after(0, self._hide_reasoning_end_btn)
                        break
                    if not line: continue
                    line = line.decode()
                    if not line.startswith("data: "): continue
                    d = line[6:]
                    if d == "[DONE]": break
                    try:
                        chunk = json.loads(d)
                        if self._current_gen_id is None and chunk.get("id"):
                            self._current_gen_id = chunk["id"]
                        choice = chunk["choices"][0]
                        delta_obj = choice.get("delta", {})
                        delta = delta_obj.get("content", "")
                        r_delta = delta_obj.get("reasoning_content", "")
                        finish_reason = choice.get("finish_reason") or finish_reason

                        # Acumular tool calls fragmentados
                        for tc in delta_obj.get("tool_calls", []):
                            tidx = tc.get("index", 0)
                            if tidx not in tool_calls_acc:
                                tool_calls_acc[tidx] = {"id": "", "name": "", "arguments": ""}
                            fn = tc.get("function", {})
                            if fn.get("name"):      tool_calls_acc[tidx]["name"]      += fn["name"]
                            if fn.get("arguments"): tool_calls_acc[tidx]["arguments"] += fn["arguments"]
                            if tc.get("id"):        tool_calls_acc[tidx]["id"]         = tc["id"]

                        # reasoning_content nativo (Qwen, DeepSeek)
                        if r_delta:
                            if not native_in_think:
                                native_in_think = True
                                self.after(0, lambda: (
                                    self.chat_display.configure(state="normal"),
                                    self.chat_display._textbox.insert("end",
                                        "\n\U0001f4ad Pensando...\n", "think_hdr")
                                    if self.thinking_var.get() else None,
                                    self.chat_display.configure(state="disabled")
                                ))
                                self.after(0, self._show_reasoning_end_btn)
                            full_raw += r_delta
                            self.after(0, lambda x=r_delta: (
                                self.chat_display.configure(state="normal"),
                                self._insert_with_thinking(x, True),
                                self.chat_display.configure(state="disabled")
                            ))
                            continue

                        if native_in_think and not r_delta and delta:
                            native_in_think = False
                            self.after(0, lambda: (
                                self.chat_display.configure(state="normal"),
                                self.chat_display._textbox.insert("end",
                                    "\n─────────────────────\n\n", "think_hdr")
                                if self.thinking_var.get() else None,
                                self.chat_display.configure(state="disabled")
                            ))
                            self.after(0, self._hide_reasoning_end_btn)

                        if not delta:
                            if "timings" in chunk: timings = chunk["timings"]
                            continue

                        full_raw += delta

                        # Filtrar <|channel>thought<channel|> (Gemma4) — respeta toggle thinking
                        if "<|channel>" in delta or in_channel:
                            think_buf += delta
                            if not in_channel and "<|channel>" in think_buf:
                                before = think_buf[:think_buf.find("<|channel>")]
                                think_buf = think_buf[think_buf.find("<|channel>") + len("<|channel>"):]
                                in_channel = True
                                if before:
                                    full += before
                                    self.after(0, lambda x=before: (
                                        self.chat_display.configure(state="normal"),
                                        self._insert_with_thinking(x, False),
                                        self.chat_display.configure(state="disabled")
                                    ))
                                # Mostrar header "Pensando..." si toggle activo
                                self.after(0, lambda: (
                                    self.chat_display.configure(state="normal"),
                                    self.chat_display._textbox.insert("end",
                                        "\n\U0001f4ad Pensando...\n", "think_hdr")
                                    if self.thinking_var.get() else None,
                                    self.chat_display.configure(state="disabled")
                                ))
                                self.after(0, self._show_reasoning_end_btn)
                            if in_channel and "<channel|>" in think_buf:
                                idx_end = think_buf.find("<channel|>")
                                think_part = think_buf[:idx_end]
                                after = think_buf[idx_end + len("<channel|>"):]
                                think_buf = ""
                                in_channel = False
                                # Mostrar contenido del thinking si toggle activo
                                if think_part and self.thinking_var.get():
                                    self.after(0, lambda x=think_part: (
                                        self.chat_display.configure(state="normal"),
                                        self._insert_with_thinking(x, True),
                                        self.chat_display.configure(state="disabled")
                                    ))
                                # Cerrar header
                                self.after(0, lambda: (
                                    self.chat_display.configure(state="normal"),
                                    self.chat_display._textbox.insert("end",
                                        "\n─────────────────────\n\n", "think_hdr")
                                    if self.thinking_var.get() else None,
                                    self.chat_display.configure(state="disabled")
                                ))
                                self.after(0, self._hide_reasoning_end_btn)
                                if after:
                                    full += after
                                    self.after(0, lambda x=after: (
                                        self.chat_display.configure(state="normal"),
                                        self._insert_with_thinking(x, False),
                                        self.chat_display.configure(state="disabled")
                                    ))
                            elif in_channel and self.thinking_var.get():
                                # Mientras estamos dentro del channel, mostrar tokens como thinking
                                self.after(0, lambda x=delta: (
                                    self.chat_display.configure(state="normal"),
                                    self._insert_with_thinking(x, True),
                                    self.chat_display.configure(state="disabled")
                                ))
                            continue
                        think_buf += delta

                        # Parser <think> / </think>
                        while True:
                            if not in_think:
                                idx = think_buf.find("<think>")
                                if idx != -1:
                                    before = think_buf[:idx]
                                    think_buf = think_buf[idx + len("<think>"):]
                                    in_think = True
                                    if before:
                                        full += before
                                        self.after(0, lambda x=before: (
                                            self.chat_display.configure(state="normal"),
                                            self._insert_with_thinking(x, False),
                                            self.chat_display.configure(state="disabled")
                                        ))
                                    self.after(0, lambda: (
                                        self.chat_display.configure(state="normal"),
                                        self.chat_display._textbox.insert("end",
                                            "\n\U0001f4ad Pensando...\n", "think_hdr")
                                        if self.thinking_var.get() else None,
                                        self.chat_display.configure(state="disabled")
                                    ))
                                    self.after(0, self._show_reasoning_end_btn)
                                else:
                                    flush_idx = len(think_buf)
                                    for i in range(1, 7):
                                        if think_buf.endswith("<think>"[:i]):
                                            flush_idx = len(think_buf) - i
                                            break
                                    out = think_buf[:flush_idx]
                                    think_buf = think_buf[flush_idx:]
                                    if out:
                                        full += out
                                        self.after(0, lambda x=out: (
                                            self.chat_display.configure(state="normal"),
                                            self._insert_with_thinking(x, False),
                                            self.chat_display.configure(state="disabled")
                                        ))
                                    break
                            else:
                                idx = think_buf.find("</think>")
                                if idx != -1:
                                    think_part = think_buf[:idx]
                                    think_buf  = think_buf[idx + len("</think>"):]
                                    in_think   = False
                                    if think_part:
                                        self.after(0, lambda x=think_part: (
                                            self.chat_display.configure(state="normal"),
                                            self._insert_with_thinking(x, True),
                                            self.chat_display.configure(state="disabled")
                                        ))
                                    self.after(0, lambda: (
                                        self.chat_display.configure(state="normal"),
                                        self.chat_display._textbox.insert("end",
                                            "\n─────────────────────\n\n", "think_hdr")
                                        if self.thinking_var.get() else None,
                                        self.chat_display.configure(state="disabled")
                                    ))
                                    self.after(0, self._hide_reasoning_end_btn)
                                else:
                                    flush_idx = len(think_buf)
                                    for i in range(1, 8):
                                        if think_buf.endswith("</think>"[:i]):
                                            flush_idx = len(think_buf) - i
                                            break
                                    out = think_buf[:flush_idx]
                                    think_buf = think_buf[flush_idx:]
                                    if out:
                                        self.after(0, lambda x=out: (
                                            self.chat_display.configure(state="normal"),
                                            self._insert_with_thinking(x, True),
                                            self.chat_display.configure(state="disabled")
                                        ))
                                    break

                        if "timings" in chunk: timings = chunk["timings"]
                    except: pass

                # ── Procesar tool calls ───────────────────────────────────────
                if (use_web or use_fs) and tool_calls_acc and finish_reason == "tool_calls":
                    tc_list = []
                    for tidx in sorted(tool_calls_acc.keys()):
                        tc = tool_calls_acc[tidx]
                        tc_list.append({
                            "id": tc["id"] or f"call_{tidx}",
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]}
                        })
                    current_messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tc_list
                    })

                    FS_ICONS = {
                        "list_dir": "\U0001f4c2",
                        "describe_image": "\U0001f5bc",
                        "create_dir": "\U0001f4c1",
                        "move_file": "\U0001f4e6",
                        "write_file": "\U0001f4dd",
                    }

                    for tc in tc_list:
                        name = tc["function"]["name"]
                        try:
                            args = json.loads(tc["function"]["arguments"] or "{}")
                        except Exception:
                            args = {}

                        if name == "web_search":
                            query = args.get("query", "")
                            self.after(0, lambda q=query: (
                                self.chat_display.configure(state="normal"),
                                self.chat_display._textbox.insert("end",
                                    f"\n\U0001f310 Buscando: {q}\n", "think_hdr"),
                                self.chat_display.configure(state="disabled")
                            ))
                            result = self._web_search(query)
                            self._log(f"[Web] {query}")
                        else:
                            icon = FS_ICONS.get(name, "\U0001f527")
                            label = " ".join(str(v) for v in args.values()) if args else ""
                            self.after(0, lambda i=icon, n=name, l=label: (
                                self.chat_display.configure(state="normal"),
                                self.chat_display._textbox.insert("end",
                                    f"\n{i} {n}: {l}\n", "think_hdr"),
                                self.chat_display.configure(state="disabled")
                            ))
                            result = _dispatch_fs_tool(name, args)
                            self._log(f"[FS] {name}({args})")

                        current_messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"] or "call_0",
                            "content": result
                        })

                    # Reset para siguiente ronda
                    full = ""
                    full_raw = ""
                    in_think = False
                    native_in_think = False
                    think_buf = ""

                    self.after(0, lambda: (
                        self.chat_display.configure(state="normal"),
                        self.chat_display._textbox.insert("end",
                            "─────────────────────\n\n", "think_hdr"),
                        self.chat_display.configure(state="disabled")
                    ))
                    continue  # siguiente ronda del for tool_round

                # ── Auto-continuación si la respuesta se cortó por max_tokens ──
                if finish_reason == "length" and self.autocont_var.get() and not self._stop_generation and _autocont_count < max_autocont_rounds:
                    _autocont_count += 1
                    # Añadir respuesta parcial al historial de mensajes de esta llamada
                    current_messages.append({"role": "assistant", "content": full or full_raw})
                    current_messages.append({
                        "role": "user",
                        "content": "Tu respuesta se cortó por el límite de tokens. Continúa exactamente desde donde lo dejaste, sin repetir nada de lo ya dicho."
                    })
                    # Separador visual sutil en el chat
                    self.after(0, lambda: (
                        self.chat_display.configure(state="normal"),
                        self.chat_display._textbox.insert("end",
                            "\n↩ continuando...\n", "think_hdr"),
                        self.chat_display.configure(state="disabled")
                    ))
                    # Resetear acumuladores para la siguiente parte
                    full = ""
                    full_raw = ""
                    in_think = False
                    native_in_think = False
                    think_buf = ""
                    continue

                break  # sin tool calls y sin continuación, salir del bucle

            self.after(0, self._hide_reasoning_end_btn)
            # ── Stats finales ─────────────────────────────────────────────────
            elapsed = time.time() - t_start
            ctx_size = int(self.current_prof.get("ctx_size", 4096))
            if timings:
                tps      = timings.get("predicted_per_second", 0)
                pp_tps   = timings.get("prompt_per_second", 0)
                n_pred   = timings.get("predicted_n", 0)
                n_prompt = timings.get("prompt_n", 0)
                self._session_tokens += n_pred + n_prompt
                ctx_pct = min((self._session_tokens / ctx_size * 100) if ctx_size > 0 else 0, 100.0)
                stats = (
                    f"\n\u26a1 {tps:.1f} tok/s  |  {n_pred} tokens  |  "
                    f"prompt: {pp_tps:.0f} tok/s ({n_prompt} t)  |  {elapsed:.1f}s total  |  "
                    f"ctx: {self._session_tokens}/{ctx_size} ({ctx_pct:.1f}%)"
                )
            else:
                stats = f"\n\u26a1 {elapsed:.1f}s"

            self.after(0, lambda s=stats: (
                self.chat_display.configure(state="normal"),
                self.chat_display.insert("end", s + "\n"),
                self.chat_display.see("end"),
                self.chat_display.configure(state="disabled")
            ))
            self.chat_history.append({"role": "assistant", "content": full or full_raw})
            # Auto-guardar sesión tras cada respuesta
            self.after(0, self._save_current_session)
            # Detectar bloques de código y ofrecer descarga
            self.after(0, lambda r=full or full_raw: self._check_code_blocks(r))
            # TTS si la respuesta fue originada por voz
            if _VOICE_OK:
                self.after(0, lambda r=full or full_raw: self._vchat_speak_response(r))
            # TTS si la respuesta fue originada por voz
            if _VOICE_OK:
                self.after(0, lambda r=full or full_raw: self._vchat_speak_response(r))
        except Exception as e:
            self.after(0, lambda err=str(e): self._append("system", f"\u26a0 Error: {err}"))
        finally:
            self.after(0, lambda: (
                self.btn_send.configure(state="normal", text=T("send")),
                self.btn_stop_gen.configure(state="disabled"),
            ))
            self.after(0, self._hide_reasoning_end_btn)

    # ── Servidor tab ──────────────────────────────────────────────────────

    def _build_server(self, p):
        f = ctk.CTkFrame(p, fg_color=C["bg"], corner_radius=0)
        hdr = ctk.CTkFrame(f, fg_color=C["panel"], height=52, corner_radius=0)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text=T("server_tab_title"),
                     font=ctk.CTkFont("Consolas", 15, "bold"),
                     text_color=C["text"]).pack(side="left", padx=20)

        sc = ctk.CTkScrollableFrame(f, fg_color=C["bg"], corner_radius=0)
        sc.pack(fill="both", expand=True, padx=24, pady=16)

        def sec(t):
            ctk.CTkLabel(sc, text=t, font=ctk.CTkFont("Consolas", 13, "bold"),
                         text_color=C["accent2"]).pack(anchor="w", pady=(12, 0))
            ctk.CTkFrame(sc, height=1, fg_color=C["border"]).pack(fill="x", pady=(4, 8))

        sec(T("server_exe"))
        er = ctk.CTkFrame(sc, fg_color="transparent"); er.pack(fill="x", pady=4)
        self.sv_exe = tk.StringVar(value=self.settings.get("server_path", find_llama_server()))
        ctk.CTkEntry(er, textvariable=self.sv_exe, fg_color=C["input"],
                      text_color=C["text"], font=ctk.CTkFont("Consolas", 12),
                      height=34).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(er, text=T("browse"), width=90, height=34,
                       fg_color=C["card"], hover_color=C["border"],
                       font=ctk.CTkFont("Consolas", 12),
                       command=lambda: self.sv_exe.set(
                           filedialog.askopenfilename(filetypes=[("exe","*.exe"),("*","*.*")]) or self.sv_exe.get()
                       )).pack(side="left", padx=(8,0))

        sec(T("server_network"))
        self.sv_port = tk.StringVar(value=self.settings.get("port","8080"))
        self.sv_host = tk.StringVar(value=self.settings.get("host","127.0.0.1"))
        for lbl, var, ph in [(T("server_port_lbl"), self.sv_port, "8080"), (T("server_host_lbl"), self.sv_host, "127.0.0.1")]:
            r = ctk.CTkFrame(sc, fg_color="transparent"); r.pack(fill="x", pady=4)
            ctk.CTkLabel(r, text=lbl, font=ctk.CTkFont("Consolas", 12),
                         text_color=C["text"], width=160, anchor="w").pack(side="left")
            ctk.CTkEntry(r, textvariable=var, fg_color=C["input"],
                          text_color=C["text"], font=ctk.CTkFont("Consolas", 12),
                          height=32, width=200).pack(side="left", padx=8)

        ctk.CTkButton(sc, text=T("server_save"),
                       fg_color=C["accent"], hover_color="#6457e0",
                       font=ctk.CTkFont("Consolas", 12, "bold"),
                       height=38, command=self._save_srv
                       ).pack(anchor="w", pady=(16, 0))

        sec(T("server_cmd"))
        self.cmd_box = ctk.CTkTextbox(sc, height=80, fg_color=C["input"],
                                       text_color=C["accent2"],
                                       font=ctk.CTkFont("Consolas", 11),
                                       state="disabled")
        self.cmd_box.pack(fill="x")
        ctk.CTkButton(sc, text=T("server_cmd_preview"),
                       fg_color=C["card"], hover_color=C["border"],
                       font=ctk.CTkFont("Consolas", 11),
                       command=self._preview_cmd).pack(anchor="e", pady=4)

        # ── Sección Headless ──────────────────────────────────────────
        sec(T("server_headless_sec"))
        hl_card = ctk.CTkFrame(sc, fg_color=C["card"], corner_radius=10)
        hl_card.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(hl_card, text=T("server_headless_desc"),
                     font=ctk.CTkFont("Consolas", 11), text_color=C["sub"],
                     justify="left").pack(anchor="w", padx=14, pady=(10, 4))
        ctk.CTkLabel(hl_card, text=T("server_headless_cmd_label"),
                     font=ctk.CTkFont("Consolas", 11), text_color=C["text"]
                     ).pack(anchor="w", padx=14, pady=(4, 2))
        self.headless_cmd_box = ctk.CTkTextbox(hl_card, height=56, fg_color=C["input"],
                                                text_color=C["accent2"],
                                                font=ctk.CTkFont("Consolas", 11),
                                                corner_radius=6, state="normal")
        self.headless_cmd_box.pack(fill="x", padx=14, pady=(0, 4))
        self._update_headless_cmd()
        self.headless_cmd_box.configure(state="disabled")

        def _copy_hl_srv():
            txt = self.headless_cmd_box.get("1.0", "end").strip()
            f.clipboard_clear(); f.clipboard_append(txt)
        ctk.CTkButton(hl_card, text=T("copy"), width=80, height=24,
                       fg_color=C["card2"], hover_color=C["border"],
                       text_color=C["sub"], font=ctk.CTkFont("Consolas", 11),
                       corner_radius=6, command=_copy_hl_srv
                       ).pack(anchor="e", padx=14, pady=(0, 10))

        # ── Sección Limpiar Backups ───────────────────────────────────
        sec(T("cleanup_backups_sec"))
        cb_card = ctk.CTkFrame(sc, fg_color=C["card"], corner_radius=10)
        cb_card.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(cb_card, text=T("cleanup_backups_desc"),
                     font=ctk.CTkFont("Consolas", 11), text_color=C["sub"],
                     justify="left").pack(anchor="w", padx=14, pady=(10, 4))
        cb_status = ctk.CTkLabel(cb_card, text="",
                                  font=ctk.CTkFont("Consolas", 11), text_color=C["dim"])
        cb_status.pack(anchor="w", padx=14, pady=(0, 4))
        cb_btn_row = ctk.CTkFrame(cb_card, fg_color="transparent")
        cb_btn_row.pack(anchor="w", padx=14, pady=(0, 10))

        def _find_backups():
            dirs = [LLAMA_CPP_OFFICIAL_DIR, LLAMA_CPP_TURBOQUANT_DIR]
            found = []
            parent_dirs = set()
            for d in dirs:
                p = Path(d)
                parent_dirs.add(p.parent)
            for parent in parent_dirs:
                if parent.exists():
                    for item in parent.iterdir():
                        if "_backup_" in item.name and item.is_dir():
                            found.append(item)
            return found

        def _scan_backups():
            backups = _find_backups()
            if not backups:
                cb_status.configure(text=T("cleanup_backups_none"), text_color=C["dim"])
                del_btn.configure(state="disabled")
            else:
                total = sum(
                    sum(f.stat().st_size for f in b.rglob("*") if f.is_file())
                    for b in backups
                )
                size_str = f"{total/1_073_741_824:.1f} GB" if total > 1_073_741_824 else f"{total/1_048_576:.0f} MB"
                cb_status.configure(
                    text=T("cleanup_backups_found", n=len(backups), size=size_str),
                    text_color=C["yellow"]
                )
                del_btn.configure(state="normal")

        def _delete_backups():
            backups = _find_backups()
            if not backups:
                return
            if not messagebox.askyesno(
                T("cleanup_backups_sec"),
                f"¿Borrar {len(backups)} backup(s) permanentemente?\n\n" +
                "\n".join(str(b) for b in backups)
            ):
                return
            errors = []
            for b in backups:
                try:
                    shutil.rmtree(b)
                except Exception as e:
                    errors.append(str(e))
            if errors:
                messagebox.showerror(T("cleanup_backups_sec"), "\n".join(errors))
            else:
                cb_status.configure(text=T("cleanup_backups_done"), text_color=C["green"])
                del_btn.configure(state="disabled")

        ctk.CTkButton(cb_btn_row, text=T("cleanup_backups_scan"), height=30,
                       fg_color=C["card2"], hover_color=C["border"],
                       font=ctk.CTkFont("Consolas", 11),
                       command=_scan_backups).pack(side="left", padx=(0, 8))
        del_btn = ctk.CTkButton(cb_btn_row, text=T("cleanup_backups_del"), height=30,
                                 fg_color="#7a1a1a", hover_color="#5a1a1a",
                                 font=ctk.CTkFont("Consolas", 11),
                                 state="disabled",
                                 command=_delete_backups)
        del_btn.pack(side="left")

        # ── Sección Watchdog ──────────────────────────────────────────
        sec(T("watchdog_sec"))
        wd_card = ctk.CTkFrame(sc, fg_color=C["card"], corner_radius=10)
        wd_card.pack(fill="x", pady=(0, 8))
        wd_row = ctk.CTkFrame(wd_card, fg_color="transparent")
        wd_row.pack(fill="x", padx=14, pady=10)
        ctk.CTkLabel(wd_row, text=T("watchdog_auto_relaunch"),
                     font=ctk.CTkFont("Consolas", 12), text_color=C["text"]
                     ).pack(side="left")
        self._wd_var.set(self.settings.get("watchdog_auto_relaunch", False))
        def _on_wd_toggle():
            self.settings["watchdog_auto_relaunch"] = self._wd_var.get()
            save_settings(self.settings)
        ctk.CTkSwitch(wd_row, variable=self._wd_var, text="",
                       fg_color=C["input"], progress_color=C["accent"],
                       button_color=C["accent2"],
                       command=_on_wd_toggle).pack(side="right")
        return f

    def _update_headless_cmd(self):
        """Actualiza el comando headless mostrado en el tab Servidor."""
        if not hasattr(self, "headless_cmd_box"):
            return
        if self.current_model:
            cmd = f'python llamastation.py --no-gui --model "{self.current_model}"'
            port = self.settings.get("port", "8080")
            if port != "8080":
                cmd += f" --port {port}"
        else:
            cmd = T("server_headless_no_model")
        self.headless_cmd_box.configure(state="normal")
        self.headless_cmd_box.delete("1.0", "end")
        self.headless_cmd_box.insert("1.0", cmd)
        self.headless_cmd_box.configure(state="disabled")

    def _save_srv(self):
        self.settings["server_path"] = self.sv_exe.get()
        self.settings["port"]        = self.sv_port.get()
        self.settings["host"]        = self.sv_host.get()
        save_settings(self.settings)
        messagebox.showinfo(T("saved_title"), T("server_saved_ok"))

    def _build_cmd_list(self):
        exe   = self.settings.get("server_path","llama-server")
        port  = self.settings.get("port","8080")
        host  = self.settings.get("host","127.0.0.1")
        p     = self.current_prof
        args  = [exe]
        if self.current_model: args += ["-m", self.current_model]
        mmproj = str(p.get("mmproj", "")).strip()
        mmproj_disabled = bool(p.get("mmproj_disable", False))
        if mmproj and os.path.isfile(mmproj) and not mmproj_disabled: args += ["--mmproj", mmproj]
        args += [
            "--host", host, "--port", port,
            "-ngl",  str(int(p.get("gpu_layers",-1))),
            "-c",    str(int(p.get("ctx_size",4096))),
            "-b",    str(int(p.get("batch_size",512))),
            "-ub",   str(int(p.get("ubatch_size",512))),
            "-t",    str(int(p.get("threads",8))),
            "-tb",   str(int(p.get("threads_batch",8))),
            "-np",   str(int(p.get("max_concurrent",1))),  # peticiones paralelas
        ]
        if p.get("flash_attn"):      args += ["--flash-attn", "on"]
        # mmap: es ON por defecto en llama.cpp. Solo tiene sentido cuando hay capas en CPU.
        # Si todo está en GPU (ngl=-1), desactivarlo ahorra toda la RAM del mapeo del GGUF.
        if p.get("mmap"):
            pass  # llama.cpp default — no hace falta pasar nada
        else:
            args.append("--no-mmap")  # desmarcado → deshabilitar explícitamente
        if p.get("mlock"):           args.append("--mlock")
        if p.get("cont_batching"):   args.append("--cont-batching")
        if p.get("kv_cache_offload"):args.append("--kv-offload")
        if p.get("embeddings"):      args.append("--embeddings")
        # K y V cache pueden ser distintos (asimetría TurboQuant: q8_0-K + turbo3-V)
        kv_k = p.get("kv_type",   "f16")
        kv_v = p.get("kv_type_v", kv_k)  # fallback: igual que K si no hay V separado
        if kv_k != "f16": args += ["--cache-type-k", kv_k]
        if kv_v != "f16": args += ["--cache-type-v", kv_v]
        rb = float(p.get("rope_freq_base",0))
        rs = float(p.get("rope_freq_scale",0))
        if rb > 0: args += ["--rope-freq-base", str(rb)]
        if rs > 0: args += ["--rope-freq-scale", str(rs)]
        # Modo de ejecución CPU/Vulkan
        exec_mode = p.get("exec_mode", "gpu")
        if exec_mode == "cpu":
            # ngl ya se puso a 0 en el diálogo, pero forzamos por si acaso
            for i, a in enumerate(args):
                if a == "-ngl" and i+1 < len(args):
                    args[i+1] = "0"
        elif exec_mode == "vulkan":
            args += ["--device", "vulkan0"]

        # Multi-GPU: split mode y tensor split
        sm = p.get("split_mode", "layer")
        if sm == "none":
            args += ["--split-mode", "none"]
            args += ["--main-gpu", str(int(p.get("main_gpu", 0)))]
        elif sm:
            args += ["--split-mode", sm]
        ts = str(p.get("tensor_split", "")).strip()
        if ts:
            args += ["--tensor-split", ts]
        # TurboQuant fork: deshabilitar prompt cache para no saturar RAM
        # (el fork activa un cache de hasta 8 GB por defecto)
        if "llama-turboquant" in exe.replace("\\", "/"):
            args += ["--cache-ram", "0"]
        # Sampling — defaults del servidor para todos los clientes
        args += ["--temp",           str(round(float(p.get("temperature", 0.7)), 4))]
        args += ["--top-k",          str(int(p.get("top_k", 40)))]
        args += ["--top-p",          str(round(float(p.get("top_p", 0.95)), 4))]
        args += ["--min-p",          str(round(float(p.get("min_p", 0.05)), 4))]
        args += ["--repeat-penalty", str(round(float(p.get("repeat_penalty", 1.1)), 4))]
        args += ["--repeat-last-n",  str(int(p.get("repeat_last_n", 64)))]
        seed = int(p.get("seed", -1))
        if seed != -1: args += ["--seed", str(seed)]
        max_tok = int(p.get("max_tokens", 2048))
        if max_tok > 0: args += ["-n", str(max_tok)]
        # Deshabilitar reasoning_content en el stream para compatibilidad con
        # clientes OpenAI-compatible que no lo soportan (OpenClaw, etc.)
        args += ["--reasoning-format", "none"]
        # Thinking mode: inyectar chat_template_kwargs para que afecte a todos los clientes
        enable_thinking = getattr(self, "enable_thinking_var", None)
        if enable_thinking is not None:
            val = "true" if enable_thinking.get() else "false"
            if enable_thinking.get():
                _rbv = getattr(self, "reasoning_budget_var", None)
                budget = int(_rbv.get()) if _rbv is not None else 8000
                args += ["--chat-template-kwargs",
                         f'{{"enable_thinking":{val},"thinking_budget":{budget}}}']
            else:
                args += ["--chat-template-kwargs", f'{{"enable_thinking":{val}}}']
        ex = str(p.get("extra_args","")).strip()
        if ex: args.extend(ex.split())
        # Draft Model (especulación clásica con modelo pequeño)
        # NOTA: los binarios llama.cpp (oficial y forks AtomicChat/TurboQuant/MTP)
        # esperan "draft" como valor de --spec-type, no "draft-simple" (ese nombre
        # no existe en el binario y provoca "unknown speculative decoding type").
        _SPEC_TYPE_MAP = {"draft-simple": "draft", "draft-mtp": "mtp"}
        draft_model = str(p.get("draft_model", "")).strip()
        if draft_model and os.path.isfile(draft_model):
            draft_spec_raw = str(p.get("draft_spec_type", "draft-simple")).strip() or "draft-simple"
            draft_spec = _SPEC_TYPE_MAP.get(draft_spec_raw, draft_spec_raw)
            args += ["--model-draft", draft_model,
                     "--spec-type", draft_spec,
                     "-ngld", "99"]
        # MTP (Multi-Token Prediction) — solo backends compatibles
        # ✓ Oficial (llama.cpp master >= b3620), MTP (PR#22673), AtomicChat
        # ✗ TurboQuant (fork sin MTP), BeeLlama (usa dflash propio)
        _MTP_COMPATIBLE = ("llama-mtp", "llama-atomic", "llama.cpp")
        _MTP_INCOMPATIBLE = ("llama-turboquant", "llama-bee")
        exe_norm = exe.replace("\\", "/").lower()
        _backend_ok = (
            any(k in exe_norm for k in _MTP_COMPATIBLE) and
            not any(k in exe_norm for k in _MTP_INCOMPATIBLE)
        )
        if p.get("mtp_enabled") and _backend_ok:
            # MTP requiere exactamente -np 1
            for i, a in enumerate(args):
                if a == "-np" and i + 1 < len(args):
                    args[i + 1] = "1"
            args += ["--spec-type", "mtp",
                     "--spec-draft-n-max", str(int(p.get("mtp_draft_n_max", 6)))]
        return args

    def _preview_cmd(self):
        cmd = " ".join(self._build_cmd_list())
        self.cmd_box.configure(state="normal")
        self.cmd_box.delete("1.0","end")
        self.cmd_box.insert("1.0", cmd)
        self.cmd_box.configure(state="disabled")

    # ── Logs ──────────────────────────────────────────────────────────────

    def _build_logs(self, p):
        f = ctk.CTkFrame(p, fg_color=C["bg"], corner_radius=0)
        hdr = ctk.CTkFrame(f, fg_color=C["panel"], height=52, corner_radius=0)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text=T("logs_title"),
                     font=ctk.CTkFont("Consolas", 15, "bold"),
                     text_color=C["text"]).pack(side="left", padx=20)
        self.log_box = ctk.CTkTextbox(f, fg_color=C["panel"], text_color="#a8ff78",
                                       font=ctk.CTkFont("Consolas", 12),
                                       wrap="word", state="disabled", corner_radius=0)
        self.log_box.pack(fill="both", expand=True)
        ctk.CTkButton(hdr, text=T("logs_clear"), fg_color="transparent",
                       hover_color=C["card"], text_color=C["sub"],
                       font=ctk.CTkFont("Consolas", 11), width=70,
                       command=lambda: (self.log_box.configure(state="normal"),
                                        self.log_box.delete("1.0","end"),
                                        self.log_box.configure(state="disabled"))
                       ).pack(side="right", padx=16)
        return f

    def _log(self, t):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", t+"\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    # ── Info ──────────────────────────────────────────────────────────────

    def _build_info(self, p):
        f = ctk.CTkFrame(p, fg_color=C["bg"], corner_radius=0)
        hdr = ctk.CTkFrame(f, fg_color=C["panel"], height=52, corner_radius=0)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text=T("info_title"),
                     font=ctk.CTkFont("Consolas", 15, "bold"),
                     text_color=C["text"]).pack(side="left", padx=20)
        self.info_box = ctk.CTkTextbox(f, fg_color=C["panel"], text_color=C["text"],
                                        font=ctk.CTkFont("Consolas", 13),
                                        state="disabled", corner_radius=0)
        self.info_box.pack(fill="both", expand=True)
        return f

    def _refresh_info(self):
        if not self.server_running:
            self._iwrite(T("info_no_server")); return
        port = self.settings.get("port","8080")
        try:
            txt = json.dumps(requests.get(f"http://127.0.0.1:{port}/v1/models",timeout=5).json(), indent=2, ensure_ascii=False)
        except Exception as e:
            txt = f"Error: {e}"
        try:
            txt += "\n\n── Props ──\n" + json.dumps(requests.get(f"http://127.0.0.1:{port}/props",timeout=5).json(), indent=2, ensure_ascii=False)
        except: pass
        self._iwrite(txt)

    def _iwrite(self, t):
        self.info_box.configure(state="normal")
        self.info_box.delete("1.0","end")
        self.info_box.insert("1.0", t)
        self.info_box.configure(state="disabled")

    # ── VRAM meter ────────────────────────────────────────────────────────

    def _build_vram_bars(self):
        """Construye las barras de VRAM para cada GPU detectada."""
        for w in self.vram_frame.winfo_children():
            w.destroy()
        self._vram_bars = []

        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, **_NOWIN
            )
            if r.returncode != 0:
                return
            lines = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
        except Exception:
            return

        for line in lines:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3:
                continue
            idx, name, total_str = parts[0], parts[1], parts[2]
            try:
                total = int(total_str)
            except ValueError:
                continue

            short = name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")

            row = ctk.CTkFrame(self.vram_frame, fg_color="transparent")
            row.pack(fill="x", pady=(4, 0))

            ctk.CTkLabel(row, text=f"GPU{idx}",
                         font=ctk.CTkFont("Consolas", 9, "bold"),
                         text_color=C["sub"], width=36, anchor="w").pack(side="left")

            bar = ctk.CTkProgressBar(row, height=8, corner_radius=4,
                                     fg_color=C["card2"],
                                     progress_color=C["accent"])
            bar.set(0)
            bar.pack(side="left", fill="x", expand=True, padx=(4, 6))

            val_lbl = ctk.CTkLabel(row, text="—",
                                   font=ctk.CTkFont("Consolas", 9),
                                   text_color=C["sub"], width=70, anchor="e")
            val_lbl.pack(side="left")

            self._vram_bars.append({
                "idx": idx, "total": total,
                "bar": bar, "val": val_lbl
            })

    def _start_vram_poll(self):
        self._vram_running = True
        self._poll_vram()

    def _stop_vram_poll(self):
        self._vram_running = False
        if self._vram_timer:
            try: self.after_cancel(self._vram_timer)
            except Exception: pass
            self._vram_timer = None
        for entry in self._vram_bars:
            entry["bar"].set(0)
            entry["val"].configure(text="—", text_color=C["sub"])
            entry["bar"].configure(progress_color=C["accent"])

    def _poll_vram(self):
        if not getattr(self, "_vram_running", False):
            return

        def _fetch():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, **_NOWIN
                )
                if r.returncode != 0:
                    return {}
                result = {}
                for line in r.stdout.strip().splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) == 3:
                        try:
                            result[parts[0]] = (int(parts[1]), int(parts[2]))
                        except ValueError:
                            pass
                return result
            except Exception:
                return {}

        def _apply(data):
            if not getattr(self, "_vram_running", False):
                return
            for entry in self._vram_bars:
                info = data.get(entry["idx"])
                if not info:
                    continue
                used, total = info
                pct = used / total if total > 0 else 0
                entry["bar"].set(min(pct, 1.0))
                if pct >= 0.90:
                    color = C["red"]
                elif pct >= 0.70:
                    color = C["yellow"]
                else:
                    color = C["green"]
                entry["bar"].configure(progress_color=color)
                used_gb  = used  / 1024
                total_gb = total / 1024
                entry["val"].configure(
                    text=f"{used_gb:.1f}/{total_gb:.0f}GB",
                    text_color=color
                )
            if getattr(self, "_vram_running", False):
                self._vram_timer = self.after(2000, self._poll_vram)

        threading.Thread(
            target=lambda: self.after(0, lambda: _apply(_fetch())),
            daemon=True
        ).start()

    # ── Sonido de notificación ────────────────────────────────────────────

    def _play_ready_sound(self):
        """Sonido de Windows cuando el servidor está listo — solo si está activado."""
        if not self.settings.get("sound", True):
            return
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass

    # ── Servidor control ──────────────────────────────────────────────────

    def _log_gpu_info(self):
        """Muestra info de GPUs en los logs para diagnóstico."""
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8, **_NOWIN
            )
            if r.returncode == 0:
                lines = r.stdout.strip().splitlines()
                self.after(0, lambda: self._log("── GPUs detectadas ──"))
                for line in lines:
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) >= 5:
                        msg = f"  GPU{parts[0]}: {parts[1]}  |  VRAM total: {parts[2]} MiB  |  libre: {parts[3]} MiB  |  driver: {parts[4]}"
                    else:
                        msg = f"  {line}"
                    self.after(0, lambda m=msg: self._log(m))
                self.after(0, lambda: self._log("─────────────────────"))
            else:
                self.after(0, lambda: self._log("⚠ nvidia-smi no disponible o sin GPUs NVIDIA detectadas"))
        except FileNotFoundError:
            self.after(0, lambda: self._log("⚠ nvidia-smi no encontrado en PATH — ¿está instalado CUDA?"))
        except Exception as e:
            self.after(0, lambda err=str(e): self._log(f"⚠ Error GPU info: {err}"))

    def _on_backend_change(self):
        """Cambia el ejecutable al seleccionar otro backend en la sidebar."""
        bname = self.backend_var.get()
        bpath = BACKENDS.get(bname, "")
        if bpath and os.path.isfile(bpath):
            self.settings["server_path"] = bpath
            save_settings(self.settings)
            # Actualizar también el campo de la pestaña Servidor si ya está construida
            if hasattr(self, "sv_exe"):
                self.sv_exe.set(bpath)
            self._log(f"[{datetime.now():%H:%M:%S}] Backend: {bname}  →  {bpath}")
            self._detect_llama_version()
            # Resetear texto del botón de actualizar al cambiar de backend
            if hasattr(self, "btn_update_backend"):
                self.btn_update_backend.configure(
                    text=T("update_llama"),
                    fg_color=C["card2"], hover_color=C["border"],
                    text_color=C["sub"]
                )
        else:
            messagebox.showwarning(T("warn_backend_title"),
                f"No se encontró el ejecutable:\n{bpath}\n\nCompila primero el fork TurboQuant.")
            # Revertir selección al backend actual guardado
            cur = self.settings.get("server_path", "")
            for bn, bp in BACKENDS.items():
                if bp == cur:
                    self.backend_var.set(bn)
                    break

    def start_server(self):
        if not self.current_model:
            messagebox.showerror(T("err_no_model_title"), T("err_no_model")); return
        exe = self.settings.get("server_path","")
        if not exe or not os.path.isfile(exe):
            messagebox.showerror(T("err_server_title"), T("err_no_server")); return
        # Diagnóstico GPU antes de arrancar
        threading.Thread(target=self._log_gpu_info, daemon=True).start()
        args = self._build_cmd_list()
        self._log(f"[{datetime.now():%H:%M:%S}] $ {' '.join(args)}")
        try:
            self.server_process = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0
            )
        except FileNotFoundError:
            messagebox.showerror(T("err_title"), f"{T('err_not_found')}: {args[0]}"); return
        self._set_status(T("srv_starting"), C["yellow"])
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._log_proc = self.server_process   # referencia para _read_logs
        self._log_stop = threading.Event()     # señal de parada manual
        threading.Thread(target=self._read_logs, daemon=True).start()
        threading.Thread(target=self._wait_ready, daemon=True).start()

    def _read_logs(self):
        """Lee stdout linea a linea. Para limpiamente cuando stop_server()
        setea _log_stop y cierra/mata el proceso."""
        self._last_log_lines = []
        proc = self._log_proc   # referencia local al proceso activo
        try:
            while True:
                if self._log_stop.is_set():
                    break
                line = proc.stdout.readline()
                if line == "":   # EOF - proceso terminado
                    break
                l = line.rstrip()
                self._last_log_lines.append(l)
                if len(self._last_log_lines) > 30:
                    self._last_log_lines.pop(0)
                self.after(0, lambda x=l: self._log(x))
        except Exception:
            pass
        # Si fue parada manual _srv_exit ya fue llamado desde stop_server
        if not self._log_stop.is_set():
            try:
                rc = proc.wait(timeout=2)
            except Exception:
                rc = None
            self.after(0, lambda: self._srv_exit(rc))

    def _wait_ready(self):
        port = self.settings.get("port","8080")
        for _ in range(90):
            time.sleep(1)
            if not self.server_process or self.server_process.poll() is not None:
                return  # proceso ya muerto, _read_logs llamará _srv_exit
            try:
                if requests.get(f"http://127.0.0.1:{port}/health",timeout=2).status_code==200:
                    self.after(0, self._srv_ready); return
            except: pass

    def _srv_ready(self):
        self.server_running = True
        port = self.settings.get("port","8080")
        self._set_status(T("srv_running"), C["green"])
        self.port_label.configure(text=T("srv_port", port=port))
        self._log(f"[{datetime.now():%H:%M:%S}] ✓ Ready :{port}")
        self._update_headless_cmd()
        self._start_vram_poll()
        self._play_ready_sound()
        # Arrancar proxy Anthropic Messages API
        self._start_anthropic_proxy()

    def _start_anthropic_proxy(self):
        """Arranca el proxy /v1/messages en puerto+1 para Claude Code."""
        port = int(self.settings.get("port", 8080))
        proxy_port = port + 1
        openai_url = f"http://127.0.0.1:{port}"
        if self._proxy_server:
            self._proxy_server.stop()
        self._proxy_server = AnthropicProxyServer(openai_url, proxy_port)
        if self._proxy_server.start():
            self._log(f"[{datetime.now():%H:%M:%S}] ✓ Anthropic proxy :{proxy_port} (/v1/messages)")
        else:
            self._log(f"[{datetime.now():%H:%M:%S}] ⚠ Proxy Anthropic no pudo arrancar en :{proxy_port}")

    def _srv_exit(self, rc=None):
        was_running = self.server_running
        manual_stop = getattr(self, "_stopping", False)
        self._stopping = False
        self.server_running = False
        # Parar proxy Anthropic
        if self._proxy_server:
            self._proxy_server.stop()
            self._proxy_server = None
        self._set_status("Detenido", C["red"])
        self.port_label.configure(text=T("server_port"))
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self._stop_vram_poll()

        # Detectar crash: cualquier salida no-manual con código != 0 o None-inesperado
        is_crash = not manual_stop and (
            (was_running) or                          # crashó estando corriendo (fix bug anterior)
            (rc not in (None, 0, -1, -15, 1))         # falló al iniciar
        )

        if is_crash:
            last = getattr(self, "_last_log_lines", [])
            log_tail = "\n".join(last[-10:]) if last else "(sin logs)"
            rc_str = str(rc) if rc is not None else "?"
            self._log(f"[{datetime.now():%H:%M:%S}] 💥 Servidor caído (código {rc_str})")

            # ── Auto-relaunch watchdog ────────────────────────────────
            auto_relaunch = getattr(self, "_wd_var", None)
            if auto_relaunch and auto_relaunch.get() and self.current_model:
                DELAY = 5
                self._log(T("watchdog_relaunch_log", delay=DELAY))
                self._set_status(T("watchdog_relaunching"), C["yellow"])
                self.btn_start.configure(state="disabled")
                self.btn_stop.configure(state="disabled")
                def _do_relaunch():
                    # Solo relanzar si el usuario no arrancó/paró manualmente mientras esperábamos
                    if not self.server_running and not self.server_process:
                        self.start_server()
                self.after(DELAY * 1000, _do_relaunch)
            else:
                # Sin auto-relaunch: mostrar popup de aviso
                self.after(100, lambda r=rc_str, m=log_tail: messagebox.showerror(
                    T("watchdog_crashed_title"),
                    T("watchdog_crashed_msg", rc=r, log=m)
                ))

    def stop_server(self, wait=False):
        """Detiene el servidor correctamente liberando VRAM y RAM."""
        proc = self.server_process
        if not proc:
            return
        self.server_process = None
        self._stopping = True
        self._set_status("Deteniendo...", C["yellow"])
        self._log(f"[{datetime.now():%H:%M:%S}] Deteniendo servidor...")

        # Señalizar al hilo _read_logs que pare ANTES de matar el proceso
        if hasattr(self, "_log_stop"):
            self._log_stop.set()

        def _kill():
            try:
                proc.kill()            # kill directo en Windows es más fiable
                proc.wait(timeout=5)
            except Exception:
                pass
            # Cerrar stdout tras matar para desbloquear readline() si sigue bloqueado
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            self.after(0, lambda: (
                self._log(f"[{datetime.now():%H:%M:%S}] {T('srv_stopped_log')}"),
                self._srv_exit(None),
            ))

        threading.Thread(target=_kill, daemon=True).start()

    def _show_ready(self):
        """Muestra la ventana una vez que tkinter ha pintado todos los widgets."""
        self.deiconify()
        self.lift()
        self.focus_force()

    def _on_map(self, event):
        """
        Se dispara cada vez que la ventana se restaura desde minimizado.
        Forzamos un repintado completo para evitar el flash negro.
        """
        if event.widget is self:
            self.after(10,  self._force_redraw)
            self.after(80,  self._force_redraw)
            self.after(200, self._force_redraw)

    def _force_redraw(self):
        """Fuerza un repintado completo de la ventana (fix fondo negro customtkinter)."""
        try:
            w = self.winfo_width()
            h = self.winfo_height()
            if w <= 1 or h <= 1:
                return
            # Micro-resize: fuerza a Windows a repintar toda la ventana
            self.geometry(f"{w+1}x{h}")
            self.update_idletasks()
            self.geometry(f"{w}x{h}")
            self.update_idletasks()
        except Exception:
            pass

    def _on_close(self):
        """Cierre de ventana: mata el servidor antes de salir."""
        if self.server_process:
            try:
                self.server_process.kill()
                self.server_process.wait(timeout=4)
            except Exception:
                pass
        self.destroy()

    def _set_status(self, t, color):
        self.status_dot.configure(text_color=color)
        self.status_label.configure(text=t)

    def _check_server_on_start(self):
        # Restaurar último modelo usado
        last = self.settings.get("last_model", "")
        if last and os.path.isfile(last):
            self.current_model = last
            saved_prof = self.profiles.get(last, {})
            self.current_prof = {**DEFAULT_PROFILE, **saved_prof}
            self.model_label.configure(text=Path(last).name)
            sp = self.current_prof.get("system_prompt", "")
            if sp and hasattr(self, "sys_entry"):
                self.sys_entry.delete(0, "end")
                self.sys_entry.insert(0, sp)
            self._log(f"[{datetime.now():%H:%M:%S}] Modelo restaurado: {Path(last).name}")
        try:
            port = self.settings.get("port","8080")
            if requests.get(f"http://127.0.0.1:{port}/health",timeout=1).status_code==200:
                self.server_running = True
                self._set_status(T("srv_running_ext"), C["green"])
                self.port_label.configure(text=T("srv_port", port=port))
        except: pass

    def _open_update_dialog(self, backend_key=None):
        """Abre el modal de actualización para el backend indicado."""
        if backend_key is None:
            backend_key = "⚡ Oficial  (llama.cpp)"
        dlg = UpdateDialog(self, backend_key=backend_key)
        self.wait_window(dlg)
        self._detect_llama_version()

    def _toggle_theme(self):
        cur = self.settings.get("theme", "dark")
        new_theme = "light" if cur == "dark" else "dark"
        self.settings["theme"] = new_theme
        save_settings(self.settings)
        apply_theme(new_theme)
        # Reconstruir toda la UI con los nuevos colores
        sessions_backup = list(getattr(self, "_chat_sessions", []))
        cur_sid_backup  = getattr(self, "_current_session_id", None)
        self.configure(fg_color=C["bg"])
        for widget in self.winfo_children():
            widget.destroy()
        self.frames   = {}
        self.nav_btns = {}
        self._build_ui()
        self._chat_sessions = sessions_backup
        self._current_session_id = cur_sid_backup
        self._refresh_history_list()
        self._check_server_on_start()
        self._detect_llama_version()

    def _silent_update_check(self):
        """
        Al arrancar, consulta silenciosamente GitHub para cada backend.
        Si hay nueva versión disponible, resalta el botón correspondiente.
        """
        for bkey, bmeta in BACKEND_META.items():
            try:
                resp = requests.get(bmeta["api"],
                                    headers={"User-Agent": "LlamaStation-Updater"},
                                    timeout=10)
                if resp.status_code != 200:
                    continue
                tag = resp.json().get("tag_name", "")
                if not tag:
                    continue

                exe = os.path.join(bmeta["dir"], "llama-server.exe")
                if not os.path.isfile(exe):
                    self.after(0, lambda k=bkey, t=tag: self._mark_update_available(k, t))
                    continue

                r2 = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=5, **_NOWIN)
                ver_line = r2.stdout + r2.stderr

                m_tag = re.search(r"b(\d+)", tag)
                if m_tag:
                    # Oficial: comparar número de build
                    build_latest    = int(m_tag.group(1))
                    build_installed = _extract_build_number(ver_line)
                    if build_installed is None:
                        continue
                    if build_latest > build_installed:
                        self.after(0, lambda k=bkey, t=tag: self._mark_update_available(k, t))
                else:
                    # Tag semántico (tqp-v0.x.x): comparar con version.txt
                    version_file = os.path.join(bmeta["dir"], "llamastation_version.txt")
                    if os.path.isfile(version_file):
                        try:
                            saved_tag = open(version_file).read().strip()
                            if saved_tag != tag:
                                self.after(0, lambda k=bkey, t=tag: self._mark_update_available(k, t))
                        except Exception:
                            pass
                    else:
                        # Sin version.txt → instalado manualmente → amarillo
                        self.after(0, lambda k=bkey, t=tag: self._mark_update_available(k, t))
            except Exception:
                pass

    def _mark_update_available(self, backend_key, tag):
        """Pinta el botón del backend en amarillo si hay actualización disponible."""
        btn = self._update_btns.get(backend_key)
        if btn:
            meta = BACKEND_META.get(backend_key, {})
            btn.configure(
                text=f"⬆  {meta.get('label', backend_key)}  {tag}",
                fg_color=C["yellow"], hover_color="#d97706",
                text_color="#0f0f13",
            )

    def _detect_llama_version(self):
        exe_oficial = os.path.join(LLAMA_CPP_OFFICIAL_DIR, "llama-server.exe")
        exe = exe_oficial if os.path.isfile(exe_oficial) else self.settings.get("server_path", find_llama_server())
        if not exe: return
        try:
            r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=5, **_NOWIN)
            raw = r.stdout + r.stderr
            ver = _extract_version_label(raw)
            self.after(0, lambda: self.ver_label.configure(text=f"llama.cpp: {ver}"))
        except: pass



    # ── Model Browser ────────────────────────────────────────────────────

    def _open_model_browser(self):
        """Abre el modal para seleccionar modelo desde la carpeta de modelos."""
        models_dir = self.settings.get("models_dir", str(Path.home() / "models"))
        dlg = ModelBrowserDialog(self, models_dir, self.profiles, self.settings)
        self.wait_window(dlg)
        if dlg.result_path:
            path = dlg.result_path
            # Actualizar models_dir si cambió
            if dlg.result_dir:
                self.settings["models_dir"] = dlg.result_dir
                save_settings(self.settings)
                # Sincronizar con el tab downloader si existe
                if hasattr(self, "_downloader_frame"):
                    self._downloader_frame.dest_var.set(dlg.result_dir)
            load_dlg = LoadModelDialog(self, path, self.profiles)
            self.wait_window(load_dlg)
            if load_dlg.result is not None:
                self.current_model = path
                self.current_prof  = load_dlg.result
                self.model_label.configure(text=Path(path).name)
                self._log(f"[{datetime.now():%H:%M:%S}] Modelo: {Path(path).name}")
                self.settings["last_model"] = path
                save_settings(self.settings)
                sp = self.current_prof.get("system_prompt", "")
                if sp and hasattr(self, "sys_entry"):
                    self.sys_entry.delete(0, "end")
                    self.sys_entry.insert(0, sp)
                self._update_headless_cmd()

    # ── API Docs tab ─────────────────────────────────────────────────────

    def _build_api_docs(self, parent):
        """Tab de documentación de la API local — ejemplos listos para copiar."""
        f = ctk.CTkFrame(parent, fg_color=C["bg"], corner_radius=0)

        # Header
        hdr = ctk.CTkFrame(f, fg_color=C["panel"], height=52, corner_radius=0)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="📡  API Docs",
                     font=ctk.CTkFont("Consolas", 15, "bold"),
                     text_color=C["text"]).pack(side="left", padx=20, pady=14)
        ctk.CTkLabel(hdr, text="llama.cpp OpenAI-compatible API",
                     font=ctk.CTkFont("Consolas", 11),
                     text_color=C["sub"]).pack(side="left", padx=4)

        sc = ctk.CTkScrollableFrame(f, fg_color=C["bg"], corner_radius=0)
        sc.pack(fill="both", expand=True)

        def section(title, color=None):
            color = color or C["accent2"]
            ctk.CTkLabel(sc, text=title,
                         font=ctk.CTkFont("Consolas", 13, "bold"),
                         text_color=color).pack(anchor="w", padx=24, pady=(18, 0))
            ctk.CTkFrame(sc, height=1, fg_color=C["border"]).pack(fill="x", padx=24, pady=(4, 8))

        def endpoint_block(method, path, description, code_curl, code_python):
            card = ctk.CTkFrame(sc, fg_color=C["card"], corner_radius=10)
            card.pack(fill="x", padx=20, pady=(0, 10))

            # Título endpoint
            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=14, pady=(12, 4))

            method_color = C["green"] if method == "GET" else C["accent"]
            ctk.CTkLabel(top, text=f" {method} ",
                         font=ctk.CTkFont("Consolas", 11, "bold"),
                         fg_color=method_color, text_color="white",
                         corner_radius=4).pack(side="left")
            ctk.CTkLabel(top, text=f"  {path}",
                         font=ctk.CTkFont("Consolas", 12, "bold"),
                         text_color=C["text"]).pack(side="left")
            ctk.CTkLabel(top, text=description,
                         font=ctk.CTkFont("Consolas", 11),
                         text_color=C["sub"]).pack(side="left", padx=12)

            # Tabs curl / Python
            tab_frame = ctk.CTkFrame(card, fg_color="transparent")
            tab_frame.pack(fill="x", padx=14, pady=(4, 0))

            code_box = ctk.CTkTextbox(card, height=120, fg_color=C["input"],
                                       text_color=C["accent2"],
                                       font=ctk.CTkFont("Consolas", 11),
                                       corner_radius=6, state="normal")
            code_box.pack(fill="x", padx=14, pady=(0, 4))
            code_box.insert("1.0", code_curl)
            code_box.configure(state="disabled")

            active_tab = {"v": "curl"}

            def _switch(tab, cb=code_curl, cp=code_python, box=code_box, at=active_tab,
                        btn_curl=None, btn_py=None):
                at["v"] = tab
                content = cb if tab == "curl" else cp
                box.configure(state="normal")
                box.delete("1.0", "end")
                box.insert("1.0", content)
                box.configure(state="disabled")
                if btn_curl and btn_py:
                    btn_curl.configure(fg_color=C["accent"] if tab == "curl" else C["card2"],
                                       text_color="white" if tab == "curl" else C["sub"])
                    btn_py.configure(fg_color=C["accent"] if tab == "python" else C["card2"],
                                     text_color="white" if tab == "python" else C["sub"])

            btn_curl_ref = [None]
            btn_py_ref   = [None]

            def _make_switch_curl():
                _switch("curl", code_curl, code_python, code_box, active_tab,
                        btn_curl_ref[0], btn_py_ref[0])
            def _make_switch_py():
                _switch("python", code_curl, code_python, code_box, active_tab,
                        btn_curl_ref[0], btn_py_ref[0])

            bc = ctk.CTkButton(tab_frame, text="curl", width=60, height=24,
                                fg_color=C["accent"], text_color="white",
                                font=ctk.CTkFont("Consolas", 11),
                                corner_radius=6, command=_make_switch_curl)
            bc.pack(side="left", padx=(0, 4))
            btn_curl_ref[0] = bc

            bp = ctk.CTkButton(tab_frame, text="Python", width=60, height=24,
                                fg_color=C["card2"], text_color=C["sub"],
                                font=ctk.CTkFont("Consolas", 11),
                                corner_radius=6, command=_make_switch_py)
            bp.pack(side="left")
            btn_py_ref[0] = bp

            def _copy(box=code_box):
                txt = box.get("1.0", "end").strip()
                f.clipboard_clear(); f.clipboard_append(txt)

            ctk.CTkButton(tab_frame, text="📋 Copiar", width=80, height=24,
                           fg_color=C["card2"], hover_color=C["border"],
                           text_color=C["sub"], font=ctk.CTkFont("Consolas", 11),
                           corner_radius=6, command=_copy).pack(side="right")

        port_hint = self.settings.get("port", "8080")

        # ── Intro ──────────────────────────────────────────────────────
        intro = ctk.CTkFrame(sc, fg_color=C["card"], corner_radius=10)
        intro.pack(fill="x", padx=20, pady=(16, 4))
        ctk.CTkLabel(intro,
                     text=(f"  Tu servidor llama.cpp expone una API compatible con OpenAI en:\n"
                           f"  http://127.0.0.1:{port_hint}/v1\n\n"
                           f"  Cualquier app que use la API de OpenAI funciona apuntando a esa URL."),
                     font=ctk.CTkFont("Consolas", 12),
                     text_color=C["text"], justify="left"
                     ).pack(anchor="w", padx=14, pady=12)

        # ── Chat Completions ──────────────────────────────────────────
        section("💬  Chat Completions")
        endpoint_block(
            "POST", f"/v1/chat/completions", "Genera una respuesta de chat",
            f"""curl http://127.0.0.1:{port_hint}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "local",
    "messages": [
      {{"role": "system", "content": "Eres un asistente útil."}},
      {{"role": "user",   "content": "Hola, ¿cómo estás?"}}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }}'""",
            f"""from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:{port_hint}/v1",
    api_key="no-key-needed"
)

response = client.chat.completions.create(
    model="local",
    messages=[
        {{"role": "system", "content": "Eres un asistente útil."}},
        {{"role": "user",   "content": "Hola, ¿cómo estás?"}}
    ],
    temperature=0.7,
    max_tokens=512
)
print(response.choices[0].message.content)"""
        )

        # ── Streaming ─────────────────────────────────────────────────
        section("⚡  Streaming (respuesta en tiempo real)")
        endpoint_block(
            "POST", f"/v1/chat/completions", "Streaming SSE — tokens en tiempo real",
            f"""curl http://127.0.0.1:{port_hint}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "local",
    "messages": [{{"role": "user", "content": "Escribe un poema"}}],
    "stream": true
  }}'""",
            f"""from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:{port_hint}/v1",
    api_key="no-key-needed"
)

stream = client.chat.completions.create(
    model="local",
    messages=[{{"role": "user", "content": "Escribe un poema"}}],
    stream=True
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)"""
        )

        # ── List Models ───────────────────────────────────────────────
        section("📋  Modelos")
        endpoint_block(
            "GET", f"/v1/models", "Lista los modelos cargados",
            f"""curl http://127.0.0.1:{port_hint}/v1/models""",
            f"""import requests

resp = requests.get("http://127.0.0.1:{port_hint}/v1/models")
print(resp.json())"""
        )

        # ── Health ────────────────────────────────────────────────────
        section("🩺  Estado del servidor")
        endpoint_block(
            "GET", f"/health", "Comprueba si el servidor está listo",
            f"""curl http://127.0.0.1:{port_hint}/health""",
            f"""import requests

resp = requests.get("http://127.0.0.1:{port_hint}/health")
# {{"status": "ok"}} cuando está listo
print(resp.json())"""
        )

        # ── Anthropic Messages API (Claude Code) ─────────────────────
        section("🤖  Anthropic Messages API (Claude Code / SDK Anthropic)")
        ant_info = ctk.CTkFrame(sc, fg_color=C["card"], corner_radius=10)
        ant_info.pack(fill="x", padx=20, pady=(0, 4))
        proxy_port_hint = int(port_hint) + 1
        ctk.CTkLabel(ant_info,
                     text=(
                         "  LlamaStation incluye un proxy Anthropic-compatible en:\n"
                         f"  http://127.0.0.1:{proxy_port_hint}/v1/messages\n\n"
                         "  Usalo con Claude Code, el SDK oficial de Anthropic, o cualquier\n"
                         "  cliente que hable el formato Anthropic Messages API.\n"
                         "  Se activa automaticamente al arrancar el servidor."
                     ),
                     font=ctk.CTkFont("Consolas", 12),
                     text_color=C["text"], justify="left"
                     ).pack(anchor="w", padx=14, pady=12)

        endpoint_block(
            "POST", f"/v1/messages", "Anthropic Messages API — compatible con Claude Code",
            f"""# Variables de entorno para Claude Code
set ANTHROPIC_BASE_URL=http://127.0.0.1:{proxy_port_hint}
set ANTHROPIC_AUTH_TOKEN=llamastation
set ANTHROPIC_DEFAULT_OPUS_MODEL=tu-modelo.gguf
claude --dangerously-skip-permissions""",
            f"""import anthropic

client = anthropic.Anthropic(
    base_url="http://127.0.0.1:{proxy_port_hint}",
    api_key="llamastation"
)

message = client.messages.create(
    model="local",
    max_tokens=1024,
    messages=[
        {{"role": "user", "content": "Hola desde el SDK de Anthropic!"}}
    ]
)
print(message.content[0].text)"""
        )

        # ── Headless ──────────────────────────────────────────────────
        section("🖥️  Modo Headless (sin ventana)")
        hl_card = ctk.CTkFrame(sc, fg_color=C["card"], corner_radius=10)
        hl_card.pack(fill="x", padx=20, pady=(0, 10))
        hl_box = ctk.CTkTextbox(hl_card, height=100, fg_color=C["input"],
                                 text_color=C["accent2"],
                                 font=ctk.CTkFont("Consolas", 11),
                                 corner_radius=6, state="normal")
        hl_box.pack(fill="x", padx=14, pady=12)
        hl_txt = (
            "# Arrancar sin abrir la ventana — usa el perfil guardado del modelo\n"
            "python llamastation.py --no-gui --model C:\\models\\qwen3.gguf\n\n"
            "# Sobreescribir puerto y host\n"
            "python llamastation.py --no-gui --model C:\\models\\qwen3.gguf --port 8081 --host 0.0.0.0"
        )
        hl_box.insert("1.0", hl_txt)
        hl_box.configure(state="disabled")

        def _copy_hl():
            f.clipboard_clear(); f.clipboard_append(hl_box.get("1.0", "end").strip())
        ctk.CTkButton(hl_card, text="📋 Copiar", width=80, height=24,
                       fg_color=C["card2"], hover_color=C["border"],
                       text_color=C["sub"], font=ctk.CTkFont("Consolas", 11),
                       corner_radius=6, command=_copy_hl).pack(anchor="e", padx=14, pady=(0, 10))

        # Spacer final
        ctk.CTkFrame(sc, height=24, fg_color="transparent").pack()
        return f

    def _show_api_docs(self): self._show_frame("API Docs")

    # ── Downloader tab ───────────────────────────────────────────────────

    def _build_downloader(self, parent):
        try:
            from llamastation_downloader import ModelDownloaderFrame
            models_dir = self.settings.get("models_dir", str(Path.home() / "models"))
            frame = ModelDownloaderFrame(parent, colors=C, models_dir=models_dir)
            self._downloader_frame = frame
            return frame
        except ImportError:
            f = ctk.CTkFrame(parent, fg_color=C["bg"], corner_radius=0)
            ctk.CTkLabel(f,
                text=T("downloader_missing"),
                font=ctk.CTkFont("Consolas", 13), text_color=C["red"],
                justify="center").pack(expand=True)
            return f

    def _show_download(self):
        self._show_frame("Descargar")

    def _show_about(self):
        self._show_frame("Acerca de")

    def _build_about(self, parent):
        f = ctk.CTkFrame(parent, fg_color=C["bg"], corner_radius=0)
        sc = ctk.CTkScrollableFrame(f, fg_color=C["bg"], corner_radius=0)
        sc.pack(fill="both", expand=True, padx=0, pady=0)

        def _section(title):
            ctk.CTkFrame(sc, height=1, fg_color=C["border"]).pack(fill="x", padx=24, pady=(18, 0))
            ctk.CTkLabel(sc, text=title,
                         font=ctk.CTkFont("Consolas", 10, "bold"),
                         text_color=C["sub"]).pack(anchor="w", padx=28, pady=(6, 8))

        def _card():
            c = ctk.CTkFrame(sc, fg_color=C["card"], corner_radius=10)
            c.pack(fill="x", padx=20, pady=(0, 8))
            return c

        def _row(card, label, value, value_color=None):
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=4)
            ctk.CTkLabel(row, text=label,
                         font=ctk.CTkFont("Consolas", 12),
                         text_color=C["sub"], width=160, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=value,
                         font=ctk.CTkFont("Consolas", 12),
                         text_color=value_color or C["text"], anchor="w").pack(side="left")

        def _link_row(card, label, url):
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=4)
            ctk.CTkLabel(row, text=label,
                         font=ctk.CTkFont("Consolas", 12),
                         text_color=C["sub"], width=160, anchor="w").pack(side="left")
            btn = ctk.CTkButton(row, text=url,
                                fg_color="transparent", hover_color=C["card2"],
                                text_color=C["accent2"],
                                font=ctk.CTkFont("Consolas", 12),
                                anchor="w", height=24,
                                command=lambda u=url: __import__("webbrowser").open(u))
            btn.pack(side="left")

        # ── Header ────────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(sc, fg_color="transparent")
        hdr.pack(fill="x", padx=24, pady=(28, 4))
        ctk.CTkLabel(hdr, text="⚡ LlamaStation",
                     font=ctk.CTkFont("Consolas", 26, "bold"),
                     text_color=C["accent2"]).pack(side="left")
        ctk.CTkLabel(hdr, text=f"  {APP_VERSION}",
                     font=ctk.CTkFont("Consolas", 14),
                     text_color=C["dim"]).pack(side="left", pady=6)

        ctk.CTkLabel(sc, text="AI Model Workstation — llama.cpp GUI for Windows",
                     font=ctk.CTkFont("Consolas", 12),
                     text_color=C["sub"]).pack(anchor="w", padx=28, pady=(0, 4))

        # ── App info ──────────────────────────────────────────────────────
        _section("APLICACIÓN")
        c = _card()
        _row(c, "Versión",   APP_VERSION,  C["accent2"])
        _row(c, "Licencia",  "MIT — libre para uso personal y comercial", C["green"])
        _row(c, "Plataforma","Windows 10 / 11")
        _row(c, "Python",    "3.10+")
        ctk.CTkFrame(c, height=4, fg_color="transparent").pack()

        # ── Backends / créditos ───────────────────────────────────────────
        _section("BACKENDS Y CRÉDITOS")

        c2 = _card()
        ctk.CTkLabel(c2, text="⚡  llama.cpp  —  backend oficial",
                     font=ctk.CTkFont("Consolas", 13, "bold"),
                     text_color=C["text"]).pack(anchor="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(c2,
                     text="LLM inference engine en C/C++. Creado por Georgi Gerganov y la comunidad ggml-org.\n"
                          "Licencia MIT  ·  Copyright © 2023-2026 The ggml authors",
                     font=ctk.CTkFont("Consolas", 11),
                     text_color=C["sub"], justify="left", wraplength=_scale(640)).pack(anchor="w", padx=16, pady=(0, 6))
        _link_row(c2, "Repositorio", "https://github.com/ggml-org/llama.cpp")
        ctk.CTkFrame(c2, height=8, fg_color="transparent").pack()

        c3 = _card()
        ctk.CTkLabel(c3, text="🔬  llama-cpp-turboquant  —  fork TurboQuant (TheTom)",
                     font=ctk.CTkFont("Consolas", 13, "bold"),
                     text_color=C["text"]).pack(anchor="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(c3,
                     text="Fork experimental de llama.cpp con cuantización TurboQuant del KV cache (turbo2/3/4).\n"
                          "Permite contextos mucho más largos con mínima pérdida de calidad.\n"
                          "Basado en TurboQuant (arXiv:2504.19874, ICLR 2026) por Zirlin et al.\n"
                          "Licencia MIT  ·  Copyright © 2023-2026 The ggml authors / TheTom",
                     font=ctk.CTkFont("Consolas", 11),
                     text_color=C["sub"], justify="left", wraplength=_scale(640)).pack(anchor="w", padx=16, pady=(0, 6))
        _link_row(c3, "Repositorio", "https://github.com/TheTom/llama-cpp-turboquant")
        _link_row(c3, "Paper TurboQuant", "https://arxiv.org/abs/2504.19874")
        ctk.CTkFrame(c3, height=8, fg_color="transparent").pack()

        c4 = _card()
        ctk.CTkLabel(c4, text="⚛️  atomic-llama-cpp-turboquant  —  fork AtomicChat (TurboQuant + MTP)",
                     font=ctk.CTkFont("Consolas", 13, "bold"),
                     text_color=C["text"]).pack(anchor="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(c4,
                     text="Fork de llama.cpp con TurboQuant KV cache + especulación NextN (MTP nativo para Qwen3.x).\n"
                          "Permite combinar contextos largos y alta velocidad de generación.\n"
                          "Licencia MIT  ·  Copyright © 2023-2026 The ggml authors / AtomicBot-ai",
                     font=ctk.CTkFont("Consolas", 11),
                     text_color=C["sub"], justify="left", wraplength=_scale(640)).pack(anchor="w", padx=16, pady=(0, 6))
        _link_row(c4, "Repositorio", "https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant")
        ctk.CTkFrame(c4, height=8, fg_color="transparent").pack()

        c5 = _card()
        ctk.CTkLabel(c5, text="🐝  beellama.cpp  —  fork BeeLlama (DFlash + TurboQuant)",
                     font=ctk.CTkFont("Consolas", 13, "bold"),
                     text_color=C["text"]).pack(anchor="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(c5,
                     text="Fork de llama.cpp con TurboQuant KV cache + decodificación especulativa DFlash.\n"
                          "Experimental — ideal para modelos con soporte DFlash.\n"
                          "Licencia MIT  ·  Copyright © 2023-2026 The ggml authors / Anbeeld",
                     font=ctk.CTkFont("Consolas", 11),
                     text_color=C["sub"], justify="left", wraplength=_scale(640)).pack(anchor="w", padx=16, pady=(0, 6))
        _link_row(c5, "Repositorio", "https://github.com/Anbeeld/beellama.cpp")
        ctk.CTkFrame(c5, height=8, fg_color="transparent").pack()

        # ── Voz: dependencias ─────────────────────────────────────────────
        _section("MODO VOZ — DEPENDENCIAS")

        c_voice = _card()
        ctk.CTkLabel(c_voice, text="🎤  faster-whisper  —  reconocimiento de voz (STT)",
                     font=ctk.CTkFont("Consolas", 13, "bold"),
                     text_color=C["text"]).pack(anchor="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(c_voice,
                     text="Transcripción de voz a texto basada en Whisper de OpenAI, optimizada con CTranslate2.\n"
                          "Modelos disponibles: tiny · base · small · medium · large-v3\n"
                          "El modo voz siempre usa CPU (int8) para no competir con la GPU del LLM.",
                     font=ctk.CTkFont("Consolas", 11),
                     text_color=C["sub"], justify="left", wraplength=_scale(640)).pack(anchor="w", padx=16, pady=(0, 4))
        _link_row(c_voice, "Repositorio", "https://github.com/SYSTRAN/faster-whisper")
        ctk.CTkLabel(c_voice, text="pip install faster-whisper",
                     font=ctk.CTkFont("Consolas", 11),
                     text_color=C["accent2"]).pack(anchor="w", padx=16, pady=(2, 10))

        c_xtts = _card()
        ctk.CTkLabel(c_xtts, text="🗣️  Qwen3-TTS  —  síntesis de voz con clonación (TTS)",
                     font=ctk.CTkFont("Consolas", 13, "bold"),
                     text_color=C["text"]).pack(anchor="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(c_xtts,
                     text="Síntesis de voz multilingüe con clonación a partir de un clip de audio corto (~5s).\n"
                          "Modelos: Qwen3-TTS-12Hz-1.7B-Base (clonado de voz)\n"
                          "Corre como servidor HTTP en localhost:8777 (GPU 1 por defecto).\n"
                          "Requiere entorno conda 'qwen3tts'. Instalar con install_qwen3tts.bat.",
                     font=ctk.CTkFont("Consolas", 11),
                     text_color=C["sub"], justify="left", wraplength=_scale(640)).pack(anchor="w", padx=16, pady=(0, 4))
        _link_row(c_xtts, "Repositorio", "https://github.com/QwenLM/Qwen3-TTS")
        ctk.CTkLabel(c_xtts, text="pip install qwen-tts  (dentro del entorno conda qwen3tts)",
                     font=ctk.CTkFont("Consolas", 11),
                     text_color=C["accent2"]).pack(anchor="w", padx=16, pady=(2, 10))

        c_audio = _card()
        ctk.CTkLabel(c_audio, text="🔊  Audio — dependencias adicionales",
                     font=ctk.CTkFont("Consolas", 13, "bold"),
                     text_color=C["text"]).pack(anchor="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(c_audio,
                     text="sounddevice  —  captura y reproducción de audio en tiempo real\n"
                          "soundfile    —  lectura/escritura de archivos WAV\n"
                          "pydub        —  conversión de audio al importar clips (requiere ffmpeg en el PATH)\n"
                          "scipy        —  procesado de audio (resampling)",
                     font=ctk.CTkFont("Consolas", 11),
                     text_color=C["sub"], justify="left", wraplength=_scale(640)).pack(anchor="w", padx=16, pady=(0, 4))
        ctk.CTkLabel(c_audio,
                     text="pip install sounddevice soundfile pydub scipy",
                     font=ctk.CTkFont("Consolas", 11),
                     text_color=C["accent2"]).pack(anchor="w", padx=16, pady=(0, 10))

        # ── Licencia MIT completa ─────────────────────────────────────────
        _section("AVISO DE LICENCIA (MIT)")
        lc = _card()
        mit_text = (
            "MIT License\n\n"
            "LlamaStation — Copyright © 2024-2026\n\n"
            "Se concede permiso, de forma gratuita, a cualquier persona que obtenga una copia\n"
            "de este software y los archivos de documentación asociados, para utilizar el\n"
            "software sin restricciones, incluyendo sin limitación los derechos de usar, copiar,\n"
            "modificar, fusionar, publicar, distribuir, sublicenciar y/o vender copias del\n"
            "software, sujeto a las siguientes condiciones:\n\n"
            "El aviso de copyright anterior y este aviso de permiso deben incluirse en todas\n"
            "las copias o partes sustanciales del software.\n\n"
            "EL SOFTWARE SE PROPORCIONA «TAL CUAL», SIN GARANTÍA DE NINGÚN TIPO.\n\n"
            "──────────────────────────────────────────────\n"
            "Este software utiliza los siguientes proyectos bajo licencia MIT:\n"
            "llama.cpp · llama-cpp-turboquant · atomic-llama-cpp-turboquant · beellama.cpp\n"
            "MIT License — Copyright © 2023-2026 The ggml authors\n"
            "https://github.com/ggml-org/llama.cpp/blob/master/LICENSE"
        )
        tb = ctk.CTkTextbox(lc, fg_color=C["input"], text_color=C["sub"],
                             font=ctk.CTkFont("Consolas", 11),
                             height=220, corner_radius=8, wrap="word")
        tb.pack(fill="x", padx=14, pady=12)
        tb.insert("1.0", mit_text)
        tb.configure(state="disabled")

        def _copy_license():
            f.clipboard_clear(); f.clipboard_append(mit_text)
        ctk.CTkButton(lc, text="📋 Copiar licencia", width=140, height=28,
                       fg_color=C["card2"], hover_color=C["border"],
                       text_color=C["sub"], font=ctk.CTkFont("Consolas", 11),
                       corner_radius=6, command=_copy_license).pack(anchor="e", padx=14, pady=(0, 10))

        # Spacer
        ctk.CTkFrame(sc, height=24, fg_color="transparent").pack()
        return f

def _run_headless(model_path: str, port: str, host: str):
    """
    Modo headless: arranca llama-server sin GUI usando el perfil guardado del modelo.
    Ctrl+C para detener.
    """
    import signal

    profiles = load_profiles()
    settings = load_settings()
    if port:
        settings["port"] = port
    if host:
        settings["host"] = host

    key   = model_path
    saved = profiles.get(key, {})
    prof  = {**DEFAULT_PROFILE, **saved}

    exe = settings.get("server_path", find_llama_server())
    if not exe or not os.path.isfile(exe):
        print(f"[LlamaStation] ERROR: No se encontró llama-server.exe\n"
              f"  Configura la ruta en llamastation_settings.json → \"server_path\"")
        sys.exit(1)

    if not os.path.isfile(model_path):
        print(f"[LlamaStation] ERROR: Modelo no encontrado: {model_path}")
        sys.exit(1)

    # Construir args igual que la GUI (_build_cmd_list)
    p = prof
    args = [exe, "-m", model_path,
            "--host", settings.get("host", "127.0.0.1"),
            "--port", settings.get("port", "8080"),
            "-ngl",  str(int(p.get("gpu_layers", -1))),
            "-c",    str(int(p.get("ctx_size", 4096))),
            "-b",    str(int(p.get("batch_size", 512))),
            "-ub",   str(int(p.get("ubatch_size", 512))),
            "-t",    str(int(p.get("threads", 8))),
            "-tb",   str(int(p.get("threads_batch", 8))),
            "-np",   str(int(p.get("max_concurrent", 1))),
    ]
    if p.get("flash_attn"):     args += ["--flash-attn", "on"]
    if not p.get("mmap"):       args.append("--no-mmap")
    if p.get("mlock"):          args.append("--mlock")
    if p.get("cont_batching"):  args.append("--cont-batching")
    if p.get("kv_cache_offload"): args.append("--kv-offload")
    if p.get("embeddings"):     args.append("--embeddings")
    kv_k = p.get("kv_type", "f16")
    kv_v = p.get("kv_type_v", kv_k)
    if kv_k != "f16": args += ["--cache-type-k", kv_k]
    if kv_v != "f16": args += ["--cache-type-v", kv_v]
    sm = p.get("split_mode", "layer")
    if sm == "none":
        args += ["--split-mode", "none"]
        args += ["--main-gpu", str(int(p.get("main_gpu", 0)))]
    elif sm:
        args += ["--split-mode", sm]
    ts = str(p.get("tensor_split", "")).strip()
    if ts: args += ["--tensor-split", ts]
    if "llama-turboquant" in exe.replace("\\", "/"):
        args += ["--cache-ram", "0"]
    # Sampling — defaults del servidor para todos los clientes
    args += ["--temp",           str(round(float(p.get("temperature", 0.7)), 4))]
    args += ["--top-k",          str(int(p.get("top_k", 40)))]
    args += ["--top-p",          str(round(float(p.get("top_p", 0.95)), 4))]
    args += ["--min-p",          str(round(float(p.get("min_p", 0.05)), 4))]
    args += ["--repeat-penalty", str(round(float(p.get("repeat_penalty", 1.1)), 4))]
    args += ["--repeat-last-n",  str(int(p.get("repeat_last_n", 64)))]
    seed = int(p.get("seed", -1))
    if seed != -1: args += ["--seed", str(seed)]
    max_tok = int(p.get("max_tokens", 2048))
    if max_tok > 0: args += ["-n", str(max_tok)]
    # Deshabilitar reasoning_content en el stream para compatibilidad con
    # clientes OpenAI-compatible que no lo soportan (OpenClaw, etc.)
    args += ["--reasoning-format", "none"]
    ex = str(p.get("extra_args", "")).strip()
    if ex: args.extend(ex.split())
    # Draft Model (BeeLlama DFlash speculative decoding)
    # Mismo mapeo que en el otro punto de construcción de args: "draft-simple"
    # no es un valor real de --spec-type en los binarios llama.cpp.
    _SPEC_TYPE_MAP = {"draft-simple": "draft", "draft-mtp": "mtp"}
    draft_model = str(p.get("draft_model", "")).strip()
    if draft_model and os.path.isfile(draft_model):
        draft_spec_raw = str(p.get("draft_spec_type", "draft-simple")).strip() or "draft-simple"
        draft_spec = _SPEC_TYPE_MAP.get(draft_spec_raw, draft_spec_raw)
        args += ["--model-draft", draft_model,
                 "--spec-type", draft_spec,
                 "-ngld", "99"]
    # MTP (Multi-Token Prediction)
    _MTP_COMPATIBLE = ("llama-mtp", "llama-atomic", "llama.cpp")
    _MTP_INCOMPATIBLE = ("llama-turboquant", "llama-bee")
    exe_norm = exe.replace("\\", "/").lower()
    _backend_ok = (
        any(k in exe_norm for k in _MTP_COMPATIBLE) and
        not any(k in exe_norm for k in _MTP_INCOMPATIBLE)
    )
    if p.get("mtp_enabled") and _backend_ok:
        for i, a in enumerate(args):
            if a == "-np" and i + 1 < len(args):
                args[i + 1] = "1"
        args += ["--spec-type", "mtp",
                 "--spec-draft-n-max", str(int(p.get("mtp_draft_n_max", 6)))]
    model_name = Path(model_path).name
    pport = settings.get("port", "8080")
    phost = settings.get("host", "127.0.0.1")

    print(f"""
╔══════════════════════════════════════════════════════╗
║  ⚡ LlamaStation  —  Modo Headless                     ║
╚══════════════════════════════════════════════════════╝
  Modelo  : {model_name}
  Servidor: http://{phost}:{pport}
  Perfil  : {'guardado' if key in profiles else 'por defecto'}
  Cmd     : {' '.join(args[:6])} ...

  Presiona Ctrl+C para detener.
""")

    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, **_NOWIN)

    stop_flag = threading.Event()

    def _read():
        for line in proc.stdout:
            if stop_flag.is_set():
                break
            print(line, end="", flush=True)

    threading.Thread(target=_read, daemon=True).start()

    def _signal_handler(sig, frame):
        print("\n[LlamaStation] Deteniendo servidor...")
        stop_flag.set()
        proc.kill()
        proc.wait()
        print("[LlamaStation] Servidor detenido. VRAM liberada.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    proc.wait()
    print(f"[LlamaStation] Proceso terminado con código {proc.returncode}")


if __name__ == "__main__":
    # ── Modo headless ─────────────────────────────────────────────────────
    # Uso: python llamastation.py --no-gui --model ruta/modelo.gguf [--port 8080] [--host 127.0.0.1]
    import argparse
    parser = argparse.ArgumentParser(prog="LlamaStation", add_help=True)
    parser.add_argument("--no-gui",  action="store_true", help="Arranca sin ventana gráfica")
    parser.add_argument("--model",   type=str, default="",    help="Ruta al modelo GGUF")
    parser.add_argument("--port",    type=str, default="",    help="Puerto del servidor (sobreescribe settings)")
    parser.add_argument("--host",    type=str, default="",    help="Host del servidor (sobreescribe settings)")
    args = parser.parse_args()

    if args.no_gui:
        if not args.model:
            # ── Menú interactivo de selección de modelo ────────────────
            settings = load_settings()
            profiles = load_profiles()
            models_dir = settings.get("models_dir", str(Path.home() / "models"))

            # Buscar todos los .gguf en la carpeta de modelos (recursivo 1 nivel)
            gguf_files = []
            if os.path.isdir(models_dir):
                for entry in sorted(Path(models_dir).rglob("*.gguf")):
                    gguf_files.append(str(entry))

            if not gguf_files:
                print(f"[LlamaStation] No se encontraron modelos .gguf en: {models_dir}")
                print(f"  Configura la carpeta en LlamaStation → Servidor → models_dir")
                print(f"  O usa: python llamastation.py --no-gui --model <ruta_completa.gguf>")
                sys.exit(1)

            print("""
╔══════════════════════════════════════════════════════╗
║  ⚡ LlamaStation  —  Selecciona modelo                 ║
╚══════════════════════════════════════════════════════╝""")

            for i, path in enumerate(gguf_files, 1):
                name = Path(path).name
                has_profile = path in profiles
                tag = "  (perfil guardado ✓)" if has_profile else "  (sin perfil, usará defaults)"
                print(f"  {i}. {name}{tag}")

            print(f"\n  0. Cancelar")
            print()

            while True:
                try:
                    choice = input("  Elige número: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\n[LlamaStation] Cancelado.")
                    sys.exit(0)

                if choice == "0":
                    print("[LlamaStation] Cancelado.")
                    sys.exit(0)

                if choice.isdigit() and 1 <= int(choice) <= len(gguf_files):
                    selected = gguf_files[int(choice) - 1]
                    break

                print(f"  ⚠ Número no válido. Elige entre 1 y {len(gguf_files)}, o 0 para cancelar.")

            _run_headless(selected, args.port, args.host)
        else:
            _run_headless(args.model, args.port, args.host)
    else:
        app = LlamaStation()
        app.mainloop()
