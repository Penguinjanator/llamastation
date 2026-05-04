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

# Flag para ocultar ventanas de consola en Windows al lanzar subprocesos
_NOWIN = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}

ctk.set_default_color_theme("blue")

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

APP_VERSION = "v2.3.0"

DEFAULT_PROFILE = {
    "gpu_layers": -1, "threads": 8, "threads_batch": 8,
    "ctx_size": 4096, "batch_size": 512, "ubatch_size": 512, "max_concurrent": 1,
    "temperature": 0.7, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
    "repeat_penalty": 1.1, "repeat_last_n": 64, "max_tokens": 2048, "seed": -1,
    "flash_attn": True, "mmap": False, "mlock": False, "cont_batching": True,
    "kv_cache_offload": True, "keep_in_memory": False, "embeddings": False,
    "kv_type": "f16", "kv_type_v": "f16",
    "rope_freq_base": 0.0, "rope_freq_scale": 0.0,
    "extra_args": "", "system_prompt": "", "mmproj": "",
    "split_mode": "layer", "tensor_split": "",
}

BACKENDS = {
    "⚡ Oficial  (llama.cpp)": r"C:\llama.cpp\llama-server.exe",
    "🔬 TurboQuant  (TheTom fork)": r"C:\llama-turboquant\llama-server.exe",
}


def find_llama_server():
    # Prefer TurboQuant if available, else official
    for c in [r"C:\llama-turboquant\llama-server.exe",
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
        self.geometry("700x860")
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
            wraplength=560, justify="left"
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

        ctk.CTkLabel(c,
            text=T("tensor_tip"),
            font=ctk.CTkFont("Consolas", 11), text_color=C["sub"],
            wraplength=560, justify="left"
        ).pack(anchor="w", padx=16, pady=(6, 4))
        ts_var = tk.StringVar(value="")
        self._vars["tensor_split"] = ts_var
        ctk.CTkEntry(c, textvariable=ts_var,
                      fg_color=C["input"], text_color=C["accent2"],
                      font=ctk.CTkFont("Consolas", 12),
                      placeholder_text=T("tensor_ph"),
                      height=32).pack(fill="x", padx=16, pady=(0, 12))

    def _sec_cpu_ram(self, s):
        self._title(s, T("sec_cpu"))
        c = self._card(s)
        ctk.CTkLabel(c,
            text=T("cpu_mode_tip"),
            font=ctk.CTkFont("Consolas", 11), text_color=C["sub"],
            wraplength=560, justify="left"
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
            wraplength=560, justify="left"
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
                     wraplength=560, justify="left"
                     ).pack(anchor="w", padx=16, pady=(10, 2))

        for cache_key, cache_lbl, tip_txt in [
            ("kv_type",   "Cache-K:",   "K-cache: usa q8_0 para máxima calidad, turbo3/4 para máximo ahorro VRAM"),
            ("kv_type_v", "Cache-V:",   "V-cache: turbo3 tiene mínima pérdida de calidad — ideal para comprimir"),
        ]:
            ctk.CTkLabel(c, text=f"  {cache_lbl}  {tip_txt}",
                         font=ctk.CTkFont("Consolas", 10), text_color=C["dim"],
                         wraplength=560, justify="left"
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
                     wraplength=560, justify="left"
                     ).pack(anchor="w", padx=16, pady=(8, 4))

        self._mmproj_status_label = ctk.CTkLabel(c,
            text=T("mmproj_none"),
            font=ctk.CTkFont("Consolas", 10), text_color=C["dim"],
            wraplength=560, justify="left"
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
                      height=36).pack(fill="x", padx=16, pady=(0, 16))

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
        self.geometry("620x500")
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
        try:
            resp = requests.get(api_url,
                                headers={"User-Agent": "LlamaStation-Updater"},
                                timeout=20)
            resp.raise_for_status()
            release = resp.json()
        except Exception as e:
            self.after(0, lambda: self._log(f"✗ Error al contactar GitHub: {e}"))
            self.after(0, lambda: self._set_progress(0, "Error de conexión"))
            return

        tag = release.get("tag_name", "desconocida")
        assets = release.get("assets", [])
        self.after(0, lambda: self.lbl_latest.configure(text=tag))
        self.after(0, lambda: self._log(f"  Última release: {tag}"))

        # Buscar assets — usar asset_fn custom si existe, si no el estándar
        asset_fn = self._meta.get("asset_fn")
        if asset_fn:
            asset  = asset_fn(assets, cuda_mm, cuda_maj)
            cudart = None  # TheTom no tiene cudart separado
        else:
            asset  = _find_best_asset(assets, cuda_mm, cuda_maj)
            cudart = _find_cudart_asset(assets, cuda_mm, cuda_maj)

        if not asset:
            self.after(0, lambda: self._log("✗ No se encontró asset compatible para tu CUDA/Windows/x64"))
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
                    self.after(0, lambda: self._log(f"  ⚠ Backup falló: {e} (continuando de todos modos)"))

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
            self.after(0, lambda: self._log(f"\n✗ Error durante la instalación: {e}"))
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
        self.geometry("680x580")
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
                     wraplength=420, justify="left").pack(anchor="w")

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

        # Click en toda la fila también selecciona
        for w in [row, inner, left_col]:
            w.bind("<Button-1>", lambda e, path=m["path"]: self._select(path))
            w.bind("<Enter>", lambda e, r=row: r.configure(fg_color=C["card2"]))
            w.bind("<Leave>", lambda e, r=row: r.configure(fg_color=C["card"]))

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

class LlamaStation(ctk.CTk):
    def __init__(self):
        self.profiles         = load_profiles()
        self.settings         = load_settings()
        # Aplicar tema e idioma guardados ANTES de construir la UI
        apply_theme(self.settings.get("theme", "dark"))
        set_lang(self.settings.get("lang", "es"))

        super().__init__()
        self.title("LlamaStation")
        self.geometry("1300x840")
        self.minsize(1000, 700)

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
        self.chat_history     = []
        self.current_model    = ""
        self.current_prof     = dict(DEFAULT_PROFILE)
        self._stop_generation = False   # flag para abortar generación
        self._session_tokens  = 0       # tokens de contexto acumulados en sesión
        self._stopping        = False   # flag de parada manual del servidor
        self._attached_image  = None    # ruta de imagen adjunta para vision
        self._attached_files  = []      # lista de archivos de texto adjuntos (py, html, etc.)

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
        self.left_sidebar = ctk.CTkFrame(self, width=240, fg_color=C["panel"], corner_radius=0)
        self.left_sidebar.pack(side="left", fill="y")
        self.left_sidebar.pack_propagate(False)

        # ── Panel central (chat / tabs) ────────────────────────────────
        self.main = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        self.main.pack(side="left", fill="both", expand=True)

        # ── Sidebar derecha (controles) ────────────────────────────────
        self.sidebar = ctk.CTkFrame(self, width=260, fg_color=C["panel"], corner_radius=0)
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
                         anchor="w", wraplength=170, justify="left"
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
                                         wraplength=210, justify="left")
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

        for bname in BACKENDS:
            rb = ctk.CTkRadioButton(sb, text=bname, variable=self.backend_var, value=bname,
                                    fg_color=C["accent"], hover_color=C["accent2"],
                                    font=ctk.CTkFont("Consolas", 11),
                                    text_color=C["text"],
                                    command=self._on_backend_change)
            rb.pack(anchor="w", padx=14, pady=2)

        ctk.CTkFrame(sb, height=1, fg_color=C["border"]).pack(fill="x", padx=16, pady=(6, 4))
        # ────────────────────────────────────────────────────────────────

        self.nav_btns = {}
        for icon, label_key, cmd in [
            ("💬", "nav_chat",     self._show_chat),
            ("⚙️", "nav_server",  self._show_server),
            ("📋", "nav_logs",     self._show_logs),
            ("ℹ️", "nav_info",    self._show_info),
            ("🌐", "nav_download", self._show_download),
            ("📡", "nav_api",      self._show_api_docs),
            ("⚖️", "nav_about",   self._show_about),
        ]:
            b = ctk.CTkButton(sb, text=f"  {icon}  {T(label_key)}",
                               fg_color="transparent", hover_color=C["card"],
                               text_color=C["sub"], font=ctk.CTkFont("Consolas", 13),
                               anchor="w", height=40, command=cmd)
            b.pack(fill="x", padx=8, pady=2)
            self.nav_btns[label_key] = b

        self.ver_label = ctk.CTkLabel(sb, text="llama.cpp: —",
                                       font=ctk.CTkFont("Consolas", 10),
                                       text_color=C["sub"])
        self.ver_label.pack(side="bottom", anchor="w", padx=14, pady=(0, 12))

        # Botones de actualización — uno por cada backend en BACKEND_META
        self._update_btns = {}
        for bkey, bmeta in reversed(list(BACKEND_META.items())):
            btn = ctk.CTkButton(
                sb, text=f"⬆  {bmeta['label']}",
                fg_color=C["card2"], hover_color=C["border"],
                text_color=C["sub"], font=ctk.CTkFont("Consolas", 11),
                height=32, corner_radius=8,
                command=lambda k=bkey: self._open_update_dialog(k)
            )
            btn.pack(side="bottom", fill="x", padx=12, pady=(0, 4))
            self._update_btns[bkey] = btn

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
        self.btn_theme.pack(side="bottom", fill="x", padx=12, pady=(0, 4))

        # Botón de idioma
        self.btn_lang = ctk.CTkButton(
            sb, text=T("lang_btn"),
            fg_color=C["card2"], hover_color=C["border"],
            text_color=C["sub"], font=ctk.CTkFont("Consolas", 11),
            height=32, corner_radius=8,
            command=self._toggle_lang
        )
        self.btn_lang.pack(side="bottom", fill="x", padx=12, pady=(0, 4))

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
        self.btn_sound.pack(side="bottom", fill="x", padx=12, pady=(0, 4))

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
            "Acerca de": "nav_about",
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

        # Toggle thinking
        self.thinking_var = tk.BooleanVar(value=True)
        self.btn_thinking = ctk.CTkButton(
            hdr, text=T("thinking_on"), width=130, height=30,
            fg_color=C["accent"], hover_color="#6457e0",
            font=ctk.CTkFont("Consolas", 11, "bold"),
            command=self._toggle_thinking
        )
        self.btn_thinking.pack(side="right", padx=(0, 8), pady=10)

        # Toggle web search
        self.websearch_var = tk.BooleanVar(value=False)
        self.btn_websearch = ctk.CTkButton(
            hdr, text=T("web_off"), width=110, height=30,
            fg_color=C["card2"], hover_color=C["border"],
            text_color=C["sub"],
            font=ctk.CTkFont("Consolas", 11, "bold"),
            command=self._toggle_websearch
        )
        self.btn_websearch.pack(side="right", padx=(0, 6), pady=10)

        self.chat_display = ctk.CTkTextbox(f, fg_color=C["panel"],
                                            text_color=C["text"],
                                            font=ctk.CTkFont("Consolas", 13),
                                            wrap="word", state="disabled", corner_radius=0)
        self.chat_display.pack(fill="both", expand=True)
        # Configurar tags de color para el thinking
        tb = self.chat_display._textbox
        tb.tag_config("thinking", foreground="#8b7cf8",
                      font=("Consolas", 12, "italic"))
        tb.tag_config("think_hdr", foreground="#6457e0",
                      font=("Consolas", 11, "bold italic"))

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
                                          font=ctk.CTkFont("Consolas", 13), corner_radius=8)
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
        self.btn_attach_file = ctk.CTkButton(ii, text="📎", width=44, height=44,
                                              fg_color=C["card2"], hover_color=C["border"],
                                              text_color=C["sub"],
                                              font=ctk.CTkFont(size=18),
                                              corner_radius=8,
                                              command=self._attach_file)
        self.btn_attach_file.pack(side="left", padx=(6, 0))

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

    def _stop_gen(self):
        self._stop_generation = True

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
        if not txt and not self._attached_image and not self._attached_file:
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
        self.btn_send.configure(state="disabled", text=T("sending"))
        self.btn_stop_gen.configure(state="normal")
        use_web = self.websearch_var.get()
        threading.Thread(target=self._api, args=(msgs, use_web), daemon=True).start()

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

    def _api(self, messages, use_web=False):
        port = self.settings.get("port", "8080")
        p = self.current_prof
        full = ""
        full_raw = ""
        t_start = time.time()
        native_in_think = False
        in_think = False
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

        try:
            current_messages = list(messages)
            max_tool_rounds = 5

            for tool_round in range(max_tool_rounds + 1):
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
                }
                if use_web and tool_round < max_tool_rounds:
                    payload["tools"] = WEB_TOOL
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
                        break
                    if not line: continue
                    line = line.decode()
                    if not line.startswith("data: "): continue
                    d = line[6:]
                    if d == "[DONE]": break
                    try:
                        chunk = json.loads(d)
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

                        if not delta:
                            if "timings" in chunk: timings = chunk["timings"]
                            continue

                        full_raw += delta
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
                if use_web and tool_calls_acc and finish_reason == "tool_calls":
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

                    for tc in tc_list:
                        try:
                            args = json.loads(tc["function"]["arguments"])
                            query = args.get("query", "")
                        except Exception:
                            query = tc["function"]["arguments"]

                        self.after(0, lambda q=query: (
                            self.chat_display.configure(state="normal"),
                            self.chat_display._textbox.insert("end",
                                f"\n\U0001f310 Buscando: {q}\n", "think_hdr"),
                            self.chat_display.configure(state="disabled")
                        ))

                        result = self._web_search(query)
                        self._log(f"[Web] {query}")

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

                break  # sin tool calls, salir del bucle

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
        except Exception as e:
            self.after(0, lambda err=str(e): self._append("system", f"\u26a0 Error: {err}"))
        finally:
            self.after(0, lambda: (
                self.btn_send.configure(state="normal", text=T("send")),
                self.btn_stop_gen.configure(state="disabled"),
            ))

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
        if mmproj and os.path.isfile(mmproj): args += ["--mmproj", mmproj]
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
        if sm and sm != "none":
            args += ["--split-mode", sm]
        ts = str(p.get("tensor_split", "")).strip()
        if ts:
            args += ["--tensor-split", ts]
        # TurboQuant fork: deshabilitar prompt cache para no saturar RAM
        # (el fork activa un cache de hasta 8 GB por defecto)
        if "llama-turboquant" in exe.replace("\\", "/"):
            args += ["--cache-ram", "0"]
        ex = str(p.get("extra_args","")).strip()
        if ex: args.extend(ex.split())
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

    def _srv_exit(self, rc=None):
        was_running = self.server_running
        manual_stop = getattr(self, "_stopping", False)
        self._stopping = False
        self.server_running = False
        self._set_status("Detenido", C["red"])
        self.port_label.configure(text=T("server_port"))
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self._stop_vram_poll()
        # Solo mostrar error si fue un crash no esperado
        if rc not in (None, 0, -1, -15, 1) and not manual_stop and not was_running:
            last = getattr(self, "_last_log_lines", [])
            msg = "\n".join(last[-10:]) if last else "(sin logs)"
            self.after(100, lambda: messagebox.showerror(
                "Error al iniciar el servidor",
                f"El servidor se cerró con código {rc}.\n\nÚltimas líneas del log:\n\n{msg}\n\nRevisa la pestaña Logs para más detalles."
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
                     text_color=C["sub"], justify="left", wraplength=640).pack(anchor="w", padx=16, pady=(0, 6))
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
                     text_color=C["sub"], justify="left", wraplength=640).pack(anchor="w", padx=16, pady=(0, 6))
        _link_row(c3, "Repositorio", "https://github.com/TheTom/llama-cpp-turboquant")
        _link_row(c3, "Paper TurboQuant", "https://arxiv.org/abs/2504.19874")
        ctk.CTkFrame(c3, height=8, fg_color="transparent").pack()

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
            "Este software utiliza llama.cpp y llama-cpp-turboquant, ambos bajo licencia MIT.\n"
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
    if sm and sm != "none": args += ["--split-mode", sm]
    ts = str(p.get("tensor_split", "")).strip()
    if ts: args += ["--tensor-split", ts]
    if "llama-turboquant" in exe.replace("\\", "/"):
        args += ["--cache-ram", "0"]
    ex = str(p.get("extra_args", "")).strip()
    if ex: args.extend(ex.split())

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
