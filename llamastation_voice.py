"""
llamastation_voice.py
=====================
Módulo de voz para LlamaStation.
Integra faster-whisper + XTTS v2 directamente en la app.

Requisitos adicionales:
    pip install coqui-tts faster-whisper sounddevice soundfile scipy pydub numpy

Uso: importado por llamastation.py mediante VoiceMixin.
"""

import os
import sys

# Añadir cuDNN al PATH ANTES de cualquier import de torch/TTS
# Necesario en Windows para que encuentre cudnn64_8.dll
try:
    import nvidia.cudnn as _cudnn
    _cudnn_bin = os.path.join(os.path.dirname(_cudnn.__file__), "bin")
    if os.path.isdir(_cudnn_bin):
        if _cudnn_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _cudnn_bin + os.pathsep + os.environ.get("PATH", "")
        # También añadir al DLL search path de Python (Windows específico)
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(_cudnn_bin)
            except Exception:
                pass
except ImportError:
    pass

import json
import time
import threading
import tempfile
import re

import numpy as np
import sounddevice as sd
import soundfile as sf

from pathlib import Path
from tkinter import filedialog, messagebox
import customtkinter as ctk

try:
    from pydub import AudioSegment
    _PYDUB_OK = True
except ImportError:
    _PYDUB_OK = False

# ── Rutas ────────────────────────────────────────────────────────────────────
_BASE_DIR   = Path(__file__).parent
VOICES_DIR  = _BASE_DIR / "voices"
VOICES_DIR.mkdir(exist_ok=True)

# ── Carga diferida de modelos (singleton por proceso) ────────────────────────
_whisper_model = None
_tts_model     = None
_models_lock   = threading.Lock()

def get_whisper(model_name="base"):
    global _whisper_model
    with _models_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
    return _whisper_model

def get_tts(device="cpu"):
    global _tts_model
    with _models_lock:
        if _tts_model is None:
            from TTS.api import TTS
            _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return _tts_model

def reload_whisper():
    global _whisper_model
    with _models_lock:
        _whisper_model = None

def reload_tts():
    global _tts_model
    with _models_lock:
        _tts_model = None


# ── Audio utils ──────────────────────────────────────────────────────────────

def record_audio_vad(stop_event: threading.Event, samplerate=16000,
                     chunk_ms=100, silence_thresh=0.01, silence_s=1.2) -> np.ndarray:
    """Graba hasta silencio prolongado o stop_event."""
    chunk_frames  = int(samplerate * chunk_ms / 1000)
    silence_chunks = int(silence_s * 1000 / chunk_ms)
    chunks = []
    silent_count = 0
    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32",
                        blocksize=chunk_frames) as stream:
        while not stop_event.is_set():
            data, _ = stream.read(chunk_frames)
            flat = data.flatten()
            chunks.append(flat.copy())
            rms = float(np.sqrt(np.mean(flat**2)))
            if rms < silence_thresh:
                silent_count += 1
            else:
                silent_count = 0
            if silent_count >= silence_chunks and len(chunks) > silence_chunks:
                break
    return np.concatenate(chunks) if chunks else np.zeros(samplerate)

def transcribe(audio: np.ndarray, model_name="base", language="es") -> str:
    model = get_whisper(model_name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    sf.write(tmp, audio, 16000)
    segments, _ = model.transcribe(tmp, language=language, beam_size=5, vad_filter=True)
    text = " ".join(s.text for s in segments).strip()
    os.unlink(tmp)
    return text

# Flag global para interrupción de TTS
_tts_interrupt = threading.Event()

def synthesize_and_play(text: str, speaker_wav: str, language="es",
                        speed=1.0, device="cpu"):
    _tts_interrupt.clear()
    tts = get_tts(device=device)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out = f.name
    # Generar a velocidad normal y aplicar tempo después (sin tocar pitch)
    tts.tts_to_file(text=text, speaker_wav=speaker_wav,
                    language=language, file_path=out, speed=1.0)
    data, sr = sf.read(out)
    # Time-stretch con librosa: cambia tempo sin afectar tono
    if speed and abs(speed - 1.0) > 0.01:
        try:
            import librosa
            data = librosa.effects.time_stretch(data.astype("float32"), rate=float(speed))
        except Exception:
            # Fallback: cambiar sample rate (afecta pitch)
            sr = int(sr * float(speed))
    sd.play(data, sr)
    # Esperar con chequeo de interrupción cada 100ms
    while sd.get_stream() and sd.get_stream().active:
        if _tts_interrupt.is_set():
            sd.stop()
            break
        time.sleep(0.1)
    try:
        os.unlink(out)
    except Exception:
        pass

def prepare_speaker_wav(src_path: str, out_path: str):
    """Convierte cualquier audio a WAV 22050Hz mono para XTTS."""
    if not _PYDUB_OK:
        raise RuntimeError("pydub no está instalado. Ejecuta: pip install pydub")
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_channels(1).set_frame_rate(22050)
    audio.export(out_path, format="wav")

def record_audio_fixed(duration_s: float, samplerate=16000) -> np.ndarray:
    """Graba audio fijo (para capturar voz de referencia)."""
    frames = int(duration_s * samplerate)
    audio = sd.rec(frames, samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


# ════════════════════════════════════════════════════════════════════════════
#  VoiceMixin — se mezcla en la clase LlamaStation
# ════════════════════════════════════════════════════════════════════════════

class VoiceMixin:
    """
    Mixin que añade funcionalidad de voz a LlamaStation.

    Requisitos en LlamaStation.__init__ (ya añadidos vía patch):
        self._voice_recording  = False
        self._voice_stop_ev    = threading.Event()
        self._voice_ptt_active = False
        self._voice_models_ok  = False
        self._voice_active_wav = None   # ruta al .wav de referencia activo
        self._voice_voices     = {}     # nombre → ruta .wav
    """

    # ── Inicialización del estado de voz ─────────────────────────────────────

    def _init_voice_state(self):
        self._voice_recording  = False
        self._voice_stop_ev    = threading.Event()
        self._voice_ptt_active = False
        self._voice_models_ok  = False
        self._voice_active_wav = None
        self._voice_voices     = {}
        self._voice_ptt_chunks = []
        self._voice_ptt_stream = None
        self._voice_scan()
        # Restaurar voz activa desde settings
        saved_voice = self.settings.get("voice_active", None)
        if saved_voice and saved_voice in self._voice_voices:
            self._voice_active_wav = self._voice_voices[saved_voice]
        # Autoload de modelos si está activado
        if self.settings.get("voice_autoload", False):
            self.after(800, self._voice_load_models)

    def _voice_scan(self):
        self._voice_voices = {}
        for f in VOICES_DIR.glob("*.wav"):
            self._voice_voices[f.stem] = str(f)

    def _voice_save_settings(self):
        """Guarda config de voz en llamastation_settings.json."""
        if hasattr(self, "_voice_active_wav") and self._voice_active_wav:
            name = Path(self._voice_active_wav).stem
            self.settings["voice_active"] = name
        else:
            self.settings.pop("voice_active", None)
        if hasattr(self, "_voice_lang_var"):
            self.settings["voice_lang"]    = self._voice_lang_var.get()
        if hasattr(self, "_voice_speed_var"):
            self.settings["voice_speed"]   = self._voice_speed_var.get()
        if hasattr(self, "_voice_whisper_var"):
            self.settings["voice_whisper"] = self._voice_whisper_var.get()
        if hasattr(self, "_voice_device_var"):
            self.settings["voice_device"]  = self._voice_device_var.get()
        if hasattr(self, "_voice_sysprompt"):
            self.settings["voice_sysprompt"] = self._voice_sysprompt.get("1.0", "end").strip()
        from llamastation import save_settings
        save_settings(self.settings)

    # ── Construcción de la pestaña Voz ───────────────────────────────────────

    def _build_voice(self, parent):
        """Construye el frame de la pestaña Voz. Devuelve el frame."""
        import llamastation as _ls
        C = _ls.C   # tomar referencia FRESCA al diccionario de colores

        f = ctk.CTkFrame(parent, fg_color=C["bg"], corner_radius=0)

        # Header
        hdr = ctk.CTkFrame(f, fg_color=C["panel"], height=52, corner_radius=0)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="🎤  Agente de Voz",
                     font=ctk.CTkFont("Consolas", 15, "bold"),
                     text_color=C["text"]).pack(side="left", padx=18, pady=14)
        self._voice_status_dot = ctk.CTkLabel(hdr, text="●",
                                               font=ctk.CTkFont(size=16),
                                               text_color=C["dim"])
        self._voice_status_dot.pack(side="left", padx=(0, 6))
        self._voice_status_lbl = ctk.CTkLabel(hdr, text="Modelos no cargados",
                                               font=ctk.CTkFont("Consolas", 11),
                                               text_color=C["sub"])
        self._voice_status_lbl.pack(side="left")

        # Body — dos columnas
        body = ctk.CTkFrame(f, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=12)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)

        # ── Columna izquierda: chat de voz ───────────────────────────────────
        left = ctk.CTkFrame(body, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self._voice_chat = ctk.CTkTextbox(left, fg_color=C["panel"],
                                           text_color=C["text"],
                                           font=ctk.CTkFont("Consolas", 12),
                                           wrap="word", state="disabled",
                                           corner_radius=10)
        self._voice_chat.grid(row=0, column=0, sticky="nsew")
        self._voice_chat._textbox.tag_configure("user",  foreground=C["accent"])
        self._voice_chat._textbox.tag_configure("bot",   foreground=C["accent2"])
        self._voice_chat._textbox.tag_configure("sys",   foreground=C["dim"],
                                                          font=("Consolas", 11, "italic"))
        self._voice_chat._textbox.tag_configure("err",   foreground=C["red"])

        ctk.CTkButton(left, text="Limpiar", height=26,
                       fg_color=C["card2"], hover_color=C["border"],
                       text_color=C["sub"], font=ctk.CTkFont("Consolas", 10),
                       corner_radius=6,
                       command=self._voice_clear_chat
                       ).grid(row=1, column=0, sticky="e", pady=(6, 0))

        # Controles de grabación
        ctrl = ctk.CTkFrame(left, fg_color=C["card2"], height=64, corner_radius=10)
        ctrl.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ctrl.pack_propagate(False)
        ci = ctk.CTkFrame(ctrl, fg_color="transparent")
        ci.pack(fill="both", expand=True, padx=12, pady=10)

        # Botón escucha continua (toggle)
        self._voice_btn_listen = ctk.CTkButton(
            ci, text="🎙 Escuchar", width=140, height=40,
            fg_color=C["accent"], hover_color="#6457e0",
            text_color="white", font=ctk.CTkFont("Consolas", 12, "bold"),
            corner_radius=8, command=self._voice_toggle_listen
        )
        self._voice_btn_listen.pack(side="left", padx=(0, 8))

        # Botón PTT (mantener pulsado)
        self._voice_btn_ptt = ctk.CTkButton(
            ci, text="⏺ Push-to-talk", width=160, height=40,
            fg_color=C["card"], hover_color=C["border"],
            text_color=C["sub"], font=ctk.CTkFont("Consolas", 12),
            corner_radius=8, command=None
        )
        self._voice_btn_ptt.pack(side="left", padx=(0, 8))
        self._voice_btn_ptt.bind("<ButtonPress-1>",   self._voice_ptt_press)
        self._voice_btn_ptt.bind("<ButtonRelease-1>", self._voice_ptt_release)

        # Botón parar
        ctk.CTkButton(ci, text="⏹", width=40, height=40,
                       fg_color="#3a1a1a", hover_color="#5a2020",
                       text_color=C["red"], font=ctk.CTkFont("Consolas", 14, "bold"),
                       corner_radius=8, command=self._voice_stop_all
                       ).pack(side="left", padx=(0, 8))

        # Indicador REC
        self._voice_rec_lbl = ctk.CTkLabel(ci, text="",
                                            font=ctk.CTkFont("Consolas", 11),
                                            text_color=C["red"])
        self._voice_rec_lbl.pack(side="left")

        # ── Columna derecha: config ──────────────────────────────────────────
        right = ctk.CTkScrollableFrame(body, fg_color=C["panel"], corner_radius=10, width=280)
        right.grid(row=0, column=1, sticky="nsew")

        def sec(title):
            ctk.CTkLabel(right, text=title,
                         font=ctk.CTkFont("Consolas", 10, "bold"),
                         text_color=C["accent"]).pack(anchor="w", padx=14, pady=(12, 2))
            ctk.CTkFrame(right, fg_color=C["border"], height=1).pack(fill="x", padx=14, pady=(0, 6))

        def card():
            fr = ctk.CTkFrame(right, fg_color=C["card"], corner_radius=8)
            fr.pack(fill="x", padx=10, pady=4)
            return fr

        # ── Voz activa ───────────────────────────────────────────────────────
        sec("VOZ DE REFERENCIA")
        vc = card()
        ctk.CTkLabel(vc, text="Voz activa", font=ctk.CTkFont("Consolas", 10),
                     text_color=C["sub"]).pack(anchor="w", padx=10, pady=(8, 0))
        voice_names = ["— sin voz —"] + list(self._voice_voices.keys())
        self._voice_menu_var = ctk.StringVar(value="— sin voz —")
        # Restaurar selección
        if self._voice_active_wav:
            stem = Path(self._voice_active_wav).stem
            if stem in self._voice_voices:
                self._voice_menu_var.set(stem)
        self._voice_menu = ctk.CTkOptionMenu(
            vc, variable=self._voice_menu_var,
            values=voice_names,
            fg_color=C["input"], button_color=C["accent"],
            button_hover_color=C["dim"], text_color=C["text"],
            font=ctk.CTkFont("Consolas", 11), corner_radius=6,
            command=self._voice_on_select
        )
        self._voice_menu.pack(fill="x", padx=10, pady=(0, 4))

        row2 = ctk.CTkFrame(vc, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkButton(row2, text="▶ Probar", height=28, width=80,
                       fg_color=C["card2"], hover_color=C["border"],
                       text_color=C["sub"], font=ctk.CTkFont("Consolas", 10),
                       corner_radius=6, command=self._voice_preview
                       ).pack(side="left", padx=(0, 4))
        ctk.CTkButton(row2, text="🗑 Borrar", height=28, width=80,
                       fg_color=C["card2"], hover_color=C["border"],
                       text_color=C["red"], font=ctk.CTkFont("Consolas", 10),
                       corner_radius=6, command=self._voice_delete
                       ).pack(side="left")

        ctk.CTkButton(vc, text="🎤 Grabar nueva voz", height=32,
                       fg_color=C["accent"], hover_color="#6457e0",
                       text_color="white", font=ctk.CTkFont("Consolas", 11),
                       corner_radius=6, command=self._voice_record_new
                       ).pack(fill="x", padx=10, pady=(0, 4))
        ctk.CTkButton(vc, text="📂 Importar audio", height=32,
                       fg_color=C["card2"], hover_color=C["border"],
                       text_color=C["accent2"], font=ctk.CTkFont("Consolas", 11),
                       corner_radius=6, command=self._voice_import
                       ).pack(fill="x", padx=10, pady=(0, 8))

        # ── Idioma y velocidad ────────────────────────────────────────────────
        sec("IDIOMA Y VELOCIDAD")
        lc = card()

        ctk.CTkLabel(lc, text="Idioma", font=ctk.CTkFont("Consolas", 10),
                     text_color=C["sub"]).pack(anchor="w", padx=10, pady=(8, 0))
        self._voice_lang_var = ctk.StringVar(
            value=self.settings.get("voice_lang", "es"))
        ctk.CTkOptionMenu(lc, variable=self._voice_lang_var,
                           values=["es","en","fr","de","it","pt","pl","nl",
                                   "cs","ar","zh-cn","ja","hu","ko"],
                           fg_color=C["input"], button_color=C["accent"],
                           button_hover_color=C["dim"], text_color=C["text"],
                           font=ctk.CTkFont("Consolas", 11), corner_radius=6
                           ).pack(fill="x", padx=10, pady=(0, 4))

        ctk.CTkLabel(lc, text="Velocidad de voz", font=ctk.CTkFont("Consolas", 10),
                     text_color=C["sub"]).pack(anchor="w", padx=10)
        speed_row = ctk.CTkFrame(lc, fg_color="transparent")
        speed_row.pack(fill="x", padx=10, pady=(0, 4))
        self._voice_speed_var = ctk.DoubleVar(
            value=float(self.settings.get("voice_speed", 1.0)))
        spd_slider = ctk.CTkSlider(speed_row, from_=0.5, to=2.0, number_of_steps=30,
                                    variable=self._voice_speed_var,
                                    button_color=C["accent"],
                                    progress_color=C["accent"],
                                    command=lambda v: self._voice_speed_lbl.configure(
                                        text=f"{float(v):.1f}x"))
        spd_slider.pack(side="left", fill="x", expand=True)
        self._voice_speed_lbl = ctk.CTkLabel(speed_row, text=f"{self._voice_speed_var.get():.1f}x",
                                              width=36, font=ctk.CTkFont("Consolas", 10),
                                              text_color=C["accent"])
        self._voice_speed_lbl.pack(side="left", padx=(4, 0))

        ctk.CTkFrame(lc, height=8, fg_color="transparent").pack()

        # ── Configuración de modelos ──────────────────────────────────────────
        sec("MODELOS")
        mc = card()

        ctk.CTkLabel(mc, text="Modelo Whisper", font=ctk.CTkFont("Consolas", 10),
                     text_color=C["sub"]).pack(anchor="w", padx=10, pady=(8, 0))
        self._voice_whisper_var = ctk.StringVar(
            value=self.settings.get("voice_whisper", "base"))
        ctk.CTkOptionMenu(mc, variable=self._voice_whisper_var,
                           values=["tiny","base","small","medium","large-v3"],
                           fg_color=C["input"], button_color=C["accent"],
                           button_hover_color=C["dim"], text_color=C["text"],
                           font=ctk.CTkFont("Consolas", 11), corner_radius=6
                           ).pack(fill="x", padx=10, pady=(0, 4))

        ctk.CTkLabel(mc, text="Dispositivo XTTS", font=ctk.CTkFont("Consolas", 10),
                     text_color=C["sub"]).pack(anchor="w", padx=10)
        self._voice_device_var = ctk.StringVar(
            value=self.settings.get("voice_device", "cpu"))
        ctk.CTkOptionMenu(mc, variable=self._voice_device_var,
                           values=["cpu","cuda"],
                           fg_color=C["input"], button_color=C["accent"],
                           button_hover_color=C["dim"], text_color=C["text"],
                           font=ctk.CTkFont("Consolas", 11), corner_radius=6
                           ).pack(fill="x", padx=10, pady=(0, 4))

        btn_row = ctk.CTkFrame(mc, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(0, 4))
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)

        ctk.CTkButton(btn_row, text="⟳ Cargar", height=30,
                       fg_color=C["card2"], hover_color=C["border"],
                       text_color=C["accent2"], font=ctk.CTkFont("Consolas", 11),
                       corner_radius=6, command=self._voice_load_models
                       ).grid(row=0, column=0, sticky="ew", padx=(0, 2))

        ctk.CTkButton(btn_row, text="✕ Descargar", height=30,
                       fg_color=C["card2"], hover_color=C["border"],
                       text_color=C["red"], font=ctk.CTkFont("Consolas", 11),
                       corner_radius=6, command=self._voice_unload_models
                       ).grid(row=0, column=1, sticky="ew", padx=(2, 0))

        import tkinter as _tk
        self._voice_autoload_var = _tk.BooleanVar(
            value=self.settings.get("voice_autoload", False))
        ctk.CTkCheckBox(mc, text="Cargar al arrancar la app",
                         variable=self._voice_autoload_var,
                         fg_color=C["accent"], hover_color=C["dim"],
                         text_color=C["sub"], font=ctk.CTkFont("Consolas", 10),
                         command=self._voice_toggle_autoload
                         ).pack(anchor="w", padx=10, pady=(0, 8))

        # ── System prompt de voz ─────────────────────────────────────────────
        sec("SYSTEM PROMPT DE VOZ")
        pc = card()
        ctk.CTkLabel(pc, text="Se usa solo en el modo voz (no afecta al chat)",
                     font=ctk.CTkFont("Consolas", 9), text_color=C["dim"],
                     wraplength=220, justify="left").pack(anchor="w", padx=10, pady=(6, 2))
        self._voice_sysprompt = ctk.CTkTextbox(pc, height=90,
                                                fg_color=C["input"], text_color=C["text"],
                                                font=ctk.CTkFont("Consolas", 11),
                                                corner_radius=6, wrap="word")
        self._voice_sysprompt.pack(fill="x", padx=10, pady=(0, 4))
        default_sys = (
            "Eres un asistente de voz útil y conciso. "
            "Responde siempre de forma natural y breve, "
            "adecuada para conversación hablada. "
            "Evita listas, markdown o respuestas muy largas."
        )
        self._voice_sysprompt.insert("1.0", self.settings.get("voice_sysprompt", default_sys))

        ctk.CTkButton(pc, text="💾 Guardar config", height=30,
                       fg_color=C["accent"], hover_color="#6457e0",
                       text_color="white", font=ctk.CTkFont("Consolas", 11),
                       corner_radius=6, command=self._voice_save_settings
                       ).pack(fill="x", padx=10, pady=(0, 8))

        return f

    # ── Status ───────────────────────────────────────────────────────────────

    def _voice_set_status(self, text, color=None):
        import llamastation as _ls
        C = _ls.C
        col = color or C["sub"]
        def _do():
            if hasattr(self, "_voice_status_lbl"):
                self._voice_status_lbl.configure(text=text)
            if hasattr(self, "_voice_status_dot"):
                self._voice_status_dot.configure(text_color=col)
        self.after(0, _do)

    def _voice_append(self, who, text, tag="sys"):
        def _do():
            if not hasattr(self, "_voice_chat"):
                return
            self._voice_chat.configure(state="normal")
            ts = time.strftime("%H:%M")
            import llamastation as _ls
            C = _ls.C
            if who == "tú":
                self._voice_chat._textbox.insert("end", f"\n  Tú  [{ts}]\n", "user")
            elif who == "agente":
                self._voice_chat._textbox.insert("end", f"\n  Agente  [{ts}]\n", "bot")
            else:
                self._voice_chat._textbox.insert("end", f"\n  ◈  [{ts}]\n", "sys")
            self._voice_chat._textbox.insert("end", f"  {text}\n", tag)
            self._voice_chat.configure(state="disabled")
            self._voice_chat._textbox.see("end")
        self.after(0, _do)

    def _voice_clear_chat(self):
        if hasattr(self, "_voice_chat"):
            self._voice_chat.configure(state="normal")
            self._voice_chat.delete("1.0", "end")
            self._voice_chat.configure(state="disabled")

    # ── Carga de modelos ─────────────────────────────────────────────────────

    def _voice_toggle_autoload(self):
        self.settings["voice_autoload"] = self._voice_autoload_var.get()
        from llamastation import save_settings
        save_settings(self.settings)

    def _voice_unload_models(self):
        """Libera Whisper y XTTS de memoria sin reiniciar la app."""
        from tkinter import messagebox
        # Parar cualquier reproducción/grabación activa
        self._voice_recording = False
        self._vchat_listening = False
        if hasattr(self, "_voice_stop_ev") and self._voice_stop_ev:
            self._voice_stop_ev.set()
        if hasattr(self, "_vchat_stop_ev") and self._vchat_stop_ev:
            self._vchat_stop_ev.set()
        try:
            _tts_interrupt.set()
        except Exception:
            pass
        # Liberar modelos
        reload_whisper()
        reload_tts()
        # Liberar VRAM si es CUDA
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        import gc
        gc.collect()
        self._voice_models_ok = False
        self._voice_set_status("Modelos descargados")
        self._voice_append("sistema", "Modelos liberados de memoria. Pulsa 'Cargar' para volver a usarlos.", "sys")
        messagebox.showinfo("Voz", "Modelos descargados de memoria.\nPuedes cargar otros sin reiniciar la app.")

    def _voice_load_models(self):
        if self._voice_models_ok:
            messagebox.showinfo("Voz", "Los modelos ya están cargados.")
            return
        self._voice_set_status("Cargando Whisper…", "#fbbf24")
        def _load():
            try:
                reload_whisper()
                reload_tts()
                wm  = self.settings.get("voice_whisper", "base")
                dev = self.settings.get("voice_device", "cpu")
                get_whisper(wm)
                self._voice_set_status("Cargando XTTS v2 (puede tardar ~30s)…", "#fbbf24")
                get_tts(device=dev)
                self._voice_models_ok = True
                self._voice_set_status("Listo ✓", "#4ade80")
                self._voice_append("sistema", "Modelos cargados. ¡Ya puedes hablar!", "sys")
            except Exception as e:
                self._voice_set_status(f"Error: {e}", "#f87171")
                self._voice_append("sistema", f"Error cargando modelos: {e}", "err")
        threading.Thread(target=_load, daemon=True).start()

    # ── Escucha continua ─────────────────────────────────────────────────────

    def _voice_toggle_listen(self):
        if not self._voice_models_ok:
            messagebox.showinfo("Voz", "Primero carga los modelos en la pestaña Voz.")
            return
        if not self._voice_active_wav:
            messagebox.showwarning("Voz", "Selecciona una voz de referencia primero.")
            return
        if self._voice_recording:
            self._voice_stop_all()
        else:
            self._voice_start_listen()

    def _voice_start_listen(self):
        import llamastation as _ls
        C = _ls.C
        self._voice_recording = True
        self._voice_stop_ev.clear()
        self._voice_btn_listen.configure(text="⏹ Detener",
                                          fg_color=C["red"],
                                          hover_color="#b03025")
        self._voice_rec_lbl.configure(text="⬤ REC")
        self._voice_set_status("Escuchando…", C["red"])
        threading.Thread(target=self._voice_listen_loop, daemon=True).start()

    def _voice_listen_loop(self):
        while self._voice_recording and not self._voice_stop_ev.is_set():
            try:
                audio = record_audio_vad(self._voice_stop_ev)
                if self._voice_stop_ev.is_set():
                    break
                if len(audio) < 8000:
                    continue
                self._voice_set_status("Transcribiendo…", "#fbbf24")
                text = transcribe(audio,
                                  self.settings.get("voice_whisper", "base"),
                                  self._voice_lang_var.get() if hasattr(self, "_voice_lang_var") else "es")
                if not text.strip():
                    self._voice_set_status("Escuchando…")
                    continue
                self._voice_append("tú", text, "msg")
                self._voice_run_agent(text)
            except Exception as e:
                self._voice_append("sistema", f"Error: {e}", "err")
                time.sleep(0.5)
        self.after(0, self._voice_reset_idle)

    def _voice_stop_all(self):
        self._voice_recording = False
        self._voice_stop_ev.set()
        _tts_interrupt.set()
        self.after(0, self._voice_reset_idle)

    def _voice_reset_idle(self):
        import llamastation as _ls
        C = _ls.C
        self._voice_recording = False
        if hasattr(self, "_voice_btn_listen"):
            self._voice_btn_listen.configure(text="🎙 Escuchar",
                                              fg_color=C["accent"],
                                              hover_color="#6457e0")
        if hasattr(self, "_voice_rec_lbl"):
            self._voice_rec_lbl.configure(text="")
        self._voice_set_status("Listo")

    # ── Push-to-talk ─────────────────────────────────────────────────────────

    def _voice_ptt_press(self, event=None):
        if self._voice_recording or self._voice_ptt_active:
            return
        if not self._voice_models_ok:
            return
        if not self._voice_active_wav:
            messagebox.showwarning("Voz", "Selecciona una voz de referencia primero.")
            return
        import llamastation as _ls
        C = _ls.C
        self._voice_ptt_active = True
        self._voice_stop_ev.clear()
        self._voice_ptt_chunks = []
        self._voice_btn_ptt.configure(fg_color=C["red"], text_color="white",
                                       text="⬤ Grabando…")
        self._voice_rec_lbl.configure(text="⬤ REC")
        self._voice_set_status("PTT — grabando…", C["red"])
        self._voice_ptt_stream = sd.InputStream(
            samplerate=16000, channels=1, dtype="float32",
            blocksize=int(16000 * 0.1),
            callback=self._voice_ptt_callback)
        self._voice_ptt_stream.start()

    def _voice_ptt_callback(self, indata, frames, time_info, status):
        self._voice_ptt_chunks.append(indata.copy().flatten())

    def _voice_ptt_release(self, event=None):
        if not self._voice_ptt_active:
            return
        import llamastation as _ls
        C = _ls.C
        self._voice_ptt_active = False
        if self._voice_ptt_stream:
            try:
                self._voice_ptt_stream.stop()
                self._voice_ptt_stream.close()
            except Exception:
                pass
            self._voice_ptt_stream = None
        self._voice_btn_ptt.configure(fg_color=C["card"], text_color=C["sub"],
                                       text="⏺ Push-to-talk")
        self._voice_rec_lbl.configure(text="")
        audio = (np.concatenate(self._voice_ptt_chunks)
                 if self._voice_ptt_chunks else np.zeros(16000))
        self._voice_ptt_chunks = []
        self._voice_set_status("Transcribiendo…", "#fbbf24")
        threading.Thread(target=self._voice_ptt_process, args=(audio,), daemon=True).start()

    def _voice_ptt_process(self, audio: np.ndarray):
        try:
            text = transcribe(audio,
                              self.settings.get("voice_whisper", "base"),
                              self._voice_lang_var.get() if hasattr(self, "_voice_lang_var") else "es")
            if not text.strip():
                self._voice_set_status("Listo")
                return
            self._voice_append("tú", text, "msg")
            self._voice_run_agent(text)
        except Exception as e:
            self._voice_append("sistema", f"Error PTT: {e}", "err")
        self._voice_set_status("Listo")

    # ── Agente (LLM + TTS) ───────────────────────────────────────────────────

    def _voice_run_agent(self, user_text: str):
        self._voice_set_status("Pensando…", "#fbbf24")
        sys_prompt = self.settings.get(
            "voice_sysprompt",
            "Eres un asistente de voz útil y conciso. Responde brevemente.")
        # Usar el historial del chat de voz (últimos 10 turnos)
        if not hasattr(self, "_voice_history"):
            self._voice_history = []
        messages = [{"role": "system", "content": sys_prompt}]
        messages += self._voice_history[-20:]
        messages.append({"role": "user", "content": user_text})

        port = self.settings.get("port", "8080")
        host = self.settings.get("host", "127.0.0.1")
        url  = f"http://{host}:{port}/v1/chat/completions"

        try:
            import requests as _req
            r = _req.post(url, json={
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 300,
                "stream": False,
            }, timeout=60)
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            self._voice_append("sistema", f"Error LLM: {e}", "err")
            self._voice_set_status("Error LLM", "#f87171")
            return

        # Limpiar thinking blocks y canales antes de TTS
        import re as _re
        _s = reply
        _s = _re.sub("(?s)<[|]channel>.*?<channel[|]>", "", _s)
        _s = _re.sub("(?s)<think>.*?</think>", "", _s)
        _s = _re.sub("<[|][^|]*[|]>", "", _s)
        reply_clean = _s.strip() or reply

        self._voice_history.append({"role": "user",      "content": user_text})
        self._voice_history.append({"role": "assistant", "content": reply_clean})
        self._voice_append("agente", reply_clean, "msg")
        self._voice_set_status("Hablando…", "#4ade80")

        try:
            lang  = self._voice_lang_var.get() if hasattr(self, "_voice_lang_var") else "es"
            speed = float(self._voice_speed_var.get()) if hasattr(self, "_voice_speed_var") else 1.0
            dev   = self.settings.get("voice_device", "cpu")
            synthesize_and_play(reply_clean, self._voice_active_wav, lang, speed, dev)
        except Exception as e:
            self._voice_append("sistema", f"Error TTS: {e}", "err")

        if self._voice_recording:
            self._voice_set_status("Escuchando…")
        else:
            self._voice_set_status("Listo")

    # ── Gestión de voces ─────────────────────────────────────────────────────

    def _voice_on_select(self, name: str):
        if name == "— sin voz —":
            self._voice_active_wav = None
        else:
            self._voice_active_wav = self._voice_voices.get(name)
        self._voice_save_settings()

    def _voice_refresh_menu(self):
        self._voice_scan()
        names = ["— sin voz —"] + list(self._voice_voices.keys())
        if hasattr(self, "_voice_menu"):
            self._voice_menu.configure(values=names)

    def _voice_preview(self):
        if not self._voice_active_wav:
            messagebox.showinfo("Voz", "No hay voz activa.")
            return
        if not self._voice_models_ok:
            messagebox.showinfo("Voz", "Carga los modelos primero.")
            return
        def _do():
            try:
                lang  = self._voice_lang_var.get() if hasattr(self, "_voice_lang_var") else "es"
                speed = float(self._voice_speed_var.get()) if hasattr(self, "_voice_speed_var") else 1.0
                dev   = self.settings.get("voice_device", "cpu")
                synthesize_and_play("Hola, esta es una prueba de mi voz clonada.",
                                    self._voice_active_wav, lang, speed, dev)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error TTS", str(e)))
        threading.Thread(target=_do, daemon=True).start()

    def _voice_delete(self):
        if not self._voice_active_wav:
            messagebox.showinfo("Voz", "No hay voz seleccionada.")
            return
        name = Path(self._voice_active_wav).stem
        if not messagebox.askyesno("Borrar voz", f"¿Borrar la voz '{name}'?"):
            return
        try:
            os.remove(self._voice_active_wav)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self._voice_active_wav = None
        self._voice_refresh_menu()
        if hasattr(self, "_voice_menu_var"):
            self._voice_menu_var.set("— sin voz —")
        self._voice_save_settings()

    def _voice_import(self):
        path = filedialog.askopenfilename(
            title="Selecciona un audio de referencia",
            filetypes=[("Audio", "*.wav *.mp3 *.m4a *.ogg *.flac *.aac"),
                       ("Todos", "*.*")]
        )
        if not path:
            return
        self._voice_open_name_dialog(src_path=path)

    def _voice_record_new(self):
        import llamastation as _ls
        C = _ls.C
        dlg = ctk.CTkToplevel(self)
        dlg.title("Grabar voz de referencia")
        dlg.geometry("380x290")
        dlg.configure(fg_color=C["bg"])
        dlg.grab_set()

        ctk.CTkLabel(dlg,
                     text="Graba 10-15 segundos de voz clara,\nsin ruido, a velocidad normal.",
                     font=ctk.CTkFont("Consolas", 12), text_color=C["sub"],
                     justify="center").pack(pady=(20, 10))

        st = ctk.CTkLabel(dlg, text="Listo para grabar",
                           font=ctk.CTkFont("Consolas", 11),
                           text_color=C["dim"])
        st.pack(pady=(0, 10))

        holder = [None]

        def _record():
            btn_rec.configure(state="disabled", text="⬤ Grabando 12s…")
            st.configure(text="Grabando…", text_color=C["red"])
            def _do():
                audio = record_audio_fixed(12.0)
                holder[0] = audio
                dlg.after(0, lambda: st.configure(text="✓ Grabación completa",
                                                    text_color=C["green"]))
                dlg.after(0, lambda: btn_save.configure(state="normal"))
                dlg.after(0, lambda: btn_play.configure(state="normal"))
                dlg.after(0, lambda: btn_rec.configure(state="normal",
                                                         text="🔄 Grabar de nuevo"))
            threading.Thread(target=_do, daemon=True).start()

        def _play():
            if holder[0] is not None:
                sd.play(holder[0], 16000)

        def _save():
            if holder[0] is not None:
                dlg.destroy()
                self._voice_open_name_dialog(recorded_audio=holder[0])

        btn_rec = ctk.CTkButton(dlg, text="🎤 Grabar 12 segundos", height=40,
                                 fg_color=C["accent"], hover_color="#6457e0",
                                 text_color="white", font=ctk.CTkFont("Consolas", 12),
                                 corner_radius=8, command=_record)
        btn_rec.pack(pady=(0, 8), padx=20, fill="x")
        btn_play = ctk.CTkButton(dlg, text="▶ Escuchar grabación", height=34,
                                  fg_color=C["card2"], hover_color=C["border"],
                                  text_color=C["sub"], font=ctk.CTkFont("Consolas", 11),
                                  corner_radius=8, state="disabled", command=_play)
        btn_play.pack(pady=(0, 8), padx=20, fill="x")
        btn_save = ctk.CTkButton(dlg, text="💾 Guardar esta voz", height=34,
                                  fg_color=C["card2"], hover_color=C["border"],
                                  text_color=C["accent2"], font=ctk.CTkFont("Consolas", 11),
                                  corner_radius=8, state="disabled", command=_save)
        btn_save.pack(padx=20, fill="x")

    def _voice_open_name_dialog(self, src_path=None, recorded_audio=None):
        import llamastation as _ls
        C = _ls.C
        dlg = ctk.CTkToplevel(self)
        dlg.title("Guardar voz")
        dlg.geometry("340x160")
        dlg.configure(fg_color=C["bg"])
        dlg.grab_set()

        ctk.CTkLabel(dlg, text="Nombre para esta voz:",
                     font=ctk.CTkFont("Consolas", 12), text_color=C["text"]
                     ).pack(pady=(20, 6))
        name_e = ctk.CTkEntry(dlg, fg_color=C["input"], text_color=C["text"],
                               font=ctk.CTkFont("Consolas", 12), width=260,
                               border_color=C["border"])
        name_e.pack()
        name_e.focus()

        def _save():
            name = name_e.get().strip()
            if not name:
                messagebox.showwarning("Nombre vacío", "Escribe un nombre.", parent=dlg)
                return
            safe    = re.sub(r'[^\w\-]', '_', name)
            out_wav = str(VOICES_DIR / f"{safe}.wav")
            try:
                if src_path:
                    prepare_speaker_wav(src_path, out_wav)
                elif recorded_audio is not None:
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp_name = tmp.name; tmp.close()
                    sf.write(tmp_name, recorded_audio, 16000)
                    prepare_speaker_wav(tmp_name, out_wav)
                    os.unlink(tmp_name)
                self._voice_refresh_menu()
                self._voice_menu_var.set(safe)
                self._voice_on_select(safe)
                dlg.destroy()
                messagebox.showinfo("Voz guardada", f"Voz '{safe}' añadida y seleccionada.")
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=dlg)

        ctk.CTkButton(dlg, text="Guardar", height=34,
                       fg_color=C["accent"], hover_color="#6457e0",
                       text_color="white", font=ctk.CTkFont("Consolas", 12),
                       corner_radius=8, command=_save).pack(pady=16)

    # ── Botones PTT en la barra del chat principal ───────────────────────────

    def _build_voice_chat_buttons(self, parent):
        import llamastation as _ls
        C = _ls.C

        self._voice_chat_ptt = ctk.CTkButton(
            parent, text="🎤", width=44, height=44,
            fg_color=C["card2"], hover_color=C["border"],
            text_color=C["sub"], font=ctk.CTkFont(size=18),
            corner_radius=8, command=None
        )
        self._voice_chat_ptt.pack(side="left", padx=(6, 0))
        self._voice_chat_ptt.bind("<ButtonPress-1>",   self._vchat_ptt_press)
        self._voice_chat_ptt.bind("<ButtonRelease-1>", self._vchat_ptt_release)

        self._voice_chat_listen = ctk.CTkButton(
            parent, text="🔊", width=44, height=44,
            fg_color=C["card2"], hover_color=C["border"],
            text_color=C["sub"], font=ctk.CTkFont(size=18),
            corner_radius=8, command=self._vchat_toggle_listen
        )
        self._voice_chat_listen.pack(side="left", padx=(6, 0))

    # ── Lógica de voz para el chat principal ─────────────────────────────────

    # Flag para saber si el último envío fue por voz (para TTS en respuesta)
    _vchat_was_voice = False
    _vchat_listening = False
    _vchat_stop_ev   = None
    _vchat_ptt_active = False
    _vchat_ptt_chunks = []
    _vchat_ptt_stream = None

    def _vchat_ensure_stop_ev(self):
        if self._vchat_stop_ev is None:
            self._vchat_stop_ev = threading.Event()
        return self._vchat_stop_ev

    # ── PTT para chat principal ───────────────────────────────────────────────

    def _vchat_ptt_press(self, event=None):
        if self._vchat_ptt_active or self._vchat_listening:
            return
        if not self._voice_models_ok:
            messagebox.showwarning("Voz", "Carga los modelos en la pestaña Voz primero.")
            return
        if not self._voice_active_wav:
            messagebox.showwarning("Voz", "Selecciona una voz en la pestaña Voz primero.")
            return
        import llamastation as _ls
        C = _ls.C
        self._vchat_ptt_active = True
        self._vchat_ptt_chunks = []
        self._voice_chat_ptt.configure(fg_color=C["red"], text_color="white", text="⬤")
        self._vchat_ptt_stream = sd.InputStream(
            samplerate=16000, channels=1, dtype="float32",
            blocksize=int(16000 * 0.1),
            callback=self._vchat_ptt_callback)
        self._vchat_ptt_stream.start()

    def _vchat_ptt_callback(self, indata, frames, time_info, status):
        self._vchat_ptt_chunks.append(indata.copy().flatten())

    def _vchat_ptt_release(self, event=None):
        if not self._vchat_ptt_active:
            return
        import llamastation as _ls
        C = _ls.C
        self._vchat_ptt_active = False
        if self._vchat_ptt_stream:
            try:
                self._vchat_ptt_stream.stop()
                self._vchat_ptt_stream.close()
            except Exception:
                pass
            self._vchat_ptt_stream = None
        self._voice_chat_ptt.configure(fg_color=C["card2"], text_color=C["sub"], text="🎤")
        audio = np.concatenate(self._vchat_ptt_chunks) if self._vchat_ptt_chunks else np.zeros(16000)
        self._vchat_ptt_chunks = []
        threading.Thread(target=self._vchat_process_audio, args=(audio,), daemon=True).start()

    # ── Escucha continua para chat principal ─────────────────────────────────

    def _vchat_toggle_listen(self):
        if not self._voice_models_ok:
            messagebox.showwarning("Voz", "Carga los modelos en la pestaña Voz primero.")
            return
        if not self._voice_active_wav:
            messagebox.showwarning("Voz", "Selecciona una voz en la pestaña Voz primero.")
            return
        if self._vchat_listening:
            self._vchat_stop_listen()
        else:
            self._vchat_start_listen()

    def _vchat_start_listen(self):
        import llamastation as _ls
        C = _ls.C
        self._vchat_listening = True
        ev = self._vchat_ensure_stop_ev()
        ev.clear()
        self._voice_chat_listen.configure(fg_color=C["red"], text_color="white", text="🔴")
        threading.Thread(target=self._vchat_listen_loop, daemon=True).start()

    def _vchat_stop_listen(self):
        import llamastation as _ls
        C = _ls.C
        self._vchat_listening = False
        if self._vchat_stop_ev:
            self._vchat_stop_ev.set()
        # Interrumpir TTS si está hablando, sin llamar sd.stop() directamente
        _tts_interrupt.set()
        self.after(0, lambda: self._voice_chat_listen.configure(
            fg_color=C["card2"], text_color=C["sub"], text="🔊"))

    def _vchat_listen_loop(self):
        ev = self._vchat_ensure_stop_ev()
        while self._vchat_listening and not ev.is_set():
            try:
                # Esperar a que la IA termine de hablar antes de escuchar
                while getattr(self, "_vchat_speaking", False):
                    if ev.is_set():
                        break
                    time.sleep(0.1)
                if ev.is_set() or not self._vchat_listening:
                    break
                audio = record_audio_vad(ev)
                if ev.is_set() or not self._vchat_listening:
                    break
                if len(audio) < 8000:
                    continue
                self._vchat_process_audio(audio)
            except Exception:
                time.sleep(0.3)
        self.after(0, lambda: self._vchat_stop_listen() if self._vchat_listening else None)

    # ── Procesar audio → transcribir → enviar al chat → TTS respuesta ────────

    def _vchat_process_audio(self, audio: np.ndarray):
        try:
            lang = self._voice_lang_var.get() if hasattr(self, "_voice_lang_var") else "es"
            text = transcribe(audio, self.settings.get("voice_whisper", "base"), lang)
            if not text.strip():
                return
            # Meter texto en el input del chat y enviar
            def _send_from_voice():
                self._vchat_was_voice = True
                self.chat_input.delete("1.0", "end")
                self.chat_input.insert("1.0", text)
                self._send()
            self.after(0, _send_from_voice)
        except Exception as e:
            self.after(0, lambda: self._append("system", f"[Voz error]: {e}"))

    def _vchat_speak_response(self, text: str):
        """Llamado desde _api cuando la respuesta fue originada por voz."""
        if not self._vchat_was_voice:
            return
        self._vchat_was_voice = False
        if not self._voice_models_ok or not self._voice_active_wav:
            return
        def _do():
            try:
                import re as _re
                clean = _re.sub("(?s)<[|]channel>.*?<channel[|]>", "", text)
                clean = _re.sub("(?s)<think>.*?</think>", "", clean)
                clean = _re.sub("<[|][^|]*[|]>", "", clean).strip() or text
                lang  = self._voice_lang_var.get() if hasattr(self, "_voice_lang_var") else "es"
                speed = float(self._voice_speed_var.get()) if hasattr(self, "_voice_speed_var") else 1.0
                dev   = self.settings.get("voice_device", "cpu")
                self._vchat_speaking = True
                try:
                    synthesize_and_play(clean, self._voice_active_wav, lang, speed, dev)
                finally:
                    self._vchat_speaking = False
                    # Si escucha continua activa, resetear stop_ev para seguir escuchando
                    if self._vchat_listening and self._vchat_stop_ev:
                        self._vchat_stop_ev.clear()
            except Exception:
                self._vchat_speaking = False
        threading.Thread(target=_do, daemon=True).start()
