@echo off
echo ========================================
echo   LlamaStation - Instalador
echo ========================================
echo.

:: Moverse a la carpeta donde esta el .bat, sea cual sea
cd /d "%~dp0"
echo Directorio: %~dp0
echo.

:: Verificar que llamastation.py existe aqui
if not exist "llamastation.py" (
    echo ERROR: No se encuentra llamastation.py en esta carpeta.
    echo Pon llamastation.py en: %~dp0
    echo.
    pause
    exit /b 1
)

:: Verificar que llamastation_i18n.py existe aqui
if not exist "llamastation_i18n.py" (
    echo ERROR: No se encuentra llamastation_i18n.py en esta carpeta.
    echo Pon llamastation_i18n.py en: %~dp0
    echo.
    pause
    exit /b 1
)

:: Verificar Python 3.11
py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 3.11 no encontrado. Instala Python 3.11 desde python.org
    pause
    exit /b 1
)

echo Instalando dependencias...
py -3.11 -m pip install customtkinter requests tkinterdnd2 --quiet
py -3.11 -m pip install coqui-tts faster-whisper sounddevice soundfile scipy pydub --quiet
py -3.11 -m pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 --quiet
py -3.11 -m pip install "numpy<2" --quiet

echo.
echo ========================================
echo   Lanzando LlamaStation...
echo ========================================
echo.

py -3.11 llamastation.py 2> error_log.txt
type error_log.txt

echo.
echo ========================================
echo   LlamaStation se cerro con error:  %errorlevel%
echo ========================================
echo.
pause
