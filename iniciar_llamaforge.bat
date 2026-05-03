@echo off
echo ========================================
echo   LlamaForge - Instalador
echo ========================================
echo.

:: Moverse a la carpeta donde esta el .bat, sea cual sea
cd /d "%~dp0"
echo Directorio: %~dp0
echo.

:: Verificar que llama_gui.py existe aqui
if not exist "llama_gui.py" (
    echo ERROR: No se encuentra llama_gui.py en esta carpeta.
    echo Pon llama_gui.py en: %~dp0
    echo.
    pause
    exit /b 1
)

:: Verificar que llamaforge_i18n.py existe aqui
if not exist "llamaforge_i18n.py" (
    echo ERROR: No se encuentra llamaforge_i18n.py en esta carpeta.
    echo Pon llamaforge_i18n.py en: %~dp0
    echo.
    pause
    exit /b 1
)

:: Verificar Python
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python no encontrado. Instala Python 3.10+ desde python.org
    pause
    exit /b 1
)

echo Instalando dependencias...
py -m pip install customtkinter requests tkinterdnd2 --quiet

echo.
echo ========================================
echo   Lanzando LlamaForge...
echo ========================================
echo.

py llama_gui.py 2> error_log.txt
type error_log.txt

echo.
echo ========================================
echo   LlamaForge se cerro con error:  %errorlevel%
echo ========================================
echo.
pause
