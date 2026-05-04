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
echo   Lanzando LlamaStation...
echo ========================================
echo.

py llamastation.py 2> error_log.txt
type error_log.txt

echo.
echo ========================================
echo   LlamaStation se cerro con error:  %errorlevel%
echo ========================================
echo.
pause
