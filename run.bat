@echo off
rem One command setup and start for Windows:   run.bat
rem Creates .venv, installs the pinned packages, checks Ollama,
rem downloads the default model and starts the app.
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")
%PY% -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" 2>nul
if errorlevel 1 (
    echo Error: Python 3.10 to 3.13 is needed. Install it from https://www.python.org/downloads/
    exit /b 1
)

if not exist .venv\Scripts\python.exe (
    echo ==^> Creating the Python environment ^(.venv^)
    %PY% -m venv .venv || exit /b 1
)
set "VENV_PY=.venv\Scripts\python.exe"
set PIP_DISABLE_PIP_VERSION_CHECK=1
set PIP_NO_CACHE_DIR=1

%VENV_PY% -c "import streamlit, sentence_transformers, faiss" 2>nul
if errorlevel 1 (
    echo ==^> Installing packages ^(first run only, a few minutes^)
    %VENV_PY% -m pip install -q --upgrade pip
    %VENV_PY% -m pip install -q -r requirements.txt || exit /b 1
)

if "%OLLAMA_URL%"=="" set "OLLAMA_URL=http://localhost:11434"
curl -sf %OLLAMA_URL%/api/tags >nul 2>nul
if errorlevel 1 (
    where ollama >nul 2>nul
    if errorlevel 1 (
        echo Error: Ollama is not installed. Get it from https://ollama.com/download and run run.bat again.
        exit /b 1
    )
    echo ==^> Starting Ollama
    start "" /b ollama serve
    timeout /t 5 /nobreak >nul
)

for /f "delims=" %%m in ('%VENV_PY% -c "import json; print(json.load(open('config.json'))['chosen_model'])"') do set "MODEL=%%m"
curl -sf %OLLAMA_URL%/api/tags | findstr /c:"\"%MODEL%\"" >nul
if errorlevel 1 (
    echo ==^> Downloading the default model %MODEL% ^(first run only^)
    ollama pull %MODEL% || exit /b 1
)

if "%PORT%"=="" set "PORT=8501"
echo ==^> Starting the app at http://localhost:%PORT%
%VENV_PY% -m streamlit run ui\app.py --server.port %PORT% --server.fileWatcherType none
