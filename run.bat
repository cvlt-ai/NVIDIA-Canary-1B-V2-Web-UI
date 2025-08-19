@echo off
REM NVIDIA Canary-1B Flash Gradio Web UI Launcher for Windows
REM This script sets up and launches the web interface

echo 🎤 NVIDIA Canary-1B V2 Gradio Web UI
echo ========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if virtual environment should be created
if "%1"=="--setup" (
    echo 🔧 Setting up virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
    echo 🤖 Setting up model...
    python setup_model.py
    echo ✅ Setup complete!
    echo.
)

REM Activate virtual environment if it exists
if exist "venv" (
    echo 🔄 Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if required packages are installed
echo 🔍 Checking dependencies...
python -c "import gradio, torch, nemo" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Some dependencies are missing. Installing...
    pip install -r requirements.txt
)

REM Check if model needs to be downloaded
if not exist "models\cache" (
    echo 🤖 Model not found. Downloading...
    python setup_model.py
)

echo 🚀 Starting NVIDIA Canary-1B V2 Web UI...
echo 📱 The interface will be available at: http://localhost:7860
echo 🛑 Press Ctrl+C to stop the application
echo.

REM Launch the application
python app.py

pause

