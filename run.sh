#!/bin/bash

# NVIDIA Canary-1B Flash Gradio Web UI Launcher
# This script sets up and launches the web interface

echo "🎤 NVIDIA Canary-1B v2 Gradio Web UI"
echo "========================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 is not installed or not in PATH"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if virtual environment should be created
if [ "$1" = "--setup" ]; then
    echo "🔧 Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo "🤖 Setting up model..."
    python3 setup_model.py
    echo "✅ Setup complete!"
    echo ""
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python3 -c "import gradio, torch, nemo" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing. Installing..."
    pip3 install -r requirements.txt
fi

# Check if model needs to be downloaded
if [ ! -d "models/cache" ]; then
    echo "🤖 Model not found. Downloading..."
    python3 setup_model.py
fi

echo "🚀 Starting NVIDIA Canary-1B V2 Web UI..."
echo "📱 The interface will be available at: http://localhost:7860"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

# Launch the application
python3 app.py

