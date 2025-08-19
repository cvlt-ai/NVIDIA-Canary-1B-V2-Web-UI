# Installation Guide - NVIDIA Canary-1B V@ Gradio Web UI

This guide provides step-by-step instructions for installing and running the NVIDIA Canary-1B Flash Gradio Web UI on different operating systems.

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher (3.11 recommended)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space for model and dependencies
- **Internet**: Required for initial model download

### Recommended Requirements
- **GPU**: NVIDIA GPU with CUDA support (for optimal performance)
- **VRAM**: 4GB+ for GPU acceleration
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)

## Installation Methods

### Method 1: Quick Start (Recommended)

#### For Linux/macOS:
```bash
# 1. Clone or download the application
git clone https://github.com/cvlt-ai/NVIDIA-Canary-1B-V2-Web-UI
cd NVIDIA-Canary-1B-V2-Web-UI

# 2. Run the setup script
./run.sh --setup

# 3. Launch the application
./run.sh
```

#### For Windows:
```cmd
# 1. Download and extract the application
# 2. Open Command Prompt in the application folder
# 3. Run the setup script
run.bat --setup

# 4. Launch the application
run.bat
```

### Method 2: Manual Installation

#### Step 1: Install Python
- **Windows**: Download from [python.org](https://python.org) and install
- **macOS**: Use Homebrew: `brew install python3` or download from python.org
- **Linux**: Usually pre-installed, or use: `sudo apt install python3 python3-pip`

#### Step 2: Create Virtual Environment (Recommended)
```bash
python3 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Launch Application
```bash
python app.py
```

## GPU Support (Optional but Recommended)

### NVIDIA GPU Setup
1. **Install NVIDIA Drivers**: Download latest drivers from NVIDIA website
2. **Install CUDA Toolkit**: Download CUDA 11.8 or 12.x from NVIDIA
3. **Verify Installation**:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### AMD GPU Support
AMD GPUs are supported through ROCm on Linux. Windows users should use CPU mode.

## Troubleshooting

### Common Issues

#### 1. "No module named 'nemo'" Error
**Solution**: Install NeMo toolkit
```bash
pip install nemo_toolkit
```

#### 2. "CUDA out of memory" Error
**Solutions**:
- Reduce batch size in the code
- Use CPU mode by setting `CUDA_VISIBLE_DEVICES=""`
- Close other GPU-intensive applications

#### 3. Model Download Fails
**Solutions**:
- Check internet connection
- Ensure sufficient disk space (5GB+)
- Try running with VPN if in restricted region

#### 4. "Permission denied" on Linux/macOS
**Solution**: Make script executable
```bash
chmod +x run.sh
```

#### 5. Python Version Issues
**Solution**: Ensure Python 3.8+ is installed
```bash
python3 --version
# Should show 3.8.0 or higher
```

### Performance Optimization

#### For Better Performance:
1. **Use GPU**: Ensure CUDA is properly installed
2. **Increase RAM**: Close unnecessary applications
3. **SSD Storage**: Install on SSD for faster model loading
4. **Batch Processing**: Process multiple files together

#### For Lower Resource Usage:
1. **CPU Mode**: Set `CUDA_VISIBLE_DEVICES=""`
2. **Reduce Model Precision**: Use half-precision if supported
3. **Smaller Batch Size**: Reduce memory usage

## Verification

After installation, verify the setup:

1. **Launch the application**
2. **Open browser** to `http://localhost:7860`
3. **Check interface** loads properly
4. **Test with sample audio** (optional)

### Expected Interface Elements:
- ✅ Audio upload area with drag-and-drop
- ✅ ASR/AST mode selection
- ✅ Language dropdowns (English, German, French, Spanish)
- ✅ Inference mode options (Short-form/Long-form)
- ✅ Timestamp and punctuation controls
- ✅ Results display areas

## Getting Help

### Log Files
Check console output for error messages when running the application.

### Common Log Locations:
- **Application logs**: Console output where you ran the script
- **Model cache**: `~/.cache/huggingface/` (Linux/macOS) or `%USERPROFILE%\.cache\huggingface\` (Windows)

### Support Resources:
1. **README.md**: Comprehensive usage guide
2. **GitHub Issues**: Report bugs and get help
3. **NVIDIA NeMo Documentation**: For model-specific issues
4. **Gradio Documentation**: For interface-related questions

## Uninstallation

To remove the application:

1. **Delete application folder**
2. **Remove virtual environment** (if created)
3. **Clear model cache** (optional):
   ```bash
   rm -rf ~/.cache/huggingface/transformers/nvidia--canary-1b-flash
   ```

## Security Notes

- The application runs locally on your machine
- No data is sent to external servers (except initial model download)
- Audio files are processed locally and temporarily stored
- Model weights are cached locally for faster subsequent launches


## Note about using Timestamps

--to use timestamps do the following 

in terminal : 

pip uninstall nemo_toolkit
pip install git+https://github.com/NVIDIA/NeMo.git@main
pip install texterrors
