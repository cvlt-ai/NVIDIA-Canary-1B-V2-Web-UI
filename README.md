# NVIDIA Canary-1B v2 Gradio Web UI

A comprehensive web interface for the **NVIDIA Canary-1B v2** multilingual automatic speech recognition (ASR) and speech translation (AST) model.

##  What's New in v2

### Enhanced Language Support
- **25 European Languages** (vs 4 in v1)
- **21 New Languages Added**: Bulgarian, Croatian, Czech, Danish, Dutch, Estonian, Finnish, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovenian, Swedish, Russian, Ukrainian

### Performance Improvements
- **10x Faster** than comparable models
- **State-of-the-art Performance** among similar-sized models
- **Comparable Quality** to 3x larger models
- **978M Parameters** (vs 883M in v1)

### Technical Enhancements
- **FastConformer Encoder** + **Transformer Decoder** architecture
- **Enhanced Tokenizer** with 16,384 vocabulary optimized across all 25 languages
- **Improved Timestamps** for both transcription and translation outputs
- **Better Punctuation & Capitalization** handling

##  Supported Languages (25)

**European Languages:**
- **Western Europe**: English, German, French, Spanish, Dutch, Italian, Portuguese
- **Nordic**: Danish, Finnish, Swedish, Estonian, Latvian, Lithuanian
- **Central Europe**: Czech, Slovak, Polish, Hungarian, Slovenian, Croatian
- **Eastern Europe**: Bulgarian, Romanian, Russian, Ukrainian
- **Mediterranean**: Greek, Maltese

##  Features

- ** Drag & Drop Audio Upload** - Support for WAV, MP3, FLAC, M4A, OGG formats
- ** 25 Language Support** - Comprehensive European language coverage
- ** Dual Mode Operation**:
  - **ASR Mode**: Speech Recognition (speech → text in same language)
  - **AST Mode**: Speech Translation (speech → text in different language)
- ** Inference Options**:
  - **Short-form**: Optimized for audio < 40 seconds
  - **Long-form**: Chunked processing for longer audio files
- ** Advanced Output Options**:
  - Word-level and segment-level timestamps
  - Automatic punctuation and capitalization
  - Formatted output with proper spacing
- ** Enhanced Performance**: 10x faster processing with state-of-the-art accuracy

##  Quick Start

### Option 1: Automated Setup (Recommended)

**Linux/macOS:**
```bash
./run.sh --setup    # Install dependencies and download model
./run.sh            # Launch the application
```

**Windows:**
```cmd
run.bat --setup     # Install dependencies and download model
run.bat              # Launch the application
```

### Option 2: Manual Setup

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download Model:**
```bash
python setup_model.py
```

3. **Launch Application:**
```bash
python app.py
```
## Note about using Timestamps

!!!!to use timestamps do the following 

in terminal : 
```bash
pip uninstall nemo_toolkit
pip install git+https://github.com/NVIDIA/NeMo.git@main
pip install texterrors
```
##  Requirements

### System Requirements
- **Python**: 3.8+ (3.11 recommended)
- **Memory**: 8GB+ RAM recommended
- **Storage**: 5GB+ free space for model cache
- **GPU**: Optional (CUDA-compatible for faster inference)

### Dependencies
- **Core**: `gradio>=4.0.0`, `torch>=1.13.0`, `torchaudio`
- **Audio**: `librosa`, `soundfile`
- **NeMo**: `nemo_toolkit[asr]`, `lightning>=2.0.0`
- **Additional**: See `requirements.txt` for complete list

##  Usage Guide

### Basic Workflow
1. **Upload Audio**: Drag and drop your audio file
2. **Select Mode**: Choose ASR (recognition) or AST (translation)
3. **Configure Languages**: Set source and target languages
4. **Choose Options**: Enable timestamps, punctuation as needed
5. **Process**: Click "Process Audio" and wait for results

### ASR Mode (Speech Recognition)
- Transcribes speech to text in the **same language**
- Perfect for creating transcripts, subtitles, or notes
- Supports all 25 languages

### AST Mode (Speech Translation)
- Translates speech from one language to another
- **English ↔ 24 Languages**: Bidirectional translation
- **24 Languages → English**: Any supported language to English
- **English → 24 Languages**: English to any supported language

### Advanced Features
- **Timestamps**: Get word-level and segment-level timing information
- **Long-form Processing**: Automatic chunking for files > 40 seconds
- **Quality Control**: Automatic punctuation and capitalization
- **Batch Processing**: Process multiple files efficiently

##  Configuration

### Audio Format Support
- **Recommended**: WAV (16kHz, mono)
- **Supported**: MP3, FLAC, M4A, OGG
- **Auto-conversion**: Files automatically converted to optimal format

### Performance Tuning
- **GPU Acceleration**: Automatically detected and used when available
- **Batch Size**: Optimized automatically based on available memory
- **Caching**: Models cached locally for faster subsequent loads

##  Model Information

- **Model**: NVIDIA Canary-1B v2
- **Parameters**: 978 million (vs 883M in v1)
- **Architecture**: FastConformer Encoder + Transformer Decoder
- **Vocabulary**: 16,384 tokens optimized across 25 languages
- **License**: CC-BY-4.0 (Commercial use allowed)
- **Performance**: 10x faster than comparable models

##  Troubleshooting

### Common Issues

**Model Download Fails:**
- Check internet connection
- Ensure 5GB+ free disk space
- Try running `python setup_model.py` manually

**Audio Processing Errors:**
- Verify audio file format is supported
- Check file isn't corrupted
- Try converting to WAV format first

**Memory Issues:**
- Close other applications
- Use shorter audio files for testing
- Enable GPU acceleration if available

**Dependency Errors:**
- Run `pip install -r requirements.txt`
- Check Python version (3.8+ required)
- Consider using virtual environment

### Getting Help
- Check console output for detailed error messages
- Verify all dependencies are installed correctly
- Ensure model downloaded successfully
- Try the alternative model loader: `python alternative_model_loader.py`

##  License

This project is released under the MIT License. The NVIDIA Canary-1B v2 model is licensed under CC-BY-4.0.

##  Acknowledgments

- **NVIDIA NeMo Team** for the outstanding Canary-1B v2 model
- **Gradio Team** for the excellent web interface framework
- **HuggingFace** for model hosting and distribution
- **Open Source Community** for various supporting libraries

---

**Enjoy enhanced multilingual speech processing with NVIDIA Canary-1B v2!** 🎉

