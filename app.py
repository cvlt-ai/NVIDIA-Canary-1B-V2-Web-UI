#!/usr/bin/env python3
"""
NVIDIA Canary-1B v2 Gradio Web UI
A comprehensive web interface for the NVIDIA Canary-1B v2 multilingual ASR/AST model.

Features:
- Drag and drop audio file upload
- 25 European language support (expanded from 4 in v1)
- Language selection (source and target)
- Long-form and short-form inference modes
- Timestamp generation (word-level and segment-level)
- ASR (Automatic Speech Recognition) and AST (Automatic Speech Translation) modes
- Punctuation and capitalization control
- Enhanced performance: 10x faster than comparable models
- State-of-the-art quality comparable to 3x larger models
"""

import gradio as gr
import torch
import torchaudio
import librosa
import soundfile as sf
import tempfile
import os
import json
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Global model variable
canary_model = None

# Supported languages for NVIDIA Canary-1B v2 (25 European languages)
LANGUAGES = {
    "Bulgarian": "bg",
    "Croatian": "hr", 
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Hungarian": "hu",
    "Italian": "it",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Maltese": "mt",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Swedish": "sv",
    "Russian": "ru",
    "Ukrainian": "uk"
}

# Language options for dropdowns
LANGUAGE_OPTIONS = list(LANGUAGES.keys())

def load_model():
    """Load the NVIDIA Canary-1B Flash model with robust error handling."""
    global canary_model
    
    try:
        from nemo.collections.asr.models import EncDecMultiTaskModel
        
        if canary_model is None:
            print("Loading NVIDIA Canary-1B Flash model...")
            print("This may take a few minutes on first run to download the model...")
            
            # Create models directory if it doesn't exist
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Set cache directory for model downloads using environment variables
            cache_dir = os.path.join(models_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Set environment variables for HuggingFace cache
            os.environ['HF_HOME'] = cache_dir
            if 'TRANSFORMERS_CACHE' in os.environ:
                del os.environ['TRANSFORMERS_CACHE']  # Remove deprecated variable
            os.environ['HF_DATASETS_CACHE'] = cache_dir
            
            print(f"Model cache directory: {cache_dir}")
            
            # Check if local model exists
            local_model_path = os.path.join(models_dir, "canary-1b-v2")
            
            try:
                # Try to load from local path first if it exists
                if os.path.exists(local_model_path) and os.listdir(local_model_path):
                    print(f"Loading model from local path: {local_model_path}")
                    canary_model = EncDecMultiTaskModel.from_pretrained(local_model_path)
                else:
                    # Download from HuggingFace Hub
                    print("Downloading model from HuggingFace Hub...")
                    canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-v2')
                
                # Configure decoding strategy
                decode_cfg = canary_model.cfg.decoding
                decode_cfg.beam.beam_size = 1
                canary_model.change_decoding_strategy(decode_cfg)
                
                print("Model loaded successfully!")
                print(f"Model cached in: {cache_dir}")
                
            except Exception as model_error:
                print(f"Model loading failed: {model_error}")
                # Try alternative loading method
                print("Attempting alternative loading method...")
                try:
                    # Install huggingface_hub if not available
                    import subprocess
                    import sys
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
                    
                    from huggingface_hub import snapshot_download
                    
                    # Download model files manually
                    print("Downloading model files manually...")
                    model_path = snapshot_download(
                        repo_id="nvidia/canary-1b-v2",
                        cache_dir=cache_dir,
                        local_dir=local_model_path,
                        local_dir_use_symlinks=False
                    )
                    
                    # Try loading again
                    canary_model = EncDecMultiTaskModel.from_pretrained(model_path)
                    
                    # Configure decoding strategy
                    decode_cfg = canary_model.cfg.decoding
                    decode_cfg.beam.beam_size = 1
                    canary_model.change_decoding_strategy(decode_cfg)
                    
                    print("Model loaded successfully using alternative method!")
                    
                except Exception as alt_error:
                    raise Exception(f"All loading methods failed. Original: {model_error}, Alternative: {alt_error}")
        
        return True, "Model loaded successfully!"
        
    except ImportError as e:
        error_msg = f"Missing dependency: {str(e)}. Please install: pip install -r requirements.txt"
        print(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        print(error_msg)
        print("\nTroubleshooting tips:")
        print("1. Ensure you have a stable internet connection")
        print("2. Check that you have sufficient disk space (5GB+)")
        print("3. Try running: pip install -r requirements.txt")
        print("4. Try running: python alternative_model_loader.py")
        print("5. Restart the application if the error persists")
        return False, error_msg

def preprocess_audio(audio_file: str, target_sr: int = 16000) -> str:
    """
    Preprocess audio file to meet model requirements.
    
    Args:
        audio_file: Path to input audio file
        target_sr: Target sample rate (16000 Hz for Canary)
    
    Returns:
        Path to preprocessed audio file
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_file, sr=None)
        
        # Resample if necessary
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # Save preprocessed audio
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, audio, target_sr)
        
        return temp_file.name
        
    except Exception as e:
        raise Exception(f"Error preprocessing audio: {str(e)}")

def create_manifest(
    audio_path: str,
    source_lang: str,
    target_lang: str,
    enable_timestamps: bool,
    enable_pnc: bool
) -> str:
    """
    Create a JSONL manifest file for the model.
    
    Args:
        audio_path: Path to audio file
        source_lang: Source language code
        target_lang: Target language code
        enable_timestamps: Whether to generate timestamps
        enable_pnc: Whether to enable punctuation and capitalization
    
    Returns:
        Path to manifest file
    """
    import librosa
    
    # Get audio duration using librosa
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
        duration = len(audio_data) / sr
    except Exception as e:
        print(f"Warning: Could not determine audio duration: {e}")
        # Fallback: estimate duration from file size (rough approximation)
        import os
        file_size = os.path.getsize(audio_path)
        # Rough estimate: assume 16-bit, 16kHz mono audio
        duration = file_size / (2 * 16000)  # bytes / (bytes_per_sample * sample_rate)
    
    manifest_data = {
        "audio_filepath": audio_path,
        "duration": duration,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "pnc": "yes" if enable_pnc else "no",
        "answer": "na"  # Required field for NeMo
    }
    
    # Add timestamp configuration if requested
    if enable_timestamps:
        manifest_data["taskname"] = "asr_with_timestamps"
        manifest_data["timestamps"] = "yes"
        manifest_data["word_timestamps"] = "yes"
        print(f"Debug: Timestamps enabled in manifest: {enable_timestamps}")
    else:
        manifest_data["taskname"] = "asr"
        print(f"Debug: Timestamps disabled in manifest")
    
    print(f"Debug: Manifest data: {manifest_data}")
    
    # Create temporary manifest file
    temp_manifest = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(manifest_data, temp_manifest)
    temp_manifest.flush()
    
    return temp_manifest.name

def transcribe_audio(
    audio_file,
    source_language: str,
    target_language: str,
    inference_mode: str,
    enable_timestamps: bool,
    enable_pnc: bool,
    asr_ast_mode: str
) -> Tuple[str, str, str]:
    """
    Main transcription function.
    
    Args:
        audio_file: Uploaded audio file
        source_language: Source language selection
        target_language: Target language selection  
        inference_mode: "Short-form" or "Long-form"
        enable_timestamps: Whether to generate timestamps
        enable_pnc: Whether to enable punctuation and capitalization
        asr_ast_mode: "ASR" or "AST" mode
    
    Returns:
        Tuple of (transcription_text, timestamps_info, status_message)
    """
    global canary_model
    
    try:
        # Validate inputs
        if audio_file is None:
            return "", "", "❌ Please upload an audio file."
        
        if canary_model is None:
            success, msg = load_model()
            if not success:
                return "", "", f"❌ {msg}"
        
        # Convert language names to codes
        source_lang_code = LANGUAGES[source_language]
        target_lang_code = LANGUAGES[target_language]
        
        # For ASR mode, source and target languages should be the same
        if asr_ast_mode == "ASR":
            target_lang_code = source_lang_code
            target_language = source_language
        
        # Preprocess audio
        status_msg = "🔄 Preprocessing audio..."
        print(status_msg)
        
        preprocessed_audio = preprocess_audio(audio_file)
        
        # Determine processing method based on inference mode
        if inference_mode == "Short-form":
            # Use direct transcription for short-form
            status_msg = "🔄 Transcribing audio (short-form)..."
            print(status_msg)
            
            try:
                if source_lang_code == target_lang_code and not enable_timestamps:
                    # Simple ASR case - use direct audio input
                    output = canary_model.transcribe(
                        [preprocessed_audio],
                        batch_size=1,
                        pnc='yes' if enable_pnc else 'no'
                    )
                    transcription = output[0] if output else ""
                    timestamps_info = ""
                    
                else:
                    # Use manifest for more complex cases (AST, timestamps)
                    manifest_path = create_manifest(
                        preprocessed_audio,
                        source_lang_code,
                        target_lang_code,
                        enable_timestamps,
                        enable_pnc
                    )
                    
                    # Use the correct NeMo API for transcription with timestamps
                    if enable_timestamps:
                        print(f"Debug: Calling transcribe with timestamps=True")
                        output = canary_model.transcribe(
                            [preprocessed_audio],  # Pass as list
                            source_lang=source_lang_code,
                            target_lang=target_lang_code,
                            timestamps=True,
                            batch_size=1
                        )
                    else:
                        print(f"Debug: Calling transcribe without timestamps")
                        output = canary_model.transcribe(
                            [preprocessed_audio],  # Pass as list
                            source_lang=source_lang_code,
                            target_lang=target_lang_code,
                            timestamps=False,
                            batch_size=1
                        )
                    
                    print(f"Debug: Model output type: {type(output)}")
                    print(f"Debug: Model output length: {len(output) if output else 0}")
                    
                    if output and len(output) > 0:
                        # Handle different output formats
                        result = output[0]
                        print(f"Debug: Result type: {type(result)}")
                        print(f"Debug: Result content: {result}")
                        
                        # Extract text from Hypothesis object or string
                        if hasattr(result, 'text'):
                            transcription = result.text
                        elif isinstance(result, str):
                            transcription = result
                        else:
                            transcription = str(result)
                        
                        # Extract timestamps using the correct NeMo format
                        timestamps_info = ""
                        if enable_timestamps:
                            print(f"Debug: Attempting to extract timestamps using NeMo format...")
                            try:
                                # Use the format from NeMo documentation
                                if hasattr(result, 'timestamp') and result.timestamp:
                                    timestamp_data = result.timestamp
                                    print(f"Debug: Found timestamp data: {timestamp_data}")
                                    
                                    # Check if timestamp data is empty
                                    if (isinstance(timestamp_data, dict) and 
                                        all(not timestamp_data.get(key, []) for key in ['word', 'segment', 'char'])):
                                        timestamps_info = "⚠️ Timestamps were requested but the model returned empty timestamp arrays. This may be due to:\n• Model configuration not supporting timestamps\n• Audio too short or unclear\n• NeMo version compatibility issues"
                                        print("Debug: All timestamp arrays are empty")
                                    else:
                                        # Extract word-level timestamps (ASR only)
                                        if 'word' in timestamp_data and timestamp_data['word'] and asr_ast_mode == "ASR":
                                            word_timestamps = timestamp_data['word']
                                            timestamps_info += "📍 Word-level timestamps:\\n"
                                            for stamp in word_timestamps:
                                                if isinstance(stamp, dict):
                                                    start = stamp.get('start', 'N/A')
                                                    end = stamp.get('end', 'N/A')
                                                    word = stamp.get('word', 'N/A')
                                                    timestamps_info += f"  {start}s - {end}s : {word}\\n"
                                                else:
                                                    timestamps_info += f"  {stamp}\\n"
                                        
                                        # Extract segment-level timestamps (both ASR and AST)
                                        if 'segment' in timestamp_data and timestamp_data['segment']:
                                            segment_timestamps = timestamp_data['segment']
                                            if timestamps_info:  # Add separator if word timestamps were added
                                                timestamps_info += "\\n"
                                            timestamps_info += "📍 Segment-level timestamps:\\n"
                                            for stamp in segment_timestamps:
                                                if isinstance(stamp, dict):
                                                    start = stamp.get('start', 'N/A')
                                                    end = stamp.get('end', 'N/A')
                                                    segment = stamp.get('segment', 'N/A')
                                                    timestamps_info += f"  {start}s - {end}s : {segment}\\n"
                                                else:
                                                    timestamps_info += f"  {stamp}\\n"
                                        
                                        if not timestamps_info:
                                            timestamps_info = f"⚠️ Timestamp data found but arrays are empty or in unexpected format"
                                            
                                else:
                                    timestamps_info = "⚠️ No timestamp data found in result object"
                                    print(f"Debug: Result has timestamp attribute: {hasattr(result, 'timestamp')}")
                                    if hasattr(result, 'timestamp'):
                                        print(f"Debug: Timestamp value: {result.timestamp}")
                                        
                            except Exception as ts_error:
                                print(f"Error extracting timestamps: {ts_error}")
                                timestamps_info = f"⚠️ Error extracting timestamps: {str(ts_error)}"
                        else:
                            print(f"Debug: Timestamps not requested")
                    else:
                        transcription = ""
                        timestamps_info = ""
                    
                    # Clean up manifest file
                    if os.path.exists(manifest_path):
                        os.unlink(manifest_path)
                        
            except Exception as short_form_error:
                print(f"Short-form transcription error: {short_form_error}")
                # Fallback to simple transcription
                try:
                    print("Trying fallback transcription method...")
                    output = canary_model.transcribe([preprocessed_audio])
                    transcription = output[0] if output else "Error: Could not transcribe audio"
                    timestamps_info = ""
                except Exception as fallback_error:
                    print(f"Fallback method also failed: {fallback_error}")
                    transcription = f"Error: {str(short_form_error)}"
                    timestamps_info = ""
        
        else:
            # Long-form inference
            status_msg = "🔄 Transcribing audio (long-form)..."
            print(status_msg)
            
            try:
                # Use the correct NeMo API for long-form transcription with timestamps
                if enable_timestamps:
                    print(f"Debug: Long-form calling transcribe with timestamps=True")
                    output = canary_model.transcribe(
                        [preprocessed_audio],  # Pass as list
                        source_lang=source_lang_code,
                        target_lang=target_lang_code,
                        timestamps=True,
                        batch_size=1
                    )
                else:
                    print(f"Debug: Long-form calling transcribe without timestamps")
                    output = canary_model.transcribe(
                        [preprocessed_audio],  # Pass as list
                        source_lang=source_lang_code,
                        target_lang=target_lang_code,
                        timestamps=False,
                        batch_size=1
                    )
                
                if output and len(output) > 0:
                    # Handle different output formats
                    result = output[0]
                    print(f"Debug: Long-form result type: {type(result)}")
                    print(f"Debug: Long-form result: {result}")
                    
                    # Extract text from Hypothesis object or string
                    if hasattr(result, 'text'):
                        transcription = result.text
                    elif isinstance(result, str):
                        transcription = result
                    else:
                        transcription = str(result)
                    
                    # Extract timestamps using the correct NeMo format
                    timestamps_info = ""
                    if enable_timestamps:
                        print(f"Debug: Long-form attempting to extract timestamps using NeMo format...")
                        try:
                            # Use the format from NeMo documentation
                            if hasattr(result, 'timestamp') and result.timestamp:
                                timestamp_data = result.timestamp
                                print(f"Debug: Long-form found timestamp data: {timestamp_data}")
                                
                                # Check if timestamp data is empty
                                if (isinstance(timestamp_data, dict) and 
                                    all(not timestamp_data.get(key, []) for key in ['word', 'segment', 'char'])):
                                    timestamps_info = "⚠️ Timestamps were requested but the model returned empty timestamp arrays. This may be due to:\n• Model configuration not supporting timestamps\n• Audio too short or unclear\n• NeMo version compatibility issues"
                                    print("Debug: All timestamp arrays are empty")
                                else:
                                    # Extract word-level timestamps (ASR only)
                                    if 'word' in timestamp_data and timestamp_data['word'] and asr_ast_mode == "ASR":
                                        word_timestamps = timestamp_data['word']
                                        timestamps_info += " Word-level timestamps:\\n"
                                        for stamp in word_timestamps:
                                            if isinstance(stamp, dict):
                                                start = stamp.get('start', 'N/A')
                                                end = stamp.get('end', 'N/A')
                                                word = stamp.get('word', 'N/A')
                                                timestamps_info += f"  {start}s - {end}s : {word}\\n"
                                            else:
                                                timestamps_info += f"  {stamp}\\n"
                                    
                                    # Extract segment-level timestamps (both ASR and AST)
                                    if 'segment' in timestamp_data and timestamp_data['segment']:
                                        segment_timestamps = timestamp_data['segment']
                                        if timestamps_info:  # Add separator if word timestamps were added
                                            timestamps_info += "\\n"
                                        timestamps_info += " Segment-level timestamps:\\n"
                                        for stamp in segment_timestamps:
                                            if isinstance(stamp, dict):
                                                start = stamp.get('start', 'N/A')
                                                end = stamp.get('end', 'N/A')
                                                segment = stamp.get('segment', 'N/A')
                                                timestamps_info += f"  {start}s - {end}s : {segment}\\n"
                                            else:
                                                timestamps_info += f"  {stamp}\\n"
                                    
                                    if not timestamps_info:
                                        timestamps_info = f"⚠️ Timestamp data found but arrays are empty or in unexpected format"
                                        
                            else:
                                timestamps_info = "⚠️ No timestamp data found in result object"
                                print(f"Debug: Long-form result has timestamp attribute: {hasattr(result, 'timestamp')}")
                                if hasattr(result, 'timestamp'):
                                    print(f"Debug: Long-form timestamp value: {result.timestamp}")
                                    
                        except Exception as ts_error:
                            print(f"Long-form error extracting timestamps: {ts_error}")
                            timestamps_info = f"⚠️ Error extracting timestamps: {str(ts_error)}"
                    else:
                        print(f"Debug: Long-form timestamps not requested")
                else:
                    transcription = ""
                    timestamps_info = ""
                
                # Clean up temporary files
                try:
                    import os
                    if os.path.exists(preprocessed_audio):
                        os.unlink(preprocessed_audio)
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temporary audio file: {cleanup_error}")
                
                return transcription, timestamps_info, "✅ Transcription completed successfully!"
                
            except Exception as long_form_error:
                print(f"Long-form transcription error: {long_form_error}")
                
                # Fallback: try manifest-based approach
                try:
                    print("Trying fallback transcription method...")
                    manifest_path = create_manifest(
                        preprocessed_audio,
                        source_lang_code,
                        target_lang_code,
                        enable_timestamps,
                        enable_pnc
                    )
                    
                    output = canary_model.transcribe(
                        manifest_path,
                        batch_size=1,
                        verbose=True
                    )
                    
                    if output and len(output) > 0:
                        result = output[0]
                        if hasattr(result, 'text'):
                            transcription = result.text
                        elif isinstance(result, str):
                            transcription = result
                        else:
                            transcription = str(result)
                        
                        timestamps_info = "⚠️ Using fallback method - timestamps may not be available"
                    else:
                        transcription = f"Error: {str(long_form_error)}"
                        timestamps_info = ""
                    
                    # Clean up manifest file
                    try:
                        import os
                        if os.path.exists(manifest_path):
                            os.unlink(manifest_path)
                    except:
                        pass
                        
                except Exception as fallback_error:
                    print(f"Fallback transcription also failed: {fallback_error}")
                    transcription = f"Error: {str(long_form_error)}"
                    timestamps_info = ""
                
                # Clean up manifest file
                if os.path.exists(manifest_path):
                    os.unlink(manifest_path)
                    
            except Exception as long_form_error:
                print(f"Long-form transcription error: {long_form_error}")
                # Fallback to simple transcription
                try:
                    print("Trying fallback transcription method...")
                    output = canary_model.transcribe([preprocessed_audio])
                    transcription = output[0] if output else "Error: Could not transcribe audio"
                    timestamps_info = ""
                except Exception as fallback_error:
                    print(f"Fallback method also failed: {fallback_error}")
                    transcription = f"Error: {str(long_form_error)}"
                    timestamps_info = ""
        
        # Clean up preprocessed audio file
        os.unlink(preprocessed_audio)
        
        # Prepare final status message
        mode_info = f"{asr_ast_mode} ({source_language}"
        if asr_ast_mode == "AST":
            mode_info += f" → {target_language}"
        mode_info += ")"
        
        final_status = f"✅ Transcription completed! Mode: {mode_info}, Form: {inference_mode}"
        if enable_timestamps:
            final_status += ", Timestamps: Enabled"
        if enable_pnc:
            final_status += ", PnC: Enabled"
        
        return transcription, timestamps_info, final_status
        
    except Exception as e:
        error_msg = f"❌ Error during transcription: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return "", "", error_msg

def update_target_language_visibility(asr_ast_mode: str):
    """Update target language dropdown visibility based on ASR/AST mode."""
    if asr_ast_mode == "ASR":
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)

def create_interface():
    """Create the Gradio interface."""
    
    # Custom CSS for better styling and proper dark/light mode support
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .header {
        text-align: center;
        margin-bottom: 30px;
    }
    .feature-box {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .status-box {
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        font-family: monospace;
    }
    
    /* Light mode - default Gradio styling */
    .light textarea,
    .light input[type="text"],
    body:not(.dark) textarea,
    body:not(.dark) input[type="text"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-color: #d1d5db !important;
    }
    
    .light .feature-box,
    body:not(.dark) .feature-box {
        background-color: #f9f9f9;
        border-color: #e0e0e0;
        color: #000000;
    }
    
    /* Dark mode - only apply when dark class is present */
    .dark textarea,
    .dark input[type="text"],
    .dark .gr-textbox textarea,
    .dark .gr-textbox input,
    .dark .gr-form textarea,
    .dark .gr-form input {
        background-color: #374151 !important;
        color: #f9fafb !important;
        border-color: #6b7280 !important;
    }
    
    .dark .feature-box {
        background-color: #1f2937 !important;
        border-color: #374151 !important;
        color: #f9fafb !important;
    }
    
    /* Dark mode placeholder text */
    .dark textarea::placeholder,
    .dark input[type="text"]::placeholder,
    .dark .gr-textbox textarea::placeholder,
    .dark .gr-textbox input::placeholder {
        color: #9ca3af !important;
    }
    
    /* Light mode placeholder text */
    .light textarea::placeholder,
    .light input[type="text"]::placeholder,
    body:not(.dark) textarea::placeholder,
    body:not(.dark) input[type="text"]::placeholder {
        color: #6b7280 !important;
    }
    
    /* Ensure proper contrast in both modes */
    .dark .gr-button {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
    }
    
    .light .gr-button,
    body:not(.dark) .gr-button {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
    }
    """
    with gr.Blocks(css=css, title="NVIDIA Canary-1B v2 Web UI") as interface:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1> NVIDIA Canary-1B v2 Web UI</h1>
            <p>Enhanced Multilingual Automatic Speech Recognition and Translation</p>
            <p><em>Supports 25 European Languages • 10x Faster • State-of-the-art Performance</em></p>
            <p><em>CVLT AI </em></p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                
                # Audio Upload Section
                gr.HTML('<div class="feature-box"><h3> Audio Upload</h3></div>')
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    format="wav"
                )
                
                # Mode Selection
                gr.HTML('<div class="feature-box"><h3> Processing Mode</h3></div>')
                asr_ast_mode = gr.Radio(
                    choices=["ASR", "AST"],
                    value="ASR",
                    label="Mode",
                    info="ASR: Speech Recognition | AST: Speech Translation"
                )
                
                # Language Selection
                gr.HTML('<div class="feature-box"><h3> Language Settings</h3></div>')
                source_language = gr.Dropdown(
                    choices=LANGUAGE_OPTIONS,
                    value="English",
                    label="Source Language",
                    info="Language of the input audio"
                )
                
                target_language = gr.Dropdown(
                    choices=LANGUAGE_OPTIONS,
                    value="English",
                    label="Target Language",
                    info="Language for translation output (AST mode only)",
                    visible=False
                )
                
                # Inference Options
                gr.HTML('<div class="feature-box"><h3> Inference Options</h3></div>')
                inference_mode = gr.Radio(
                    choices=["Short-form", "Long-form"],
                    value="Short-form",
                    label="Inference Mode",
                    info="Short-form: <40s audio | Long-form: >40s audio with chunking"
                )
                
                # Additional Options
                gr.HTML('<div class="feature-box"><h3> Output Options</h3></div>')
                enable_timestamps = gr.Checkbox(
                    label="Generate Timestamps",
                    value=False,
                    info="Generate word-level and segment-level timestamps"
                )
                
                enable_pnc = gr.Checkbox(
                    label="Punctuation & Capitalization",
                    value=True,
                    info="Include punctuation and proper capitalization"
                )
                
                # Process Button
                process_btn = gr.Button(
                    " Process Audio",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=1):
                
                # Results Section
                gr.HTML('<div class="feature-box"><h3> Results</h3></div>')
                
                # Status Display
                status_output = gr.Textbox(
                    label="Status",
                    value="Ready to process audio...",
                    interactive=False,
                    lines=2
                )
                
                # Transcription Output
                transcription_output = gr.Textbox(
                    label="Transcription/Translation",
                    placeholder="Transcription will appear here...",
                    lines=8,
                    interactive=False
                )
                
                # Timestamps Output
                timestamps_output = gr.Textbox(
                    label="Timestamps",
                    placeholder="Timestamps will appear here when enabled...",
                    lines=8,
                    interactive=False,
                    visible=False
                )
                
                # Model Info
                gr.HTML("""
                <div class="feature-box">
                    <h3> Model Information</h3>
                    <ul>
                        <li><strong>Model:</strong> NVIDIA Canary-1B v2</li>
                        <li><strong>Parameters:</strong> 978M (vs 883M in v1)</li>
                        <li><strong>Languages:</strong> 25 European Languages</li>
                        <li><strong>Performance:</strong> 10x faster than comparable models</li>
                        <li><strong>Quality:</strong> Comparable to 3x larger models</li>
                        <li><strong>Capabilities:</strong> ASR, AST, Word/Segment Timestamps</li>
                        <li><strong>Architecture:</strong> FastConformer + Transformer</li>
                        <li><strong>License:</strong> CC-BY-4.0</li>
                    </ul>
                    <details>
                        <summary><strong>Supported Languages (25)</strong></summary>
                        <p style="font-size: 0.9em; margin-top: 10px;">
                            Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, 
                            Finnish, French, German, Greek, Hungarian, Italian, Latvian, 
                            Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, 
                            Slovenian, Spanish, Swedish, Russian, Ukrainian
                        </p>
                    </details>
                </div>
                """)
        
        # Event handlers
        asr_ast_mode.change(
            fn=update_target_language_visibility,
            inputs=[asr_ast_mode],
            outputs=[target_language]
        )
        
        enable_timestamps.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[enable_timestamps],
            outputs=[timestamps_output]
        )
        
        process_btn.click(
            fn=transcribe_audio,
            inputs=[
                audio_input,
                source_language,
                target_language,
                inference_mode,
                enable_timestamps,
                enable_pnc,
                asr_ast_mode
            ],
            outputs=[
                transcription_output,
                timestamps_output,
                status_output
            ]
        )
        
        # Load model on startup
        interface.load(
            fn=load_model,
            outputs=[status_output]
        )
    
    return interface

if __name__ == "__main__":
    print("Starting NVIDIA Canary-1B v2 Gradio Web UI...")
    print("Enhanced with 25 European language support and 10x performance improvement!")
    
    # Create and launch the interface
    app = create_interface()
    
    # Launch with public sharing disabled by default
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )

