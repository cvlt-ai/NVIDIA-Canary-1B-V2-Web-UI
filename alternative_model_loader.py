#!/usr/bin/env python3
"""
Alternative Model Loader for NVIDIA Canary-1B Flash
This script provides a more robust model loading approach.
"""

import os
import sys
import subprocess
from pathlib import Path

def download_model_files():
    """Download model files using git-lfs or huggingface-hub."""
    
    print("🎤 NVIDIA Canary-1B Flash Alternative Setup")
    print("=" * 50)
    
    try:
        # Try using huggingface_hub for more reliable downloads
        print("📦 Installing huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        
        from huggingface_hub import snapshot_download
        
        # Create models directory
        script_dir = Path(__file__).parent
        models_dir = script_dir / "models"
        cache_dir = models_dir / "cache"
        
        models_dir.mkdir(exist_ok=True)
        cache_dir.mkdir(exist_ok=True)
        
        print(f"📁 Model directory: {models_dir}")
        print(f"💾 Cache directory: {cache_dir}")
        
        print("\n🔄 Downloading NVIDIA Canary-1B Flash model files...")
        print("⏳ This may take several minutes...")
        
        # Download model files to local cache
        local_model_path = snapshot_download(
            repo_id="nvidia/canary-1b-flash",
            cache_dir=str(cache_dir),
            local_dir=str(models_dir / "canary-1b-flash"),
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Model files downloaded to: {local_model_path}")
        
        # Create a simple test to verify files
        model_files = list(Path(local_model_path).glob("*"))
        print(f"📋 Downloaded {len(model_files)} files:")
        for f in model_files[:5]:  # Show first 5 files
            print(f"   • {f.name}")
        if len(model_files) > 5:
            print(f"   • ... and {len(model_files) - 5} more files")
        
        print("\n🎉 Model download complete!")
        print("💡 Note: The application will load the model from local files")
        
        return True, str(local_model_path)
        
    except Exception as e:
        print(f"❌ Error during download: {e}")
        print("\n💡 Fallback options:")
        print("   1. The application will try to download automatically on first use")
        print("   2. Ensure you have a stable internet connection")
        print("   3. Check that you have 5GB+ free disk space")
        return False, None

def create_model_info():
    """Create a model info file for the application."""
    
    script_dir = Path(__file__).parent
    info_file = script_dir / "model_info.txt"
    
    info_content = """NVIDIA Canary-1B Flash Model Information

Model: nvidia/canary-1b-flash
Parameters: 883M
Languages: English, German, French, Spanish
Capabilities: ASR (Automatic Speech Recognition), AST (Automatic Speech Translation), Timestamps
License: CC-BY-4.0

Usage:
- ASR Mode: Transcribe speech to text in the same language
- AST Mode: Translate speech from source language to target language
- Timestamps: Generate word-level and segment-level timestamps
- Long-form: For audio longer than 30 seconds
- Short-form: For audio shorter than 30 seconds

Supported Audio Formats:
- WAV, MP3, FLAC, M4A, OGG
- Automatically converted to 16kHz mono for processing
"""
    
    with open(info_file, 'w') as f:
        f.write(info_content)
    
    print(f"📄 Created model info file: {info_file}")

if __name__ == "__main__":
    success, model_path = download_model_files()
    create_model_info()
    
    if success:
        print(f"\n✅ Setup completed successfully!")
        print(f"📍 Model location: {model_path}")
    else:
        print(f"\n⚠️  Setup completed with warnings.")
        print("The application will attempt automatic download on first use.")
    
    sys.exit(0 if success else 1)

