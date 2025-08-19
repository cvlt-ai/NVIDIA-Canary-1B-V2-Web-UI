#!/usr/bin/env python3
"""
Model Setup Script for NVIDIA Canary-1B v2
This script downloads and caches the enhanced v2 model for the Gradio application.
"""

import os
import sys
from pathlib import Path

def setup_model():
    """Download and setup the NVIDIA Canary-1B v2 model."""
    
    print("🎤 NVIDIA Canary-1B v2 Model Setup")
    print("=" * 50)
    
    try:
        # Import required modules
        print("📦 Checking dependencies...")
        from nemo.collections.asr.models import EncDecMultiTaskModel
        print("✅ NeMo framework found")
        
        # Create models directory
        script_dir = Path(__file__).parent
        models_dir = script_dir / "models"
        cache_dir = models_dir / "cache"
        
        models_dir.mkdir(exist_ok=True)
        cache_dir.mkdir(exist_ok=True)
        
        print(f"📁 Model directory: {models_dir}")
        print(f"💾 Cache directory: {cache_dir}")
        
        # Set environment variables for caching
        os.environ['HF_HOME'] = str(cache_dir)
        if 'TRANSFORMERS_CACHE' in os.environ:
            del os.environ['TRANSFORMERS_CACHE']  # Remove deprecated variable
        os.environ['HF_DATASETS_CACHE'] = str(cache_dir)
        
        print("\n🔄 Downloading NVIDIA Canary-1B v2 model...")
        print("⏳ This may take several minutes depending on your internet connection...")
        print("📊 Model size: ~6.32GB")
        print("🆕 Enhanced with 25 European language support!")
        
        # Download the model (NeMo handles caching automatically)
        model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b-v2')
        
        print("✅ Model downloaded successfully!")
        print(f"📍 Model cached automatically by NeMo/HuggingFace")
        
        # Test model configuration
        print("\n🔧 Configuring model...")
        decode_cfg = model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        model.change_decoding_strategy(decode_cfg)
        
        print("✅ Model configuration complete!")
        
        # Display model information
        print("\n📋 Model Information:")
        print(f"   • Model: NVIDIA Canary-1B v2")
        print(f"   • Parameters: 978M (vs 883M in v1)")
        print(f"   • Languages: 25 European Languages")
        print(f"   • Performance: 10x faster than comparable models")
        print(f"   • Quality: Comparable to 3x larger models")
        print(f"   • Capabilities: ASR, AST, Word/Segment Timestamps")
        print(f"   • Architecture: FastConformer + Transformer")
        print(f"   • Cache Location: {cache_dir}")
        
        print("\n🌍 Supported Languages:")
        languages = [
            "Bulgarian", "Croatian", "Czech", "Danish", "Dutch", "English",
            "Estonian", "Finnish", "French", "German", "Greek", "Hungarian", 
            "Italian", "Latvian", "Lithuanian", "Maltese", "Polish", 
            "Portuguese", "Romanian", "Slovak", "Slovenian", "Spanish", 
            "Swedish", "Russian", "Ukrainian"
        ]
        for i, lang in enumerate(languages):
            if i % 5 == 0:
                print("   ", end="")
            print(f"{lang:<12}", end="")
            if (i + 1) % 5 == 0:
                print()
        if len(languages) % 5 != 0:
            print()
        
        print("\n🎉 Setup complete! You can now run the enhanced Gradio application.")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n💡 Solution:")
        print("   Run: pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check your internet connection")
        print("   2. Ensure you have 5GB+ free disk space")
        print("   3. Try running with administrator/sudo privileges")
        print("   4. Check firewall settings")
        print("   5. The model will be cached automatically by HuggingFace")
        return False

if __name__ == "__main__":
    success = setup_model()
    sys.exit(0 if success else 1)

