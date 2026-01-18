"""
Comprehensive test script to verify connection to all models in config.yaml
Tests both models to test AND judge models through LangChain OllamaLLM
"""

import sys
import os
import yaml
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from langchain_ollama import OllamaLLM


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / 'src' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_model_connection(model_name, base_url):
    """Test connection and basic inference for a single model"""
    print(f"\n  Testing model: {model_name}...", end=" ", flush=True)
    
    try:
        llm = OllamaLLM(
            model=model_name,
            base_url=base_url,
            timeout=120,
            temperature=0.0
        )
        
        # Try a simple inference
        response = llm.invoke("Say 'OK' and nothing else.")
        
        if "[TIMEOUT]" in str(response) or "[ERROR]" in str(response):
            print(f"❌ TIMEOUT/ERROR")
            return False
        else:
            print(f"✓ OK")
            return True
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)[:50]}")
        return False


def main():
    """Test all models from config"""
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL CONNECTION TEST")
    print("="*70)
    
    # Load configuration
    try:
        config = load_config()
        ollama_config = config.get('ollama_config', {})
        base_url = ollama_config.get('base_url', 'http://localhost:11434')
    except Exception as e:
        print(f"Error loading config: {e}")
        return False
    
    print(f"\nOllama Server: {base_url}")
    
    # Collect all models to test
    models_to_test = []
    judges = []
    
    # Get models to test
    models_config = config.get('models', [])
    for model_cfg in models_config:
        model_name = model_cfg.get('name')
        if model_name:
            models_to_test.append(model_name)
    
    # Get judge models
    judges = config.get('judges', [])
    
    print(f"\n📋 Models to test: {models_to_test}")
    print(f"📋 Judge models: {judges}")
    
    # Test all models
    all_models = list(set(models_to_test + judges))  # Unique models
    
    print(f"\n{'='*70}")
    print(f"Testing {len(all_models)} unique model(s) through LangChain OllamaLLM:")
    print(f"{'='*70}")
    
    results = {}
    passed = 0
    failed = 0
    
    for model_name in sorted(all_models):
        success = test_model_connection(model_name, base_url)
        results[model_name] = success
        if success:
            passed += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print(f"{'='*70}")
    
    for model_name, success in sorted(results.items()):
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} - {model_name}")
    
    if failed == 0:
        print(f"\n✓ All models are working correctly!")
        return True
    else:
        print(f"\n✗ {failed} model(s) failed. Check if they are installed:")
        for model_name, success in results.items():
            if not success:
                print(f"    To install: ollama pull {model_name}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
