"""
Test script to verify connection to Ollama server.

This script tests:
1. Basic HTTP connectivity to the Ollama server
2. Ollama API health check
3. List available models
4. Test model inference with a simple prompt
"""

import sys
import os
import json
import time
import requests
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import yaml
from llm_runner import get_llm, BASE_URL


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / 'src' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_http_connectivity(base_url):
    """Test basic HTTP connectivity to Ollama server."""
    print(f"\n{'='*60}")
    print(f"Test 1: HTTP Connectivity to {base_url}")
    print(f"{'='*60}")
    
    try:
        response = requests.get(f"{base_url}", timeout=5)
        if response.status_code == 200 or response.status_code == 404:
            print(f"✓ HTTP connectivity successful (Status: {response.status_code})")
            return True
    except requests.exceptions.ConnectionError as e:
        print(f"✗ Connection failed: {e}")
        print(f"  Make sure Ollama is running: `ollama serve`")
        return False
    except requests.exceptions.Timeout:
        print(f"✗ Connection timeout - Ollama server may not be responding")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_api_health(base_url):
    """Test Ollama API health endpoint."""
    print(f"\n{'='*60}")
    print(f"Test 2: API Health Check")
    print(f"{'='*60}")
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            print(f"✓ API health check passed")
            return True
        else:
            print(f"✗ API health check failed (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_list_models(base_url):
    """List available models on the Ollama server."""
    print(f"\n{'='*60}")
    print(f"Test 3: List Available Models")
    print(f"{'='*60}")
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            
            if models:
                print(f"✓ Found {len(models)} model(s):")
                for model in models:
                    name = model.get('name', 'Unknown')
                    size = model.get('size', 0)
                    size_gb = size / (1024**3) if size else 0
                    print(f"  - {name} ({size_gb:.2f} GB)")
                return True, models
            else:
                print(f"✗ No models found on server")
                print(f"  Pull a model with: `ollama pull llama3`")
                return False, []
        else:
            print(f"✗ Failed to fetch models (Status: {response.status_code})")
            return False, []
    except Exception as e:
        print(f"✗ Error listing models: {e}")
        return False, []


def test_model_inference(base_url, model_name):
    """Test inference with a specific model."""
    print(f"\n{'='*60}")
    print(f"Test 4: Model Inference Test ({model_name})")
    print(f"{'='*60}")
    
    try:
        print(f"Attempting to load model '{model_name}'...")
        llm = get_llm(model_name=model_name, base_url=base_url, temperature=0.0)
        
        test_prompt = "Say 'Connection successful' and nothing else."
        print(f"Sending test prompt: '{test_prompt}'")
        print(f"Waiting for response...")
        
        start_time = time.time()
        response = llm.invoke(test_prompt)
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Inference successful ({elapsed_time:.2f}s)")
        print(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        print(f"  Check if the model '{model_name}' is available: `ollama pull {model_name}`")
        return False


def test_langchain_integration(base_url, model_name):
    """Test LangChain integration with Ollama."""
    print(f"\n{'='*60}")
    print(f"Test 5: LangChain Integration")
    print(f"{'='*60}")
    
    try:
        print(f"Testing LangChain OllamaLLM with {model_name}...")
        llm = get_llm(model_name=model_name, base_url=base_url, temperature=0.0)
        
        if llm:
            print(f"✓ LangChain OllamaLLM initialized successfully")
            print(f"  Model: {model_name}")
            print(f"  Base URL: {base_url}")
            return True
        else:
            print(f"✗ Failed to initialize LangChain OllamaLLM")
            return False
            
    except Exception as e:
        print(f"✗ LangChain integration failed: {e}")
        return False


def main():
    """Run all connection tests."""
    print("\n" + "="*60)
    print("OLLAMA SERVER CONNECTION TEST SUITE")
    print("="*60)
    
    # Load configuration
    try:
        config = load_config()
        ollama_config = config.get('ollama_config', {})
        base_url = ollama_config.get('base_url', BASE_URL)
    except Exception as e:
        print(f"Warning: Could not load config.yaml, using default base_url: {BASE_URL}")
        base_url = BASE_URL
    
    print(f"\nTarget Ollama Server: {base_url}")
    
    # Track results
    results = {}
    
    # Test 1: HTTP Connectivity
    results['http_connectivity'] = test_http_connectivity(base_url)
    
    if not results['http_connectivity']:
        print("\n" + "="*60)
        print("TESTS ABORTED - Cannot connect to Ollama server")
        print("="*60)
        print("\nTo fix:")
        print("1. Make sure Ollama is installed: https://ollama.ai")
        print("2. Start Ollama server: `ollama serve`")
        print("3. Run this test again")
        return False
    
    # Test 2: API Health
    results['api_health'] = test_api_health(base_url)
    
    # Test 3: List Models
    list_ok, models = test_list_models(base_url)
    results['list_models'] = list_ok
    
    if not models:
        print("\n" + "="*60)
        print("NO MODELS AVAILABLE")
        print("="*60)
        print("\nTo download a model:")
        print("  ollama pull llama3     # Recommended for testing")
        print("  ollama pull mistral    # Alternative option")
        return False
    
    # Test 4: Model Inference (use first available model)
    model_to_test = models[0]['name']
    results['model_inference'] = test_model_inference(base_url, model_to_test)
    
    # Test 5: LangChain Integration
    results['langchain_integration'] = test_langchain_integration(base_url, model_to_test)
    
    # Print Summary
    print(f"\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {test_name.replace('_', ' ').title()}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Your Ollama connection is working correctly.")
        return True
    else:
        print(f"\n✗ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
