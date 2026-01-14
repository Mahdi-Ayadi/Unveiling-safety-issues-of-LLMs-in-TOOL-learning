"""
LLM Runner Module

This module handles the initialization and execution of a local LLM using LangChain.
It is designed to work with open-source models (e.g., via Ollama or HuggingFace).

Usage:
    from src.llm_runner import run_llm
    response = run_llm("Your prompt here")
"""

from langchain_ollama import OllamaLLM 
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

# Configuration for the local LLM
# Ensure you have Ollama installed and the model pulled (e.g., `ollama pull llama3`)
MODEL_NAME = "llama3"  # Default model
BASE_URL = "http://127.0.0.1:11434"

def get_llm(model_name: str = MODEL_NAME, base_url: str = BASE_URL, temperature: float = 0.0):
    """
    Initialize and return the LLM instance.
    Allows overriding model configuration at runtime.
    """
    llm = OllamaLLM(
        model=model_name,
        base_url=base_url,
        timeout=30,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        temperature=temperature,
    )
    return llm

def run_llm(prompt: str, model_name: str = MODEL_NAME, base_url: str = BASE_URL, temperature: float = 0.0) -> str:
    """
    Send a prompt to the LLM and return the response string.
    Allows specifying model configuration per call.
    """
    llm = get_llm(model_name=model_name, base_url=base_url, temperature=temperature)
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"Error running LLM: {e}"

if __name__ == "__main__":
    # Simple test to verify the runner works
    print(f"Testing LLM Runner with model: {MODEL_NAME}...")
    test_prompt = "Hello, are you ready for a security test?"
    print(f"Prompt: {test_prompt}")
    # Example of passing custom config
    result = run_llm(test_prompt, model_name="llama3", temperature=0.7)
    print(f"\nResponse:\n{result}")
