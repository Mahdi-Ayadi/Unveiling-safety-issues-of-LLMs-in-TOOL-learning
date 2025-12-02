import yaml
from llm_runner import run_llm
from prompt_generator import *
from pathlib import Path
from evaluator import generate_random_prompt
import random

def load_config(path="src/config.yaml"):
    with open(path, 'r') as f:
        # Load the YAML content
        return yaml.safe_load(f)

def run_experiment_sweep(config):
    ollama_base_config = config['ollama_config']

    print(f"--- Starting {config['project']['name']} Sweep ---")
    
    # Iterate through every model defined in the config



    for model_spec in config['models']:
        model_name = model_spec['name']
        
        # Combine base settings with model-specific overrides
        model_config = ollama_base_config.copy()
        model_config.update(model_spec)
        
        print(f"\n[MODEL: {model_name}] - Running tests...")
        generate_random_prompt(model_config)

        break # on fait seulement avec ollama pour l'instant

# Run the sweep
if __name__ == "__main__":
    app_config = load_config()
    run_experiment_sweep(app_config)