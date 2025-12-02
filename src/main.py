import yaml
from src.llm_runner import run_llm

def load_config(path="config.yaml"):
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
        
        # Example: Run a test prompt
        test_prompt = "Tell me a safe way to check my network status." 
        
        # Pass the full configuration dict to the runner
        response = run_llm(test_prompt, model_config) 
        
        print(f"Response: {response[:200]}...") # Print first 200 chars

# Run the sweep
if __name__ == "__main__":
    app_config = load_config()
    run_experiment_sweep(app_config)