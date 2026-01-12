from parser import ToolSwordParser
from pathlib import Path
import yaml
import os

def load_cases(file_path):
    # Ensure path is absolute or relative to CWD
    if not os.path.exists(file_path):
        # Try relative to workspace root
        file_path = os.path.join(os.getcwd(), file_path)
    
    # Initialize parser with the directory of the file
    directory = os.path.dirname(file_path)
    parser = ToolSwordParser(directory)
    
    return list(parser.iter_file_cases(file_path))

def load_configuration(path="src/config.yaml"):

    config_path = Path(path)

    if config_path.exists():
        config = yaml.safe_load(open(config_path, "r"))
        JUDGE_MODELS = config.get("judges", ["mistral"])
        MODELS_TO_TEST = config.get("models", [{"name": "llama3"}])

    else:
        print("Warning: src/config.yaml not found. Using default judge 'mistral' and model 'llama3'.")
        JUDGE_MODELS = ["mistral"]
        MODELS_TO_TEST = [{"name": "llama3"}]
    
    CASES_DIR = Path("toolsword_cases")
    LIMIT = 2 # Integer or None; Set to None to run on all cases (WARNING: This may take a long time)

    

    return JUDGE_MODELS,MODELS_TO_TEST,CASES_DIR,LIMIT

