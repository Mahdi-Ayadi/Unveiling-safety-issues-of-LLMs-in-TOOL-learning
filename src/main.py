import os
import json
import yaml
from pathlib import Path
from tqdm import tqdm
from parser import ToolSwordParser
from llm_runner import run_llm
from evaluator import TwoLayerEvaluator, build_prompt_from_case, proportion_ci_wilson

def load_cases(file_path):
    # Ensure path is absolute or relative to CWD
    if not os.path.exists(file_path):
        # Try relative to workspace root
        file_path = os.path.join(os.getcwd(), file_path)
    
    # Initialize parser with the directory of the file
    directory = os.path.dirname(file_path)
    parser = ToolSwordParser(directory)
    
    return list(parser.iter_file_cases(file_path))

def process_file(file_path, model_name, evaluator, limit=None):
    print(f"\nProcessing {file_path.name}...")
    try:
        cases = load_cases(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    if limit:
        print(f"Limiting to first {limit} cases.")
        cases = cases[:limit]
    
    results = []
    unsafe_count = 0
    
    for i, case in enumerate(tqdm(cases, desc=f"Evaluating {file_path.name}")):
        prompt = build_prompt_from_case(case)
        response = run_llm(prompt, model_name=model_name)
        
        # Evaluate response
        eval_result = evaluator.evaluate_response(case, response, expected_tool=None)
        
        is_safe = eval_result["is_safe"]
        if not is_safe:
            unsafe_count += 1
            
        results.append({
            "case_id": i,
            "query": case.query,
            "response": response,
            "eval_result": eval_result
        })

    total = len(cases)
    asr = unsafe_count / total if total > 0 else 0
    ci_low, ci_high = proportion_ci_wilson(unsafe_count, total)
    
    summary = {
        "file": str(file_path.name),
        "total_cases": total,
        "unsafe_count": unsafe_count,
        "asr": asr,
        "asr_percentage": f"{asr:.2%}",
        "ci_95_low": f"{ci_low:.2%}",
        "ci_95_high": f"{ci_high:.2%}"
    }
    
    print(f"Results for {file_path.name}: ASR={asr:.2%} ({unsafe_count}/{total})")
    
    output_data = {
        "summary": summary,
        "results": results
    }
    
    # Create output directory if it doesn't exist
    output_dir = Path(f"results/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"results_{file_path.stem}.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved results to {output_file}")

def main():
    # Load Configuration
    config_path = Path("src/config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        JUDGE_MODELS = config.get("judges", ["mistral"])
        MODELS_TO_TEST = config.get("models", [{"name": "llama3"}])
    else:
        print("Warning: src/config.yaml not found. Using default judge 'mistral' and model 'llama3'.")
        JUDGE_MODELS = ["mistral"]
        MODELS_TO_TEST = [{"name": "llama3"}]

    CASES_DIR = Path("toolsword_cases")
    
    # Set to None to run on all cases (WARNING: This may take a long time)
    LIMIT = None 
    
    print(f"Initializing Evaluator with judges: {JUDGE_MODELS}...")
    evaluator = TwoLayerEvaluator(judge_models=JUDGE_MODELS)
    
    # Get all JSON files except data_confict.json
    files = sorted([f for f in CASES_DIR.glob("*.json") if f.name != "data_confict.json"])
    
    if not files:
        print(f"No JSON files found in {CASES_DIR}")
        return
        
    print(f"Found {len(files)} files to process: {[f.name for f in files]}")
    
    for model_config in MODELS_TO_TEST:
        model_name = model_config["name"]
        print(f"\n{'='*40}")
        print(f"Testing Model: {model_name}")
        print(f"{'='*40}")
        
        for file_path in files:
            process_file(file_path, model_name, evaluator, limit=LIMIT)


if __name__ == "__main__":
    main()
