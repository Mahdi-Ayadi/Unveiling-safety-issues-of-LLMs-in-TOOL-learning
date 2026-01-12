import os
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def create_summary(total,unsafe_count,file_path,results,model_name):
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

def process_case(case_data):
    """Process a single case (for parallel execution)"""
    case_index, case, model_name, evaluator = case_data
    
    prompt = build_prompt_from_case(case)
    response = run_llm(prompt, model_name=model_name)
    eval_result = evaluator.evaluate_response(case, response, case_index=case_index)
    
    is_safe = eval_result["is_safe"]
    
    return {
        "case_id": case_index,
        "query": case.query,
        "response": response,
        "eval_result": eval_result,
        "is_safe": is_safe
    }

def process_file(file_path, model_name, evaluator, limit=None, max_workers=4):
    """Process all cases in a file with parallel execution"""
    print(f"\nProcessing {file_path.name}...")
    
    try:
        cases = load_cases(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    if limit:
        print(f"Limiting to first {limit} cases.")
        cases = cases[:limit]
    
    # Prepare case data for parallel processing
    case_data_list = [
        (i, case, model_name, evaluator) 
        for i, case in enumerate(cases)
    ]
    
    results = []
    unsafe_count = 0
    
    # Process cases in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_case, case_data): case_data[0] 
            for case_data in case_data_list
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(cases), desc=f"Evaluating {file_path.name}") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                if not result["is_safe"]:
                    unsafe_count += 1
                
                pbar.update(1)
    
    # Sort results by case_id to maintain order
    results.sort(key=lambda x: x["case_id"])
    
    # Remove the is_safe key from results (already in eval_result)
    for r in results:
        del r["is_safe"]
    
    create_summary(len(cases), unsafe_count, file_path, results, model_name)

def load_configuration(path="src/config.yaml"):

    config_path = Path(path)

    if config_path.exists():
        config = yaml.safe_load(open(config_path, "r"))
        JUDGE_MODELS = config.get("judges", ["mistral"])
        # JUDGE_MODELS = ["llama3"]
        MODELS_TO_TEST = config.get("models", [{"name": "llama3"}])
        MAX_WORKERS = config.get("max_workers", 4)  # Number of parallel case threads per file

    else:
        print("Warning: src/config.yaml not found. Using default judge 'mistral' and model 'llama3'.")
        JUDGE_MODELS = ["mistral"]
        MODELS_TO_TEST = [{"name": "llama3"}]
        MAX_WORKERS = 4
    
    CASES_DIR = Path("toolsword_cases")
    LIMIT = 1 # Integer or None; Set to None to run on all cases (WARNING: This may take a long time)

    return JUDGE_MODELS, MODELS_TO_TEST, CASES_DIR, LIMIT, MAX_WORKERS

def attack_launcher(model_config, files, evaluator: TwoLayerEvaluator, LIMIT, max_workers=4):

    model_name = model_config["name"]
    print(f"\n{'='*40}")
    print(f"Testing Model: {model_name}")
    print(f"{'='*40}")

    # Process files in parallel using ThreadPoolExecutor
    # Each file will internally parallelize its cases
    with ThreadPoolExecutor(max_workers=min(len(files), 2)) as file_executor:
        futures = {
            file_executor.submit(process_file, file_path, model_name, evaluator, LIMIT, max_workers): file_path
            for file_path in files
        }
        
        # Wait for all file processing to complete
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")


def main():

    # Load Configuration
    JUDGE_MODELS, MODELS_TO_TEST, CASES_DIR, LIMIT, MAX_WORKERS = load_configuration()
    print(f"Initializing Evaluator with judges: {JUDGE_MODELS}")
    print(f"Using {MAX_WORKERS} parallel worker threads per file")

    evaluator = TwoLayerEvaluator(judge_models=JUDGE_MODELS)

    # Get all JSON files except data_confict.json
    files = sorted([f for f in CASES_DIR.glob("*.json") if f.name != "data_confict.json"])
    if not files: 
        raise Exception(f"No JSON files found in {CASES_DIR}")
        
    print(f"Found {len(files)} files to process: {[f.name for f in files]}")
    
    # Process models sequentially
    for model_config in MODELS_TO_TEST:
        attack_launcher(model_config, files, evaluator, LIMIT, MAX_WORKERS)


if __name__ == "__main__":
    main()
