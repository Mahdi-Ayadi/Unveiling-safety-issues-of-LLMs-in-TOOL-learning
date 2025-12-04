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
    
    output_file = f"results_{file_path.stem}.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved results to {output_file}")

def main():
    # Configuration
    CASES_DIR = Path("toolsword_cases")
    MODEL_NAME = "llama3"
    # Using 'mistral' as the judge to avoid bias from using the same model (llama3)
    # Ensure you have run `ollama pull mistral` in your terminal.
    JUDGE_MODEL = "mistral"
    
    # Set to None to run on all cases (WARNING: This may take a long time)
    LIMIT = None 
    
    print("Initializing Evaluator...")
    evaluator = TwoLayerEvaluator(judge_model=JUDGE_MODEL)
    
    # Get all JSON files except data_confict.json
    files = sorted([f for f in CASES_DIR.glob("*.json") if f.name != "data_confict.json"])
    
    if not files:
        print(f"No JSON files found in {CASES_DIR}")
        return
        
    print(f"Found {len(files)} files to process: {[f.name for f in files]}")
    
    for file_path in files:
        process_file(file_path, MODEL_NAME, evaluator, limit=LIMIT)


if __name__ == "__main__":
    main()
