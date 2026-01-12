
import json
import threading
from pathlib import Path
from tqdm import tqdm
from loaders import *
from llm_runner import *
from parser import *
from calculate_global_score import *

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

def main():

    # Load Configuration
    JUDGE_MODELS,MODELS_TO_TEST,CASES_DIR,LIMIT = load_configuration()

    # judge_llms = [JudgeLLM(name) for name in JUDGE_MODELS]
    # target_llms = [TargetLLM(model['name']) for model in MODELS_TO_TEST]

    # Test 
    judge_llms = [JudgeLLM("llama3")]
    target_llms = [TargetLLM("llama3")]
    
    all_files = sorted([f for f in CASES_DIR.glob("*.json") if f.name != "data_confict.json"])

    


    for target in target_llms:
        for file_path in all_files:
            cases = load_cases(file_path)
            if LIMIT:
                cases = cases[:LIMIT]
            unsafe_count = 0
            results_attack = []
            for i, case in enumerate(tqdm(cases)):
                r = evaluate_LLM_with_case(target,case,judge_llms)
                r["case_id"] = i
                if not r["eval_result"]["is_safe"]:
                    unsafe_count += 1
                results_attack.append(r)

            create_summary(len(cases),unsafe_count,file_path,results_attack,target.model_name)
            
if __name__ == "__main__":

    main()
