import random
from prompt_generator import *
from llm_runner import run_llm
from pathlib import Path
from parser import ToolSwordParser
import time

def generate_random_prompt(model_config,n=10):
    base_dir = Path("toolsword_cases")
    parser = ToolSwordParser(base_dir)
    cases = list(parser.iter_all_cases())
    for i in range(n):
        t0 = time.perf_counter()
        case = random.choice(cases)
        prompt = build_attack_prompt(case)
        print(f"--- Prompt from case {case.id} ---\n{prompt[:5]}\n")
        response = run_llm(prompt, model_config['name'], model_config['base_url'], model_config.get('temperature', 0.0))
        dt = time.perf_counter() - t0
        print(f"Run {i+1}: {dt:.2f}s")
        print(f"Response: {response[:5]}...\n") # Print first 200 chars