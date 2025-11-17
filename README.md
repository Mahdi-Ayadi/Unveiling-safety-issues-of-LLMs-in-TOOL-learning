# Unveiling-safety-issues-of-LLMs-in-TOOL-learning
Research project reproducing and extending ToolSword attacks to evaluate LLM safety in tool learning. Includes parsing, prompting, and model evaluation pipelines for analyzing vulnerabilities and robustness.

## Virtual Environment Setup

### 1. Create the virtual environment
```bash
python -m venv .venv
```

### 2. Activate the environment

| OS | Command |
|----|----------|
| **Windows (PowerShell)** | `.venv\Scripts\Activate.ps1` |
| **Windows (CMD)** | `.venv\Scripts\activate.bat` |
| **macOS / Linux** | `source .venv/bin/activate` |

> *If PowerShell blocks the command, run:*  
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Update dependencies
```bash
pip freeze > requirements.txt
```