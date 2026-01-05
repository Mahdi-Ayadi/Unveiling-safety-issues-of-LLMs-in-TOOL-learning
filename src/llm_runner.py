# llm/runner/llm_runner.py
from typing import Dict, Any, Callable

# OpenAI"
"""import openai"""

# HuggingFace
from transformers import pipeline


class LLMRunner:
    def __init__(self):
        self.llms: Dict[str, Dict[str, Any]] = {}

    def register_llm(self, name: str, llm_callable: Callable, **params):
        """
        Enregistre un LLM avec ses paramètres
        :param name: nom unique du LLM
        :param llm_callable: fonction callable qui prend un prompt et des params et renvoie un texte
        :param params: paramètres spécifiques du LLM
        """
        self.llms[name] = {"llm": llm_callable, "params": params}

    def run(self, name: str, prompt: str) -> str:
        """
        Exécute le LLM sur un prompt
        """
        if name not in self.llms:
            raise ValueError(f"LLM '{name}' n'est pas enregistré.")

        llm_info = self.llms[name]
        llm_callable = llm_info["llm"]
        params = llm_info["params"]

        return llm_callable(prompt, **params)

    def run_all(self, prompt: str) -> Dict[str, str]:
        """
        Exécute tous les LLMs sur le même prompt
        """
        return {name: self.run(name, prompt) for name in self.llms}


# -------------------
# Exemples de LLM callables
# -------------------

"""def openai_llm(prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 150):
    
    Appel d'un modèle OpenAI Chat
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response['choices'][0]['message']['content']"""


def hf_llm(prompt: str, model_name: str = "gpt2", max_length: int = 50):
    """
    Appel d'un modèle HuggingFace (text-generation)
    """
    gen = pipeline("text-generation", model=model_name)
    result = gen(prompt, max_length=max_length, do_sample=True)
    return result[0]['generated_text']


runner = LLMRunner()

# Enregistrement des LLMs
"""runner.register_llm("OpenAI-GPT3.5", openai_llm, model="gpt-3.5-turbo", temperature=0.7, max_tokens=100)"""
runner.register_llm("HuggingFace-GPT2", hf_llm, model_name="gpt2", max_length=100)

# Prompt à tester
prompt = "explain special relativity in simple terms"

# Test d'un LLM
"""print("OpenAI:", runner.run("OpenAI-GPT3.5", prompt))"""""
print("HuggingFace:", runner.run("HuggingFace-GPT2", prompt))

"""# Test de tous les LLMs en même temps
all_results = runner.run_all(prompt)
print(all_results)"""

