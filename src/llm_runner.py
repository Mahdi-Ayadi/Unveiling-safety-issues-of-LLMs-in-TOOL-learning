# src/llm_runner.py

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

model = ChatOllama(model="gemma3:1b")

# content = "Il est quelle heure à Paris ?"
# response = model.invoke([HumanMessage(content=content)])
# print(response.content)

SAFETY_TAGS = {
    "safe" : "scénario validé, sans risque",
    "internal-only"	: "ne doit tourner que sur machines locales/sandbox",
    "requires-approval"	: "besoin d’accord humain avant exécution”",
    "no-exec"	: "interdit d’exécution (archivé, expérimental)",
    "mock-only"	: "doit être exécuté avec outils mockés uniquement",
    "red-team"	: "réservé aux campagnes de test offensif encadrées"
}

EXTERN_BACKENDS = [
    "gemma3:1b",
    "gemma3:3b",
    "gemma3:7b",
    "gemma3:13b",
    "gemma3:70b"
]

MOCK_TOOLS = [
    "mock_search_tool",
    "mock_calculator_tool"
]

input = {
    "id": "attack_001",
    "input_template": "prompt à envoyer",
    "expected_behavior": "réponse attendue",
    "tool_list": [],
    "safety_tags": [], # SAFETY_TAGS
    "allowed_backends": ["gemma3:1b"],
    "metadata": {}
}

def validation(input):

    safety_tags = input["safety_tags"]

    if "no-exec" in safety_tags: #runner doit refuser
        return False
    
    if "requires-approval" in safety_tags: #demander approbation humaine
        approval = input("This scenario requires human approval. Do you approve? (yes/no): ")
        if approval.lower() != "yes":
            return False
        
    if "internal-only" in safety_tags: #runner doit refuser si backend externe
        allowed_backends = input["allowed_backends"]
        for backend in allowed_backends:
            if backend in EXTERN_BACKENDS:
                return False
            
    if "mock-only" in safety_tags: #runner doit refuser si pas en mode mock
        tool_list = input["tool_list"]
        is_mock_mode = all(tool in MOCK_TOOLS for tool in tool_list)
        if not is_mock_mode:
            return False
        
    if "red-team" in safety_tags: #ne pas exécuter par défaut (False par défaut)
        return False    
    
    return True

