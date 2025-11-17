# src/evaluation.py (ou dans ton fichier actuel)

from datetime import datetime

class EvaluationResult:
    def __init__(
        self,
        scenario_id: str,
        expected_behavior: str,
        verdict: str,
        reason: str,
        response_excerpt: str,
        execution_time: float,
        backend_used: str,
        safety_status: str,
    ):
        self.scenario_id = scenario_id
        self.expected_behavior = expected_behavior
        self.verdict = verdict                # success / failure / blocked / uncertain
        self.reason = reason                  # pourquoi success / failure
        self.response_excerpt = response_excerpt
        self.execution_time = execution_time
        self.backend_used = backend_used
        self.safety_status = safety_status
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            "scenario_id": self.scenario_id,
            "expected_behavior": self.expected_behavior,
            "verdict": self.verdict,
            "reason": self.reason,
            "response_excerpt": self.response_excerpt,
            "execution_time": self.execution_time,
            "backend_used": self.backend_used,
            "safety_status": self.safety_status,
            "timestamp": self.timestamp,
        }

    def __repr__(self):
        return str(self.to_dict())
