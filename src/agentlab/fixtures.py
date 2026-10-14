"""Load the synthetic fixture corpus, including evaluator-only expected labels."""
import json
from importlib.resources import files
from pathlib import Path
from .tasks import build_request


def load_cases(path: str | Path | None = None) -> list[dict]:
    source = Path(path) if path is not None else files("agentlab").joinpath("data/cases.json")
    cases = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("fixture corpus must be a nonempty list")
    seen = set()
    for case in cases:
        request = build_request(case)
        if request.case_id in seen:
            raise ValueError(f"duplicate case ID: {request.case_id}")
        seen.add(request.case_id)
        if not isinstance(case.get("expected"), dict):
            raise ValueError("case requires evaluator-only expected labels")
    return cases
