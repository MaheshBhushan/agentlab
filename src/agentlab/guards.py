"""Traceable deterministic checks and an optional, real NeMo rail engine."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from decimal import Decimal
from functools import lru_cache
import json
from pathlib import Path
import re

from .models import TaskOutput
from .tasks import TaskRequest, request_messages

# These intentionally bounded heuristics are a baseline, not a security guarantee.
_INJECTION = re.compile(r"ignore\s+(?:all\s+)?(?:(?:previous|prior|above)\s+)?instructions|"
                        r"system\s+prompt|systemanweisung|override|you\s+are\s+now|"
                        r"(?:send|reveal|exfiltrate).{0,35}(?:secret|api.key)|"
                        r"ignoriere.{0,25}(?:anweisungen|instruktionen)|überweise.{0,25}sofort", re.I)
_SENSITIVE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b|"
                        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|"
                        r"\b(?:sk|api)[-_][A-Za-z0-9_-]{16,}\b")


class GuardError(RuntimeError):
    """A selected guard engine could not perform its checks."""


@dataclass(frozen=True)
class GuardResult:
    passed: bool
    action: str
    reasons: list[str]
    engine: str

    def to_dict(self) -> dict:
        return asdict(self)


def _decision(reasons: list[str], engine: str, action: str = "review") -> GuardResult:
    return GuardResult(not reasons, action if reasons else "allow", reasons, engine)


@lru_cache(maxsize=4)
def _nemo_rails(model: str):
    from nemoguardrails import RailsConfig, LLMRails

    config = RailsConfig.from_path(str(Path(__file__).with_name("guard_config")))
    # Keep configuration in the package; the selected model is shared with adapters.
    config.models[0].model = model.removeprefix("openai:")
    return LLMRails(config)


def _nemo_check(request: TaskRequest, output: TaskOutput | None, model: str) -> GuardResult:
    try:
        from nemoguardrails.rails.llm.options import RailType

        rails = _nemo_rails(model)
        messages = request_messages(request)[1:]
        phase = "input" if output is None else "output"
        if output is not None:
            messages.append({"role": "assistant", "content": json.dumps(output.to_dict(), ensure_ascii=False)})
        result = rails.check(messages, rail_types=[RailType.INPUT if output is None else RailType.OUTPUT])
        status = getattr(result.status, "value", result.status)
        status = str(status).lower()
        if status == "passed":
            return _decision([], "nemo")
        if status == "blocked":
            return _decision([f"nemo:{phase}:{result.rail or 'configured rail'}:blocked"], "nemo", "block")
        if status == "modified":
            # A modified answer requires human review; never silently replace the output.
            return _decision([f"nemo:{phase}:modified"], "nemo")
        raise GuardError("nemo returned an unknown rail status")
    except ImportError as exc:
        raise GuardError("nemo dependency unavailable; install agentlab[nemo] on Python 3.10–3.13") from exc
    except GuardError:
        raise
    except Exception as exc:
        raise GuardError(f"nemo check failed ({type(exc).__name__})") from exc


def check_input(request: TaskRequest, engine: str = "builtin", model: str = "gpt-4o-mini") -> GuardResult:
    if engine == "none":
        return _decision([], "none")
    if engine == "nemo":
        return _nemo_check(request, None, model)
    if engine != "builtin":
        raise ValueError(f"unknown guard engine: {engine}")
    reasons = []
    for doc in request.documents:
        if _INJECTION.search(doc.text):
            reasons.append(f"builtin:input:{doc.document_id}:instruction_in_untrusted_document")
    if _INJECTION.search(request.instruction):
        reasons.append("builtin:input:instruction:prompt_injection_pattern")
    return _decision(reasons, "builtin", "block")


def check_output(request: TaskRequest, output: TaskOutput, engine: str = "builtin",
                 model: str = "gpt-4o-mini") -> GuardResult:
    if engine == "none":
        return _decision([], "none")
    if engine == "nemo":
        return _nemo_check(request, output, model)
    if engine != "builtin":
        raise ValueError(f"unknown guard engine: {engine}")
    reasons = []
    try:
        TaskOutput.from_dict(output.to_dict())
    except (TypeError, ValueError) as exc:
        return _decision([f"builtin:output:invalid_schema:{type(exc).__name__}"], "builtin", "block")
    documents = {doc.document_id: doc.text for doc in request.documents}
    if output.task != request.task:
        reasons.append("builtin:output:task_mismatch")
    evidence = list(output.evidence)
    for index, item in enumerate(output.action_items):
        evidence.extend(item.evidence)
        if not item.evidence:
            reasons.append(f"builtin:output:action_items[{index}]:missing_evidence")
        for name, value in (("owner", item.owner), ("due_date", item.due_date)):
            if value is None:
                reasons.append(f"builtin:output:action_items[{index}]:missing_{name}")
            elif name == "owner" and not any(value in ref.quote for ref in item.evidence):
                reasons.append(f"builtin:output:action_items[{index}]:unsupported_owner")
            elif name == "due_date":
                deadline = date.fromisoformat(value)  # Strict schema already validates ISO dates.
                forms = (value, deadline.strftime("%d.%m.%Y"),
                         f"{deadline.day}.{deadline.month}.{deadline.year}")
                if not any(form in ref.quote for form in forms for ref in item.evidence):
                    reasons.append(f"builtin:output:action_items[{index}]:unsupported_due_date")
    if output.status == "completed" and not evidence:
        reasons.append("builtin:output:missing_evidence")
    for index, ref in enumerate(evidence):
        if ref.document_id not in documents:
            reasons.append(f"builtin:output:evidence[{index}]:unknown_document:{ref.document_id}")
        elif not ref.quote or ref.quote not in documents[ref.document_id]:
            reasons.append(f"builtin:output:evidence[{index}]:quote_not_in_document:{ref.document_id}")
    if request.task == "invoice_triage":
        if output.invoice is None:
            reasons.append("builtin:output:missing_invoice")
        elif any(output.invoice[key] is None for key in ("invoice_number", "supplier", "amount_eur")):
            reasons.append("builtin:output:invoice:missing_fields")
        if output.invoice is not None:
            quotes = "\n".join(ref.quote for ref in output.evidence)
            for key in ("invoice_number", "supplier"):
                value = output.invoice[key]
                if value is not None and value not in quotes:
                    reasons.append(f"builtin:output:invoice:unsupported_{key}")
            amount = output.invoice["amount_eur"]
            if amount is not None:
                numbers = re.findall(r"(?<!\w)\d[\d.,]*(?!\w)", quotes)
                values = []
                for number in numbers:
                    normalized = number.replace(".", "").replace(",", ".") if "," in number else number
                    try:
                        values.append(Decimal(normalized))
                    except ArithmeticError:
                        continue
                if Decimal(amount) not in values:
                    reasons.append("builtin:output:invoice:unsupported_amount_eur")
    elif output.invoice is not None:
        reasons.append("builtin:output:inapplicable_invoice")
    if request.task != "meeting_followup" and output.action_items:
        reasons.append("builtin:output:inapplicable_action_items")
    serialized = json.dumps(output.to_dict(), ensure_ascii=False)
    if _SENSITIVE.search(serialized):
        reasons.append("builtin:output:sensitive_data")
    if _INJECTION.search(output.answer):
        reasons.append("builtin:output:instruction_pattern")
    return _decision(reasons, "builtin")
