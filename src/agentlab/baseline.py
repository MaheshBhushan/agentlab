"""Deterministic offline baseline, never a simulated framework/model result."""
import re
from decimal import Decimal, InvalidOperation
from .models import ActionItem, Evidence, TaskOutput
from .tasks import TaskRequest

# ponytail: line-oriented heuristics cover fixture format; model adapters handle free prose.
INJECTION = re.compile(r"ignore (?:all |previous |prior )?instructions|ignoriere .*anweisungen|system prompt|systemanweisung|override|überweise.*sofort", re.I)


def _lines(request: TaskRequest) -> list[tuple[str, str]]:
    return [(doc.document_id, line.strip()) for doc in request.documents for line in doc.text.splitlines() if line.strip()]


def run_baseline(request: TaskRequest) -> TaskOutput:
    lines = _lines(request)
    suspicious = [Evidence(doc_id, line) for doc_id, line in lines if INJECTION.search(line)]
    if suspicious:
        return TaskOutput(request.task, "needs_review", "Verdächtige Dokumentanweisung erfordert Prüfung.", suspicious,
                          review_reasons=["Untrusted document contains an instruction attempt"])
    if request.task == "invoice_triage":
        return _invoice(request, lines)
    if request.task == "document_question":
        return _question(request, lines)
    if request.task == "meeting_followup":
        return _meeting(request, lines)
    raise ValueError(f"unsupported task: {request.task}")


def _invoice(request: TaskRequest, lines: list[tuple[str, str]]) -> TaskOutput:
    fields = {"invoice_number": "Rechnungsnummer", "supplier": "Lieferant", "amount_eur": "Bruttobetrag EUR"}
    invoice = {}
    evidence = []
    reasons = []
    for key, label in fields.items():
        matches = [(doc, line, line.split(":", 1)[1].strip()) for doc, line in lines if line.startswith(label + ":")]
        values = {value for _, _, value in matches}
        invoice[key] = next(iter(values)) if len(values) == 1 else None
        evidence.extend(Evidence(doc, line) for doc, line, _ in matches)
        if len(values) != 1:
            reasons.append(f"Missing or conflicting {label}")
    if invoice["amount_eur"] is not None:
        try:
            amount = Decimal(invoice["amount_eur"].replace(".", "").replace(",", ".")) if "," in invoice["amount_eur"] else Decimal(invoice["amount_eur"])
            if not amount.is_finite() or amount <= 0:
                raise InvalidOperation
            invoice["amount_eur"] = f"{amount:.2f}"
        except InvalidOperation:
            invoice["amount_eur"] = None
            reasons.append("Invalid invoice amount")
    for doc, line in lines:
        if re.search(r"nicht bestätigt|abweichung|doppelt|bankverbindung geändert|lieferschein fehlt", line, re.I):
            reasons.append(line)
            evidence.append(Evidence(doc, line))
    invoice["decision"] = "hold" if reasons else "ready_for_review"
    return TaskOutput(request.task, "needs_review" if reasons else "completed",
                      "Rechnung zurückstellen." if reasons else "Rechnung zur menschlichen Freigabe vorlegen.",
                      evidence, invoice=invoice, review_reasons=reasons)


def _question(request: TaskRequest, lines: list[tuple[str, str]]) -> TaskOutput:
    labels = {"brandschutz": "Brandschutzklasse", "beton": "Betonklasse", "zufahrt": "Baustellenzufahrt", "abnahme": "Abnahmetermin"}
    label = next((value for key, value in labels.items() if key in request.instruction.lower()), None)
    matches = [(doc, line) for doc, line in lines if label and line.startswith(label + ":")]
    evidence = [Evidence(doc, line) for doc, line in matches]
    answers = {line.split(":", 1)[1].strip() for _, line in matches}
    if len(answers) != 1:
        reason = "Conflicting source answers" if answers else "No supported answer found"
        return TaskOutput(request.task, "needs_review", "Die Quellen liefern keine eindeutige Antwort.", evidence, review_reasons=[reason])
    return TaskOutput(request.task, "completed", next(iter(answers)), evidence)


def _meeting(request: TaskRequest, lines: list[tuple[str, str]]) -> TaskOutput:
    items = []
    reasons = []
    for doc, line in lines:
        if not line.startswith("Aufgabe:"):
            continue
        parts = line.split(" | ")
        description = parts[0].split(":", 1)[1].strip()
        attrs = dict(part.split(": ", 1) for part in parts[1:] if ": " in part)
        owner = attrs.get("Verantwortlich")
        due_date = attrs.get("Frist")
        owner = None if owner in (None, "offen", "unklar") else owner
        if due_date:
            from datetime import date
            try:
                date.fromisoformat(due_date)
            except ValueError:
                due_date = None
        if not owner or not due_date:
            reasons.append(f"Unclear owner or due date: {description}")
        items.append(ActionItem(description, owner, due_date, [Evidence(doc, line)]))
    if not items:
        reasons.append("No explicit action items found")
    return TaskOutput(request.task, "needs_review" if reasons else "completed",
                      f"{len(items)} Aufgaben aus dem Protokoll erfasst.",
                      [e for item in items for e in item.evidence], action_items=items, review_reasons=reasons)
