"""Normalized outputs shared by every runner; evidence quotes must be verbatim."""
from dataclasses import asdict, dataclass, field
from typing import Any

TASKS = ("invoice_triage", "document_question", "meeting_followup")


def _text(value: Any, name: str, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class Evidence:
    document_id: str
    quote: str

    @classmethod
    def from_dict(cls, raw: dict) -> "Evidence":
        if not isinstance(raw, dict) or set(raw) != {"document_id", "quote"}:
            raise ValueError("evidence requires document_id and quote")
        return cls(_text(raw["document_id"], "document_id"), _text(raw["quote"], "quote"))


@dataclass(frozen=True)
class ActionItem:
    description: str
    owner: str | None
    due_date: str | None
    evidence: list[Evidence]

    @classmethod
    def from_dict(cls, raw: dict) -> "ActionItem":
        if not isinstance(raw, dict) or set(raw) != {"description", "owner", "due_date", "evidence"}:
            raise ValueError("invalid action item fields")
        if not isinstance(raw["evidence"], list):
            raise ValueError("action evidence must be a list")
        return cls(_text(raw["description"], "description"), _text(raw["owner"], "owner", True),
                   _text(raw["due_date"], "due_date", True), [Evidence.from_dict(x) for x in raw["evidence"]])


@dataclass(frozen=True)
class TaskOutput:
    task: str
    status: str
    answer: str
    evidence: list[Evidence]
    invoice: dict | None = None
    action_items: list[ActionItem] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "TaskOutput":
        fields = {"task", "status", "answer", "evidence", "invoice", "action_items", "review_reasons"}
        if not isinstance(raw, dict) or set(raw) != fields:
            raise ValueError("output must contain exactly the normalized output fields")
        if raw["task"] not in TASKS or raw["status"] not in ("completed", "needs_review"):
            raise ValueError("invalid task or status")
        for key in ("evidence", "action_items", "review_reasons"):
            if not isinstance(raw[key], list):
                raise ValueError(f"{key} must be a list")
        if raw["invoice"] is not None:
            invoice = raw["invoice"]
            if not isinstance(invoice, dict) or set(invoice) != {"invoice_number", "supplier", "amount_eur", "decision"}:
                raise ValueError("invalid invoice fields")
            for key in ("invoice_number", "supplier", "amount_eur"):
                _text(invoice[key], key, True)
            if invoice["decision"] not in ("ready_for_review", "hold"):
                raise ValueError("invalid invoice decision")
        for reason in raw["review_reasons"]:
            _text(reason, "review reason")
        if raw["status"] == "needs_review" and not raw["review_reasons"]:
            raise ValueError("needs_review requires a reason")
        return cls(raw["task"], raw["status"], _text(raw["answer"], "answer"),
                   [Evidence.from_dict(x) for x in raw["evidence"]], raw["invoice"],
                   [ActionItem.from_dict(x) for x in raw["action_items"]], raw["review_reasons"])
