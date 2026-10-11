from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Reasoning-effort choices each provider accepts; "default" sends nothing and
# leaves the provider's own default in place.
EFFORT_LEVELS: dict[str, tuple[str, ...]] = {
    "gemini": ("default", "off", "minimal", "low", "medium", "high"),
    "openai": ("default", "none", "minimal", "low", "medium", "high"),
    "glm": ("default", "off", "on"),
}


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str = "string"
    required: bool = False
    description: str = ""
    # With second_opinion: when_needed, one model's value is accepted only if
    # it is printed in the PDF text. Turn off for labels the model assigns
    # (such as a document classification) rather than reads.
    verify_in_text: bool = True
    # Labels printed next to the value ("Invoice No", "Inv #"). Rules-only
    # extraction looks for them, and models receive them as hints.
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    model: str
    api_key_env: str
    base_url: str | None = None
    structured_output: bool = False
    # "primary" models extract fields (see second_opinion for how many run per
    # document); an optional "arbiter" model is called only when the primaries
    # disagree or one of them fails.
    role: str = "primary"
    # "images" sends rendered JPEG pages; "pdf" sends native PDF pages
    # (flat per-page token pricing on Gemini) instead of JPEGs.
    input: str = "images"
    # One of EFFORT_LEVELS[provider].
    effort: str = "default"


# Title lines count only when a whole text line reads exactly like one of
# these, so "INVOICE NO" or "This receipt is valid..." never match.
DEFAULT_TITLES: dict[str, tuple[str, ...]] = {
    "invoice": ("INVOICE", "TAX INVOICE", "INVOIS", "INVOIS CUKAI"),
    "receipt": ("RECEIPT", "OFFICIAL RECEIPT", "PAYMENT RECEIPT", "CASH RECEIPT", "RESIT", "RESIT RASMI"),
    "statement": ("STATEMENT", "STATEMENT OF ACCOUNT", "DEBTOR STATEMENT", "CUSTOMER STATEMENT", "PENYATA AKAUN"),
    "credit_note": ("CREDIT NOTE", "NOTA KREDIT"),
    "debit_note": ("DEBIT NOTE", "NOTA DEBIT"),
    "delivery_order": ("DELIVERY ORDER", "DELIVERY NOTE"),
    "purchase_order": ("PURCHASE ORDER",),
    "quotation": ("QUOTATION",),
}


@dataclass(frozen=True)
class BoundaryRulesConfig:
    enabled: bool = True
    # "ignore": blank pages are never sent to a model and never decide a
    # boundary. "separator": a blank page also forces the next page to start a
    # new document unless that page is marked as a continuation.
    blank_pages: str = "ignore"
    # Share of dark pixels below which a page without text counts as blank.
    blank_max_ink: float = 0.005
    # Without a boundary model, pages the rules cannot decide are treated as
    # "new_document" or "same_document".
    fallback: str = "new_document"
    titles: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_TITLES))


@dataclass
class ModelResult:
    name: str
    fields: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    provider: str | None = None
    model_id: str | None = None


@dataclass
class DocumentResult:
    index: int
    page_start: int
    page_end: int
    fields: dict[str, Any]
    validation_status: str
    duplicate_status: str = "NONE"
    duplicate_of: str | None = None
    field_issues: list[str] = field(default_factory=list)
    model_results: list[ModelResult] = field(default_factory=list)
    remarks: list[str] = field(default_factory=list)


@dataclass
class BatchResult:
    source_name: str
    source_sha256: str
    config_fingerprint: str
    created_at: str
    documents: list[DocumentResult]
    boundary_audit: list[dict[str, Any]] = field(default_factory=list)
    page_decisions: list[dict[str, Any]] = field(default_factory=list)
    # Which models (or rules only) produced this result, for the Run info sheet.
    mode: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BatchResult":
        documents: list[DocumentResult] = []
        for raw_doc in value.get("documents", []):
            raw = dict(raw_doc)
            raw["model_results"] = [ModelResult(**item) for item in raw.get("model_results", [])]
            documents.append(DocumentResult(**raw))
        return cls(
            source_name=str(value["source_name"]),
            source_sha256=str(value["source_sha256"]),
            config_fingerprint=str(value["config_fingerprint"]),
            created_at=str(value["created_at"]),
            documents=documents,
            boundary_audit=list(value.get("boundary_audit", [])),
            page_decisions=list(value.get("page_decisions", [])),
            mode=str(value.get("mode", "")),
        )
