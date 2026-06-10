"""Golden QA dataset loading and validation.

The golden dataset is the measuring stick for every experiment. Records are stored as JSONL
so they diff cleanly in git and stream without loading everything into memory. ``evidence_spans``
are verbatim snippets from the source document; they let us label retrieved chunks as
relevant/irrelevant automatically (see ``retrieval_metrics``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

QUESTION_TYPES = {"factual", "multi-hop", "unanswerable"}
DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass
class GoldenRecord:
    """One golden QA record.

    Attributes:
        id: Stable identifier, e.g. ``q_001``.
        document_id: Identifier of the source document this question targets.
        question: The natural-language question.
        reference_answer: The verified correct answer (empty for unanswerable questions).
        evidence_spans: Verbatim snippets from the source that contain the answer.
            Empty for ``unanswerable`` questions.
        question_type: One of ``factual`` / ``multi-hop`` / ``unanswerable``.
        difficulty: One of ``easy`` / ``medium`` / ``hard``.
    """

    id: str
    document_id: str
    question: str
    reference_answer: str
    evidence_spans: list[str] = field(default_factory=list)
    question_type: str = "factual"
    difficulty: str = "medium"

    @property
    def is_unanswerable(self) -> bool:
        """Whether this question has no supporting evidence in the document."""
        return self.question_type == "unanswerable" or not self.evidence_spans


def load_dataset(path: Path, document_id: str | None = None) -> list[GoldenRecord]:
    """Load and validate a golden dataset from a JSONL file.

    Args:
        path: Path to the ``.jsonl`` dataset.
        document_id: If given, return only records whose ``document_id`` matches.

    Returns:
        list[GoldenRecord]: Validated records.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If any record is malformed (message includes the line number).
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records: list[GoldenRecord] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_no}: invalid JSON ({exc})") from exc
        record = _validate_record(raw, line_no)
        if record.id in seen_ids:
            raise ValueError(f"Line {line_no}: duplicate record id {record.id!r}")
        seen_ids.add(record.id)
        records.append(record)

    if document_id is not None:
        records = [r for r in records if r.document_id == document_id]

    logger.info("Loaded %d golden record(s) from %s", len(records), path.name)
    return records


def _validate_record(raw: dict, line_no: int) -> GoldenRecord:
    """Validate one raw record dict and build a :class:`GoldenRecord`.

    Args:
        raw: Parsed JSON object for a single record.
        line_no: 1-based line number, for error messages.

    Returns:
        GoldenRecord: The validated record.

    Raises:
        ValueError: If a required field is missing or an enum value is invalid.
    """
    required = ("id", "document_id", "question")
    for key in required:
        if not raw.get(key):
            raise ValueError(f"Line {line_no}: missing required field {key!r}")

    question_type = raw.get("question_type", "factual")
    if question_type not in QUESTION_TYPES:
        raise ValueError(
            f"Line {line_no}: question_type {question_type!r} not in {sorted(QUESTION_TYPES)}"
        )
    difficulty = raw.get("difficulty", "medium")
    if difficulty not in DIFFICULTIES:
        raise ValueError(
            f"Line {line_no}: difficulty {difficulty!r} not in {sorted(DIFFICULTIES)}"
        )

    evidence_spans = raw.get("evidence_spans", []) or []
    if not isinstance(evidence_spans, list):
        raise ValueError(f"Line {line_no}: evidence_spans must be a list")
    if question_type != "unanswerable" and not evidence_spans:
        raise ValueError(
            f"Line {line_no}: answerable question {raw['id']!r} has no evidence_spans"
        )

    return GoldenRecord(
        id=str(raw["id"]),
        document_id=str(raw["document_id"]),
        question=str(raw["question"]),
        reference_answer=str(raw.get("reference_answer", "")),
        evidence_spans=[str(s) for s in evidence_spans],
        question_type=question_type,
        difficulty=difficulty,
    )
