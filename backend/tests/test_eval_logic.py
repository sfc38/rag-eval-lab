"""Unit tests for the pure evaluation logic (metrics, matching, dataset, artifacts).

These require no model/vector-store dependencies, so they run anywhere with stdlib only.
Run directly:  ``PYTHONPATH=backend python3 backend/tests/test_eval_logic.py``
or with pytest: ``pytest backend/tests/test_eval_logic.py``
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from app.config import PipelineConfig
from app.eval import artifacts
from app.eval.dataset import load_dataset
from app.eval.retrieval_metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    relevant_chunk_indices,
    span_matches_chunk,
)

SAMPLE_DATASET = Path(__file__).resolve().parents[2] / "eval_data" / "sample" / "golden_qa.jsonl"


def test_recall_at_k() -> None:
    relevant = {2, 5}
    ranked = [2, 7, 5, 1]
    assert recall_at_k(relevant, ranked, 1) == 0.5
    assert recall_at_k(relevant, ranked, 3) == 1.0
    assert recall_at_k(relevant, ranked, 10) == 1.0
    assert recall_at_k(set(), ranked, 5) == 0.0  # no relevant -> 0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank({5}, [2, 7, 5]) == 1.0 / 3
    assert reciprocal_rank({2}, [2, 7, 5]) == 1.0
    assert reciprocal_rank({9}, [2, 7, 5]) == 0.0


def test_ndcg_at_k() -> None:
    assert ndcg_at_k({2}, [2, 7, 1], 3) == 1.0  # relevant first -> perfect
    got = ndcg_at_k({2}, [7, 2, 1], 3)          # relevant at rank 2
    assert math.isclose(got, 1.0 / math.log2(3), rel_tol=1e-6)
    assert ndcg_at_k(set(), [1, 2], 3) == 0.0


def test_span_matching() -> None:
    chunk = "Overlap between consecutive chunks helps preserve information across boundaries."
    assert span_matches_chunk("Overlap between consecutive chunks", chunk)         # substring
    assert span_matches_chunk("overlap   CONSECUTIVE chunks helps preserve", chunk)  # normalized
    assert not span_matches_chunk("a completely unrelated sentence about cats", chunk)


def test_relevant_chunk_indices() -> None:
    chunks = [
        (0, "RAG combines a language model with an external knowledge source."),
        (1, "The first is retrieval miss, where the passage is never retrieved."),
        (2, "Reranking uses a cross-encoder to reorder passages."),
    ]
    relevant = relevant_chunk_indices(chunks, ["The first is retrieval miss"])
    assert relevant == {1}
    assert relevant_chunk_indices(chunks, []) == set()  # unanswerable -> empty


def test_dataset_loads_and_validates() -> None:
    records = load_dataset(SAMPLE_DATASET)
    assert len(records) == 8
    by_id = {r.id: r for r in records}
    assert by_id["q_003"].question_type == "multi-hop"
    assert len(by_id["q_003"].evidence_spans) == 3
    assert by_id["q_008"].is_unanswerable
    assert by_id["q_001"].document_id == "sample_rag_overview"
    # document filter
    filtered = load_dataset(SAMPLE_DATASET, document_id="does_not_exist")
    assert filtered == []


def test_evidence_spans_are_verbatim_in_source() -> None:
    """Every evidence span must be an exact substring of the source document."""
    source = (SAMPLE_DATASET.parent / "sample_rag_overview.txt").read_text(encoding="utf-8")
    for record in load_dataset(SAMPLE_DATASET):
        for span in record.evidence_spans:
            assert span in source, f"{record.id}: span not verbatim in source: {span!r}"


def test_artifacts_round_trip() -> None:
    config = PipelineConfig()
    rows = [{"question_id": "q_001", "recall@5": 1.0}, {"question_id": "q_002", "recall@5": 0.5}]
    summary = {"n_questions": 2, "metrics": {"recall@5": 0.75}}
    original_dir = artifacts.RUNS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        artifacts.RUNS_DIR = Path(tmp)
        try:
            run_id = artifacts.new_run_id(config.config_hash())
            artifacts.write_run(run_id, config.to_dict(), rows, summary)
            loaded = artifacts.load_run(run_id)
            assert loaded is not None
            assert loaded["summary"]["n_questions"] == 2
            assert len(loaded["rows"]) == 2
            assert any(s["n_questions"] == 2 for s in artifacts.list_runs())
        finally:
            artifacts.RUNS_DIR = original_dir


def _run_all() -> None:
    """Run every ``test_*`` in this module and print a summary."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)} test(s) passed.")


if __name__ == "__main__":
    _run_all()
