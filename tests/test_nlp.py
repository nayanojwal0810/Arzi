"""Unit tests for Arzi NLP topic discovery module using synthetic data."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from arzi.nlp import (
    CLEANING_VERSION,
    HELD_OUT_RATIO,
    MAX_PER_PRODUCT_SAMPLE,
    N_TOPICS,
    NMF_PARAMS,
    RANDOM_SEED,
    TOP_TERMS_COUNT,
    VECTORIZER_PARAMS,
    analyze_topic_issue_detail,
    assign_topics_to_complaints,
    clean_narrative,
    compute_reconstruction_similarity,
    compute_source_fingerprint,
    compute_topic_diversity,
    infer_topics_for_unique_narratives,
    is_usable_narrative,
    load_and_validate_topic_results_in_transaction,
    load_model_artifacts,
    run_in_transaction_sql_checks,
    sample_unique_narratives,
    split_train_eval,
    validate_in_memory_results,
    verify_model_metadata,
)


def test_redaction_placeholder_cleanup() -> None:
    # Obvious redactions should be replaced with whitespace
    raw = "My account was disputed on XX/XX/2019 by XXXX with balance {$1500.00} and SSN XXX-XX-XXXX."
    cleaned = clean_narrative(raw)

    # Redactions must be gone
    assert "XX/XX/2019" not in cleaned
    assert "XXXX" not in cleaned
    assert "{$1500.00}" not in cleaned
    assert "XXX-XX-XXXX" not in cleaned

    # Financial vocabulary must be preserved
    for word in ["account", "disputed", "balance"]:
        assert word in cleaned.lower()


def test_narrative_text_preservation() -> None:
    original = "Original complaint narrative with {$500} and XXXX."
    cleaned = clean_narrative(original)

    # Original text variable must be completely unmodified
    assert original == "Original complaint narrative with {$500} and XXXX."
    assert cleaned != original
    assert "XXXX" not in cleaned


def test_deterministic_sampling() -> None:
    synthetic_complaints = [
        {"complaint_id": i, "product": f"Product {i % 3}", "cleaned_narrative": f"Unique narrative body {i}"}
        for i in range(30)
    ]

    sample_1, counts_1 = sample_unique_narratives(synthetic_complaints, max_per_product=5, random_seed=42)
    sample_2, counts_2 = sample_unique_narratives(synthetic_complaints, max_per_product=5, random_seed=42)

    assert sample_1 == sample_2
    assert counts_1 == counts_2
    assert len(sample_1) == 15  # 3 products * 5 each
    # No duplicate text in sample
    assert len(sample_1) == len(set(sample_1))


def test_duplicate_narrative_handling() -> None:
    # 5 complaints with the exact same narrative across different products
    identical_text = "Standard repeated dispute letter text."
    synthetic_complaints = [
        {"complaint_id": 1, "product": "Mortgage", "cleaned_narrative": identical_text},
        {"complaint_id": 2, "product": "Debt collection", "cleaned_narrative": identical_text},
        {"complaint_id": 3, "product": "Credit reporting", "cleaned_narrative": identical_text},
    ]

    sample, counts = sample_unique_narratives(synthetic_complaints, max_per_product=2, random_seed=42)

    # Across all products, the identical text must appear at most once in the sample
    assert sample.count(identical_text) == 1
    assert len(sample) == 1


def test_train_eval_split_no_leakage() -> None:
    sample = [f"Unique text item {i}" for i in range(100)]
    train_texts, eval_texts = split_train_eval(sample, held_out_ratio=0.10, random_seed=42)

    assert len(train_texts) == 90
    assert len(eval_texts) == 10
    # Strict separation
    assert set(train_texts).isdisjoint(set(eval_texts))
    assert len(set(train_texts) | set(eval_texts)) == 100


def test_valid_topic_assignment_and_strength() -> None:
    # Train simple 2-topic model on distinct vocabularies
    texts = [
        "mortgage loan payment interest rate monthly",
        "mortgage payment home escrow balance lender",
        "credit card debt dispute charge annual fee",
        "credit card billing late fee dispute charges",
    ]
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    nmf = NMF(n_components=2, random_state=42, max_iter=200)
    nmf.fit(X)

    eval_unique = [
        "mortgage loan monthly payment",
        "credit card dispute fee",
    ]
    assignments = infer_topics_for_unique_narratives(eval_unique, vec, nmf)

    for text, (tid, strength, usable) in assignments.items():
        assert tid in [0, 1]
        assert strength is not None
        assert 0.0 <= strength <= 1.0
        assert usable is True


def test_handling_empty_or_unusable_narratives() -> None:
    # Only redaction placeholders that clean to empty or non-alphabetic
    assert clean_narrative("XXXX XX") == ""
    assert is_usable_narrative("") is False
    assert is_usable_narrative("12345 6789") is False

    # Inference on unrepresentable text
    vec = TfidfVectorizer()
    vec.fit(["regular financial text here"])
    nmf = NMF(n_components=2, random_state=42, max_iter=50)
    nmf.fit(vec.transform(["regular financial text here"]))

    assignments = infer_topics_for_unique_narratives(["unseen uncommon words only"], vec, nmf)
    tid, strength, usable = assignments["unseen uncommon words only"]

    assert tid is None
    assert strength is None
    assert usable is False

    complaints = [{"complaint_id": 999, "cleaned_narrative": "unseen uncommon words only"}]
    assigned_records = assign_topics_to_complaints(complaints, assignments)
    assert len(assigned_records) == 1
    assert assigned_records[0]["complaint_id"] == 999
    assert assigned_records[0]["topic_id"] is None
    assert assigned_records[0]["topic_strength"] is None
    assert assigned_records[0]["usable_for_nlp"] is False


def test_topic_count_reconciliation() -> None:
    complaints = [
        {"complaint_id": 1, "cleaned_narrative": "text a"},
        {"complaint_id": 2, "cleaned_narrative": "text b"},
        {"complaint_id": 3, "cleaned_narrative": ""},
    ]
    text_assignments = {
        "text a": (0, 0.85, True),
        "text b": (1, 0.90, True),
    }

    assigned_records = assign_topics_to_complaints(complaints, text_assignments)
    assigned_count = sum(1 for r in assigned_records if r["topic_id"] is not None)
    unassigned_count = sum(1 for r in assigned_records if r["topic_id"] is None)

    assert len(assigned_records) == 3
    assert assigned_count == 2
    assert unassigned_count == 1
    assert assigned_count + unassigned_count == len(complaints)


def test_reconstruction_similarity_with_zero_vectors() -> None:
    # 1. Identical matrices should have cosine similarity 1.0
    X_true = sparse.csr_matrix([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
    X_pred = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
    sim, n_docs, zero_cnt = compute_reconstruction_similarity(X_true, X_pred)
    assert pytest.approx(sim, abs=1e-4) == 1.0
    assert n_docs == 2
    assert zero_cnt == 0

    # 2. An additional zero-vector row must receive 0.0 similarity and lower the overall average
    X_true_3 = sparse.csr_matrix([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    X_pred_3 = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    sim_3, n_docs_3, zero_cnt_3 = compute_reconstruction_similarity(X_true_3, X_pred_3)

    assert sim_3 < sim
    assert pytest.approx(sim_3, abs=1e-4) == 0.6667  # (1.0 + 1.0 + 0.0) / 3
    assert n_docs_3 == 3
    assert zero_cnt_3 == 1


def test_source_fingerprint_deterministic_and_sensitive() -> None:
    records = [
        {
            "complaint_id": 100,
            "product": "Credit card",
            "issue": "Billing",
            "date_received": date(2019, 5, 10),
            "consumer_complaint_narrative": "Disputed interest charge.",
        },
        {
            "complaint_id": 200,
            "product": "Mortgage",
            "issue": "Escrow",
            "date_received": date(2019, 6, 15),
            "consumer_complaint_narrative": "Incorrect escrow calculation.",
        },
    ]

    # Deterministic
    fp1 = compute_source_fingerprint(records)
    fp2 = compute_source_fingerprint(records)
    assert fp1 == fp2
    assert len(fp1) == 64

    # Order-insensitive (sorted internally by complaint_id)
    fp_reversed = compute_source_fingerprint(list(reversed(records)))
    assert fp1 == fp_reversed

    # Sensitive to text change
    rec_modified = [dict(records[0]), dict(records[1])]
    rec_modified[0]["consumer_complaint_narrative"] = "Different narrative"
    assert compute_source_fingerprint(rec_modified) != fp1

    # Sensitive to date change
    rec_date = [dict(records[0]), dict(records[1])]
    rec_date[0]["date_received"] = date(2019, 5, 11)
    assert compute_source_fingerprint(rec_date) != fp1

    # Sensitive to product change
    rec_prod = [dict(records[0]), dict(records[1])]
    rec_prod[0]["product"] = "Debt collection"
    assert compute_source_fingerprint(rec_prod) != fp1

    # Handles None values safely
    rec_none = [
        {
            "complaint_id": 300,
            "product": None,
            "issue": None,
            "date_received": None,
            "consumer_complaint_narrative": None,
        }
    ]
    fp_none = compute_source_fingerprint(rec_none)
    assert len(fp_none) == 64


def _build_valid_metadata() -> dict[str, Any]:
    return {
        "source_data_fingerprint": "a" * 64,
        "dataset_record_count": 117214,
        "cleaning_version": CLEANING_VERSION,
        "sampling_config": {
            "random_seed": RANDOM_SEED,
            "max_per_product": MAX_PER_PRODUCT_SAMPLE,
        },
        "train_eval_split_config": {
            "held_out_ratio": HELD_OUT_RATIO,
            "random_seed": RANDOM_SEED,
        },
        "vectorizer_config": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in VECTORIZER_PARAMS.items()
        },
        "model_config": dict(NMF_PARAMS),
        "topics_count": N_TOPICS,
        "top_terms_count": TOP_TERMS_COUNT,
    }


def test_verify_model_metadata_compatibility() -> None:
    valid_meta = _build_valid_metadata()
    expected_fp = "a" * 64

    # Valid metadata passes
    verify_model_metadata(valid_meta, expected_fingerprint=expected_fp, expected_dataset_count=117214)

    # Changed fingerprint raises
    with pytest.raises(ValueError, match="Source-data fingerprint mismatch"):
        verify_model_metadata(valid_meta, expected_fingerprint="b" * 64, expected_dataset_count=117214)

    # Changed record count raises
    with pytest.raises(ValueError, match="Dataset record count mismatch"):
        verify_model_metadata(valid_meta, expected_fingerprint=expected_fp, expected_dataset_count=50000)

    # Changed cleaning version raises
    bad_clean = dict(valid_meta)
    bad_clean["cleaning_version"] = "v0.9"
    with pytest.raises(ValueError, match="Cleaning version mismatch"):
        verify_model_metadata(bad_clean, expected_fingerprint=expected_fp, expected_dataset_count=117214)

    # Changed sampling config raises
    bad_sample = dict(valid_meta)
    bad_sample["sampling_config"] = {"random_seed": 999, "max_per_product": 100}
    with pytest.raises(ValueError, match="Sampling configuration mismatch"):
        verify_model_metadata(bad_sample, expected_fingerprint=expected_fp, expected_dataset_count=117214)

    # Changed vectorizer config raises
    bad_vec = dict(valid_meta)
    bad_vec["vectorizer_config"] = {"max_features": 500}
    with pytest.raises(ValueError, match="Vectorizer configuration mismatch"):
        verify_model_metadata(bad_vec, expected_fingerprint=expected_fp, expected_dataset_count=117214)

    # Changed NMF config raises
    bad_nmf = dict(valid_meta)
    bad_nmf["model_config"] = {"n_components": 5}
    with pytest.raises(ValueError, match="NMF model configuration mismatch"):
        verify_model_metadata(bad_nmf, expected_fingerprint=expected_fp, expected_dataset_count=117214)


def test_load_model_artifacts_file_handling(tmp_path: Path) -> None:
    valid_meta = _build_valid_metadata()
    expected_fp = "a" * 64

    # Missing files raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_model_artifacts(tmp_path, expected_fingerprint=expected_fp)

    # Incompatible metadata raises ValueError
    meta_path = tmp_path / "model_metadata.json"
    meta_path.write_text(json.dumps({"dataset_record_count": 10}), encoding="utf-8")
    (tmp_path / "vectorizer.joblib").write_text("dummy")
    (tmp_path / "topic_model.joblib").write_text("dummy")

    with pytest.raises(ValueError, match="Source-data fingerprint mismatch"):
        load_model_artifacts(tmp_path, expected_fingerprint=expected_fp)


def test_in_memory_validation_rules() -> None:
    # 1. Mismatched total records
    complaints = [{"complaint_id": 1, "product": "Mortgage"}]
    catalog = [{"topic_id": i, "top_terms": ["term"]} for i in range(10)]
    topics = [{"complaint_id": 1, "topic_id": 0, "topic_strength": 0.8, "usable_for_nlp": True}]
    metrics = {"nmf_converged": True, "topic_diversity": 0.9}

    # Fails record count requirement (117,214)
    with pytest.raises(ValueError, match="Source complaints count mismatch"):
        validate_in_memory_results(complaints, catalog, topics, metrics)


def test_load_and_validate_topic_results_rollback_on_failure(tmp_path: Path) -> None:
    # Test transactional rollback behavior
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Create dummy schema file
    dummy_schema = tmp_path / "schema.sql"
    dummy_schema.write_text("SELECT 1;", encoding="utf-8")

    topic_catalog = [{"topic_id": 0, "top_terms": ["a", "b"]}]
    complaint_topics = [{"complaint_id": 1, "topic_id": 0, "topic_strength": 0.9, "usable_for_nlp": True}]

    # Simulate error in SQL checks
    mock_cur.execute.side_effect = Exception("Simulated DB validation failure")

    with pytest.raises(Exception, match="Simulated DB validation failure"):
        load_and_validate_topic_results_in_transaction(
            conn=mock_conn,
            topic_catalog=topic_catalog,
            complaint_topics=complaint_topics,
            expected_topic_counts={0: 1},
            nlp_schema_path=dummy_schema,
        )

    # Rollback must be called, commit must NOT be called
    mock_conn.rollback.assert_called_once()
    mock_conn.commit.assert_not_called()


def test_topic_issue_detail_denominators() -> None:
    synthetic_issue_totals = [("Issue A", 100)]
    synthetic_issue_topics = [
        ("Issue A", 0, 60),
        ("Issue A", 1, 30),
        ("Issue A", 2, 10),
    ]

    results = analyze_topic_issue_detail(
        issue_topic_rows=synthetic_issue_topics, issue_totals=synthetic_issue_totals
    )
    assert len(results) == 1
    res = results[0]
    assert res["issue"] == "Issue A"
    assert res["total_complaints"] == 100
    assert len(res["top_three_topics"]) == 3
    assert res["top_three_topics"][0]["share_within_issue_pct"] == 60.0
    assert res["top_three_topics"][1]["share_within_issue_pct"] == 30.0
    assert res["top_three_topics"][2]["share_within_issue_pct"] == 10.0
    assert res["topics_with_at_least_10pct_share"] == 3


def test_topic_diversity_computation() -> None:
    terms_distinct = [["a", "b", "c"], ["d", "e", "f"]]
    assert compute_topic_diversity(terms_distinct) == 1.0

    terms_overlap = [["a", "b", "c"], ["a", "b", "c"]]
    assert compute_topic_diversity(terms_overlap) == 0.5
