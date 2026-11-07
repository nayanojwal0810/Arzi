"""Unit tests for Arzi NLP topic discovery module using synthetic data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from arzi.nlp import (
    N_TOPICS,
    analyze_topic_issue_detail,
    assign_topics_to_complaints,
    clean_narrative,
    compute_reconstruction_similarity,
    compute_topic_diversity,
    infer_topics_for_unique_narratives,
    is_usable_narrative,
    load_model_artifacts,
    sample_unique_narratives,
    split_train_eval,
    validate_nlp_results,
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


def test_model_metadata_compatibility_check(tmp_path: Path) -> None:
    # Write metadata with mismatching dataset count
    incompatible_meta = {
        "dataset_record_count": 50000,  # Expected 117214
        "model_config": {"n_components": N_TOPICS},
    }
    meta_path = tmp_path / "model_metadata.json"
    meta_path.write_text(json.dumps(incompatible_meta), encoding="utf-8")
    (tmp_path / "vectorizer.joblib").write_text("dummy")
    (tmp_path / "topic_model.joblib").write_text("dummy")

    with pytest.raises(ValueError, match="dataset count mismatch"):
        load_model_artifacts(tmp_path, expected_dataset_count=117214)


def test_topic_issue_detail_denominators() -> None:
    # Issue A: 100 complaints total
    # Topic 0: 60 complaints (60%)
    # Topic 1: 30 complaints (30%)
    # Topic 2: 10 complaints (10%)
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
    # All 3 topics have >= 10% share
    assert res["topics_with_at_least_10pct_share"] == 3


def test_reconstruction_similarity_computation() -> None:
    # Identical matrices should have cosine similarity 1.0
    X = sparse.csr_matrix([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
    X_pred = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
    sim = compute_reconstruction_similarity(X, X_pred)
    assert pytest.approx(sim, abs=1e-5) == 1.0


def test_topic_diversity_computation() -> None:
    # 2 topics, 3 terms each. If all 6 terms distinct -> 1.0
    terms_distinct = [["a", "b", "c"], ["d", "e", "f"]]
    assert compute_topic_diversity(terms_distinct) == 1.0

    # If 3 terms overlap -> 3 distinct out of 6 -> 0.5
    terms_overlap = [["a", "b", "c"], ["a", "b", "c"]]
    assert compute_topic_diversity(terms_overlap) == 0.5
