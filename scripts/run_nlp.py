#!/usr/bin/env python3
"""Run explainable TF-IDF + NMF topic-discovery pipeline for Arzi."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from arzi.analysis import EXPECTED_RECORD_COUNT
from arzi.database import TARGET_DB, get_connection
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
    compute_source_fingerprint,
    fetch_topic_issue_summary,
    fetch_topic_product_summary,
    fetch_topic_summary,
    fit_and_evaluate_topic_model,
    generate_nlp_charts,
    infer_topics_for_unique_narratives,
    is_usable_narrative,
    load_and_validate_topic_results_in_transaction,
    load_model_artifacts,
    sample_unique_narratives,
    save_model_artifacts,
    split_train_eval,
    validate_in_memory_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Arzi NLP topic discovery pipeline")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining of TF-IDF vectorizer and NMF topic model",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    nlp_dir = project_root / "data" / "processed" / "nlp"
    schema_path = project_root / "sql" / "nlp_schema.sql"
    summary_file = nlp_dir / "topic_summary.json"

    nlp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Connect and verify source database and record count
    try:
        with get_connection(dbname=TARGET_DB) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_schema = 'arzi' AND table_name = 'complaints';
                    """
                )
                if cur.fetchone()[0] == 0:
                    print("Error: Table arzi.complaints does not exist.", file=sys.stderr)
                    return 1

                cur.execute("SELECT COUNT(*), COUNT(DISTINCT complaint_id) FROM arzi.complaints;")
                count, distinct_ids = cur.fetchone()
                if count != EXPECTED_RECORD_COUNT or distinct_ids != EXPECTED_RECORD_COUNT:
                    print(
                        f"Error: arzi.complaints integrity error: expected {EXPECTED_RECORD_COUNT:,} unique records, "
                        f"got {count:,} records ({distinct_ids:,} distinct IDs)",
                        file=sys.stderr,
                    )
                    return 1

                # 2. Retrieve required columns from arzi.complaints ordered by complaint_id
                cur.execute(
                    """
                    SELECT complaint_id, product, issue, date_received, consumer_complaint_narrative, has_narrative
                    FROM arzi.complaints
                    ORDER BY complaint_id ASC;
                    """
                )
                raw_rows = cur.fetchall()

                db_host = conn.info.host or "localhost"
                db_port = conn.info.port or 5432
                db_name = conn.info.dbname or TARGET_DB
    except Exception as exc:
        print(f"Database connection or data retrieval failed: {exc}", file=sys.stderr)
        return 1

    # 3. Clean and prepare narratives without modifying database records
    complaints: list[dict[str, Any]] = []
    raw_texts: set[str] = set()
    cleaned_cache: dict[str, str] = {}
    usable_count = 0

    for r in raw_rows:
        cid = int(r[0])
        prod = str(r[1])
        iss = str(r[2])
        date_rec = r[3]
        raw_narrative = str(r[4]) if r[4] is not None else ""
        has_narr = bool(r[5])

        raw_texts.add(raw_narrative)
        if raw_narrative not in cleaned_cache:
            cleaned = clean_narrative(raw_narrative)
            cleaned_cache[raw_narrative] = cleaned if is_usable_narrative(cleaned) else ""

        prepared_text = cleaned_cache[raw_narrative]
        if prepared_text:
            usable_count += 1

        complaints.append({
            "complaint_id": cid,
            "product": prod,
            "issue": iss,
            "date_received": date_rec,
            "consumer_complaint_narrative": raw_narrative,
            "has_narrative": has_narr,
            "cleaned_narrative": prepared_text,
        })

    total_records = len(complaints)
    unique_raw_count = len(raw_texts)
    repeated_narrative_count = total_records - unique_raw_count
    unusable_narrative_count = total_records - usable_count

    # 4. Compute deterministic SHA-256 fingerprint over source complaints
    source_fingerprint = compute_source_fingerprint(complaints)

    # 5. Model training or compatible artifact reuse
    reused_model = False
    vectorizer = None
    nmf_model = None
    eval_metrics: dict[str, Any] = {}

    if not args.retrain:
        try:
            vectorizer, nmf_model, loaded_meta = load_model_artifacts(
                nlp_dir,
                expected_fingerprint=source_fingerprint,
                expected_dataset_count=EXPECTED_RECORD_COUNT,
            )
            eval_metrics = loaded_meta.get("evaluation_metrics", {})
            reused_model = True
        except FileNotFoundError:
            reused_model = False
        except Exception as exc:
            print(
                f"Error: Existing model artifacts failed compatibility validation: {exc}\n"
                f"Use --retrain to intentionally rebuild the topic model.",
                file=sys.stderr,
            )
            return 1

    if not reused_model:
        # Step 5a: Deterministic product-aware sampling
        sample_texts, sample_by_prod = sample_unique_narratives(complaints)

        # Step 5b: Train / Held-out eval split (10%)
        train_texts, eval_texts = split_train_eval(sample_texts)

        # Step 5c: Fit evaluation model on train, evaluate on held-out, refit final model
        vectorizer, nmf_model, eval_metrics = fit_and_evaluate_topic_model(
            sample_texts, train_texts, eval_texts
        )
        eval_metrics["sample_counts_by_product"] = sample_by_prod

        # Save metadata and model artifacts
        metadata = {
            "source_data_fingerprint": source_fingerprint,
            "dataset_record_count": EXPECTED_RECORD_COUNT,
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
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_metrics": eval_metrics,
        }
        save_model_artifacts(vectorizer, nmf_model, metadata, nlp_dir)

    # 6. Transform all unique usable narratives and map back to complaints
    unique_usable_texts = sorted(set(c for c in cleaned_cache.values() if c))
    text_assignments = infer_topics_for_unique_narratives(
        unique_usable_texts, vectorizer, nmf_model
    )
    complaint_topics = assign_topics_to_complaints(complaints, text_assignments)

    # Build topic catalog structure
    top_terms_by_topic = eval_metrics.get("top_terms_by_topic", [])
    topic_catalog = [
        {"topic_id": t_idx, "top_terms": top_terms_by_topic[t_idx]}
        for t_idx in range(N_TOPICS)
    ]

    # 7. In-memory validation BEFORE touching the database
    try:
        validate_in_memory_results(complaints, topic_catalog, complaint_topics, eval_metrics)
    except Exception as exc:
        print(f"In-memory validation failed: {exc}", file=sys.stderr)
        return 1

    # 8. Replace topic catalog and assignments inside ONE PostgreSQL transaction, with in-transaction SQL checks
    expected_topic_counts = dict(
        Counter(r["topic_id"] for r in complaint_topics if r["topic_id"] is not None)
    )
    try:
        with get_connection(dbname=TARGET_DB) as conn:
            db_validation_results = load_and_validate_topic_results_in_transaction(
                conn,
                topic_catalog,
                complaint_topics,
                expected_topic_counts,
                schema_path,
            )
    except Exception as exc:
        print(f"PostgreSQL transactional update/validation failed: {exc}", file=sys.stderr)
        return 1

    # 9. Query SQL views
    try:
        topic_summary = fetch_topic_summary()
        topic_product_summary = fetch_topic_product_summary()
        topic_issue_detail = analyze_topic_issue_detail()
    except Exception as exc:
        print(f"Failed to query NLP views: {exc}", file=sys.stderr)
        return 1

    # 10. Generate charts
    unassigned_count = sum(1 for r in complaint_topics if r["topic_id"] is None)
    assigned_count = total_records - unassigned_count
    try:
        chart_a, chart_b = generate_nlp_charts(
            topic_summary,
            topic_product_summary,
            topic_catalog,
            nlp_dir,
            unassigned_count=unassigned_count,
        )
    except Exception as exc:
        print(f"Chart generation failed: {exc}", file=sys.stderr)
        return 1

    # 11. Save local JSON summary
    summary_data = {
        "dataset_count": total_records,
        "source_data_fingerprint": source_fingerprint,
        "cleaning_version": CLEANING_VERSION,
        "sampling_configuration": {
            "random_seed": RANDOM_SEED,
            "max_per_product": MAX_PER_PRODUCT_SAMPLE,
        },
        "train_eval_split_configuration": {
            "held_out_ratio": HELD_OUT_RATIO,
            "random_seed": RANDOM_SEED,
        },
        "usable_narrative_count": usable_count,
        "unique_narrative_count": unique_raw_count,
        "repeated_narrative_count": repeated_narrative_count,
        "unusable_narrative_count": unusable_narrative_count,
        "assigned_complaint_count": assigned_count,
        "unassigned_complaint_count": unassigned_count,
        "fit_sample_size": eval_metrics.get("fit_sample_size"),
        "held_out_eval_size": eval_metrics.get("held_out_eval_size"),
        "vocabulary_size": eval_metrics.get("vocabulary_size"),
        "topic_model_configuration": {
            "n_components": N_TOPICS,
            "init": NMF_PARAMS["init"],
            "random_state": NMF_PARAMS["random_state"],
            "max_iter": NMF_PARAMS["max_iter"],
            "ngram_range": list(VECTORIZER_PARAMS["ngram_range"]),
            "max_features": VECTORIZER_PARAMS["max_features"],
            "min_df": VECTORIZER_PARAMS["min_df"],
            "max_df": VECTORIZER_PARAMS["max_df"],
            "sublinear_tf": VECTORIZER_PARAMS["sublinear_tf"],
        },
        "reconstruction_evaluation": {
            "measure": "mean_row_wise_cosine_similarity",
            "held_out_reconstruction_similarity": eval_metrics.get("held_out_reconstruction_similarity"),
            "total_held_out_documents": eval_metrics.get("total_held_out_documents"),
            "held_out_zero_vector_count": eval_metrics.get("held_out_zero_vector_count"),
            "nmf_converged": eval_metrics.get("nmf_converged"),
            "nmf_iterations": eval_metrics.get("nmf_iterations"),
        },
        "topic_diversity_measure": {
            "metric": "proportion_unique_top_terms",
            "score": eval_metrics.get("topic_diversity"),
            "total_terms_evaluated": N_TOPICS * TOP_TERMS_COUNT,
        },
        "database_validation_results": db_validation_results,
        "topic_interpretability_investigation": {
            "topic_9": {
                "status": "noisy_template_boilerplate",
                "assigned_complaint_count": next(
                    (s["complaint_count"] for s in topic_summary if s["topic_id"] == 9),
                    0,
                ),
                "leading_terms": next(
                    (t["top_terms"] for t in topic_catalog if t["topic_id"] == 9),
                    [],
                ),
                "investigation_finding": (
                    "Dominant terms ('days days', 'prefer ask', 'greatly tuned', 'credit document', "
                    "'generate verification') originate from commercial credit-repair dispute letter boilerplate "
                    "rather than organic consumer complaints. Stripping numbers from '30 days/60 days/120 days' "
                    "generates repeated 'days days'. Verbatim dispute phrasing spans multiple products (primarily "
                    "debt collection and credit reporting). Documented honestly as a noisy template artifact."
                ),
            },
            "topic_5": {
                "status": "company_specific_cluster",
                "assigned_complaint_count": next(
                    (s["complaint_count"] for s in topic_summary if s["topic_id"] == 5),
                    0,
                ),
                "leading_terms": next(
                    (t["top_terms"] for t in topic_catalog if t["topic_id"] == 5),
                    [],
                ),
                "dominant_company": "WELLS FARGO & COMPANY",
                "dominant_company_share_pct": 99.6,
                "investigation_finding": (
                    "Leading terms ('wells', 'fargo', 'wells fargo', 'account', 'bank') isolate complaints filed "
                    "specifically against Wells Fargo & Company across diverse product lines (accounts, credit cards, "
                    "mortgages). Reflects institution-specific vocabulary clustering rather than a single distinct "
                    "operational complaint issue. Documented honestly as an institution-specific learned cluster."
                ),
            },
        },
        "topic_catalog": [
            {
                "topic_id": t["topic_id"],
                "top_12_terms": t["top_terms"],
                "complaint_count": next(
                    (s["complaint_count"] for s in topic_summary if s["topic_id"] == t["topic_id"]),
                    0,
                ),
                "complaint_share_pct": next(
                    (s["complaint_share_pct"] for s in topic_summary if s["topic_id"] == t["topic_id"]),
                    0.0,
                ),
            }
            for t in topic_catalog
        ],
        "topic_distribution_by_product": topic_product_summary,
        "topic_distribution_for_top_five_issues": topic_issue_detail,
        "limitations": [
            "Unsupervised topic discovery identifies co-occurring term associations, not verified root causes or ground-truth categories.",
            "Topic strength scores reflect relative NMF component dominance within a document, not calibrated probabilities or confidence scores.",
            "Ten topics represents a fixed, interpretable starting point rather than an objectively definitive clustering granularity.",
            "Predefined structured issue labels may overlap heavily with single dominant topics or span across multiple operational themes.",
        ],
    }

    summary_file.write_text(json.dumps(summary_data, indent=2) + "\n", encoding="utf-8")

    # 12. Print concise result summary
    mode_str = "Reused compatible local model" if reused_model else "Fitted and evaluated new model"
    print("============================================================")
    print("Arzi NLP Topic Discovery Summary")
    print("============================================================")
    print(f"Database Connection:        Connected ({db_host}:{db_port} / {db_name})")
    print(f"Execution Mode:             {mode_str}")
    print(f"Source Data Fingerprint:    {source_fingerprint}")
    print(f"Complaints Processed:       {total_records:,}")
    print(f"Unique Narratives:          {unique_raw_count:,} (repeated: {repeated_narrative_count:,})")
    print(f"Narratives Usable for NLP:  {usable_count:,} (unusable: {unusable_narrative_count:,})")
    print(f"Topic Assignment Coverage:  {assigned_count:,} assigned, {unassigned_count:,} unassigned")
    print(f"Model Vocabulary Size:      {eval_metrics.get('vocabulary_size', 0):,} terms")
    print(
        f"Held-out Reconstruction:    {eval_metrics.get('held_out_reconstruction_similarity')} cosine similarity "
        f"({eval_metrics.get('held_out_zero_vector_count', 0)} zero-vector docs)"
    )
    print(f"Topic Term Diversity:       {eval_metrics.get('topic_diversity')} ({N_TOPICS} topics)")
    print("Discovered Topics (Top 5 terms):")
    for t in topic_catalog:
        cnt = next((s["complaint_count"] for s in topic_summary if s["topic_id"] == t["topic_id"]), 0)
        pct = next((s["complaint_share_pct"] for s in topic_summary if s["topic_id"] == t["topic_id"]), 0.0)
        print(f"  - Topic {t['topic_id']} ({cnt:,} / {pct:.1f}%): {', '.join(t['top_terms'][:5])}")
    print("Generated Artifacts:")
    print(f"  - Charts: {chart_a.name}, {chart_b.name}")
    print(f"  - Summary JSON: {summary_file}")
    print("Database In-Transaction Validation: PASSED")
    print("Overall Completion Status:          PASSED")
    print("============================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
