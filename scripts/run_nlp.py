#!/usr/bin/env python3
"""Run explainable TF-IDF + NMF topic-discovery pipeline for Arzi."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arzi.analysis import EXPECTED_RECORD_COUNT
from arzi.database import TARGET_DB, get_connection
from arzi.nlp import (
    N_TOPICS,
    NMF_PARAMS,
    TOP_TERMS_COUNT,
    VECTORIZER_PARAMS,
    analyze_topic_issue_detail,
    assign_topics_to_complaints,
    clean_narrative,
    fetch_topic_issue_summary,
    fetch_topic_product_summary,
    fetch_topic_summary,
    fit_and_evaluate_topic_model,
    generate_nlp_charts,
    infer_topics_for_unique_narratives,
    is_usable_narrative,
    load_model_artifacts,
    load_topic_results_to_db,
    sample_unique_narratives,
    save_model_artifacts,
    split_train_eval,
    validate_nlp_results,
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

                # 2. Retrieve required columns from arzi.complaints
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

    # 4. Model training or compatible artifact reuse
    reused_model = False
    vectorizer = None
    nmf_model = None
    eval_metrics: dict[str, Any] = {}

    if not args.retrain:
        try:
            vectorizer, nmf_model, loaded_meta = load_model_artifacts(
                nlp_dir, expected_dataset_count=EXPECTED_RECORD_COUNT
            )
            eval_metrics = loaded_meta.get("evaluation_metrics", {})
            reused_model = True
        except FileNotFoundError:
            # Artifacts do not exist yet, proceed to fit model
            reused_model = False
        except Exception as exc:
            print(
                f"Error: Existing model artifacts failed compatibility validation: {exc}\n"
                f"Use --retrain to intentionally rebuild the topic model.",
                file=sys.stderr,
            )
            return 1

    if not reused_model:
        # Step 4a: Deterministic product-aware sampling
        sample_texts, sample_by_prod = sample_unique_narratives(complaints)

        # Step 4b: Train / Held-out eval split (10%)
        train_texts, eval_texts = split_train_eval(sample_texts)

        # Step 4c: Fit evaluation model on train, evaluate on held-out, refit final model
        vectorizer, nmf_model, eval_metrics = fit_and_evaluate_topic_model(
            sample_texts, train_texts, eval_texts
        )
        eval_metrics["sample_counts_by_product"] = sample_by_prod

        # Save metadata and model artifacts
        metadata = {
            "dataset_record_count": EXPECTED_RECORD_COUNT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "vectorizer_config": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in VECTORIZER_PARAMS.items()
            },
            "model_config": NMF_PARAMS,
            "evaluation_metrics": eval_metrics,
        }
        save_model_artifacts(vectorizer, nmf_model, metadata, nlp_dir)

    # 5. Transform all unique usable narratives and map back to complaints
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

    # 6. Load topic catalog and assignments into PostgreSQL in a single transaction
    try:
        load_topic_results_to_db(topic_catalog, complaint_topics, schema_path)
    except Exception as exc:
        print(f"PostgreSQL topic storage failed: {exc}", file=sys.stderr)
        return 1

    # 7. Query SQL views
    try:
        topic_summary = fetch_topic_summary()
        topic_product_summary = fetch_topic_product_summary()
        topic_issue_detail = analyze_topic_issue_detail()
    except Exception as exc:
        print(f"Failed to query NLP views: {exc}", file=sys.stderr)
        return 1

    # 8. Validate all NLP results against quality checks
    try:
        validate_nlp_results(complaints, topic_catalog, complaint_topics, eval_metrics)
    except Exception as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    # 9. Generate charts
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

    # 10. Save local JSON summary
    summary_data = {
        "dataset_count": total_records,
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
            "nmf_converged": eval_metrics.get("nmf_converged"),
            "nmf_iterations": eval_metrics.get("nmf_iterations"),
        },
        "topic_diversity_measure": {
            "metric": "proportion_unique_top_terms",
            "score": eval_metrics.get("topic_diversity"),
            "total_terms_evaluated": N_TOPICS * TOP_TERMS_COUNT,
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

    summary_file.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    # 11. Print concise result summary
    mode_str = "Reused compatible local model" if reused_model else "Fitted and evaluated new model"
    print("============================================================")
    print("Arzi NLP Topic Discovery Summary")
    print("============================================================")
    print(f"Database Connection:        Connected ({db_host}:{db_port} / {db_name})")
    print(f"Execution Mode:             {mode_str}")
    print(f"Complaints Processed:       {total_records:,}")
    print(f"Unique Narratives:          {unique_raw_count:,} (repeated: {repeated_narrative_count:,})")
    print(f"Narratives Usable for NLP:  {usable_count:,} (unusable: {unusable_narrative_count:,})")
    print(f"Topic Assignment Coverage:  {assigned_count:,} assigned, {unassigned_count:,} unassigned")
    print(f"Model Vocabulary Size:      {eval_metrics.get('vocabulary_size', 0):,} terms")
    print(f"Held-out Reconstruction:    {eval_metrics.get('held_out_reconstruction_similarity')} cosine similarity")
    print(f"Topic Term Diversity:       {eval_metrics.get('topic_diversity')} ({N_TOPICS} topics)")
    print("Discovered Topics (Top 5 terms):")
    for t in topic_catalog:
        cnt = next((s["complaint_count"] for s in topic_summary if s["topic_id"] == t["topic_id"]), 0)
        pct = next((s["complaint_share_pct"] for s in topic_summary if s["topic_id"] == t["topic_id"]), 0.0)
        print(f"  - Topic {t['topic_id']} ({cnt:,} / {pct:.1f}%): {', '.join(t['top_terms'][:5])}")
    print("Generated Artifacts:")
    print(f"  - Charts: {chart_a.name}, {chart_b.name}")
    print(f"  - Summary JSON: {summary_file}")
    print("Overall Completion Status:  PASSED")
    print("============================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
