"""Natural Language Processing (NLP) topic-discovery module for Arzi."""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from arzi.analysis import EXPECTED_RECORD_COUNT
from arzi.database import TARGET_DB, apply_schema, get_connection

CLEANING_VERSION = "v1.0"
RANDOM_SEED = 42
N_TOPICS = 10
TOP_TERMS_COUNT = 12
MAX_PER_PRODUCT_SAMPLE = 3000
HELD_OUT_RATIO = 0.10
BATCH_SIZE = 4000

TOKEN_PATTERN = r"(?u)\b[a-zA-Z]{2,}\b"
RE_TOKEN = re.compile(TOKEN_PATTERN)

VECTORIZER_PARAMS: dict[str, Any] = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "max_features": 12000,
    "min_df": 10,
    "max_df": 0.90,
    "sublinear_tf": True,
    "token_pattern": TOKEN_PATTERN,
}

NMF_PARAMS: dict[str, Any] = {
    "n_components": N_TOPICS,
    "init": "nndsvda",
    "random_state": RANDOM_SEED,
    "max_iter": 250,
}


def clean_narrative(text: str) -> str:
    """Normalize narrative text by stripping redactions while preserving meaningful financial words."""
    if not text:
        return ""
    # Remove dates like XX/XX/XXXX or XX/XX/XX
    t = re.sub(r"\b[xX0-9]{1,2}/[xX0-9]{1,2}/[xX0-9]{2,4}\b", " ", text)
    # Remove amount placeholders {$...}
    t = re.sub(r"\{\$[^}]*\}", " ", t)
    # Remove repeated/standalone X tokens (e.g. XXXX, XX)
    t = re.sub(r"\b[xX]+\b", " ", t)
    # Remove account/phone/SSN hyphenated redactions (e.g. XXX-XX-XXXX)
    t = re.sub(r"\b[xX0-9]+(?:-[xX0-9]+)+\b", " ", t)
    # Retain standard alphanumeric words and whitespace; strip punctuation
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    return " ".join(t.split())


def is_usable_narrative(cleaned_text: str) -> bool:
    """Check if normalized narrative contains at least one alphabetic token of length >= 2."""
    if not cleaned_text:
        return False
    return bool(RE_TOKEN.search(cleaned_text))


def compute_source_fingerprint(complaints: list[dict[str, Any]]) -> str:
    """Calculate deterministic SHA-256 fingerprint over source complaint records ordered by complaint_id."""
    hasher = hashlib.sha256()
    sorted_records = sorted(complaints, key=lambda r: int(r["complaint_id"]))

    for r in sorted_records:
        cid_str = str(r["complaint_id"])
        prod_str = str(r.get("product") or "")
        iss_str = str(r.get("issue") or "")
        dt_val = r.get("date_received")
        dt_str = dt_val.isoformat() if hasattr(dt_val, "isoformat") else str(dt_val or "")
        narr_str = str(r.get("consumer_complaint_narrative") or "")

        # Unambiguous length-delimited serialization preventing field boundary collisions
        record_bytes = (
            f"{len(cid_str)}:{cid_str}|"
            f"{len(prod_str.encode('utf-8'))}:{prod_str}|"
            f"{len(iss_str.encode('utf-8'))}:{iss_str}|"
            f"{len(dt_str)}:{dt_str}|"
            f"{len(narr_str.encode('utf-8'))}:{narr_str}\n"
        ).encode("utf-8")
        hasher.update(record_bytes)

    return hasher.hexdigest()


def sample_unique_narratives(
    complaints: list[dict[str, Any]],
    max_per_product: int = MAX_PER_PRODUCT_SAMPLE,
    random_seed: int = RANDOM_SEED,
) -> tuple[list[str], dict[str, int]]:
    """Perform deterministic product-aware sampling of unique usable narratives."""
    rng = random.Random(random_seed)
    products = sorted(set(r["product"] for r in complaints))

    prod_to_usable: dict[str, list[str]] = defaultdict(list)
    for r in complaints:
        cleaned = r.get("cleaned_narrative", "")
        if cleaned and is_usable_narrative(cleaned):
            prod_to_usable[r["product"]].append(cleaned)

    sample_texts: list[str] = []
    global_seen: set[str] = set()
    sample_counts_by_product: dict[str, int] = {}

    for prod in products:
        unique_candidates: list[str] = []
        seen_in_prod: set[str] = set()
        for text in prod_to_usable[prod]:
            if text not in seen_in_prod and text not in global_seen:
                seen_in_prod.add(text)
                unique_candidates.append(text)

        if len(unique_candidates) > max_per_product:
            rng.shuffle(unique_candidates)
            selected = unique_candidates[:max_per_product]
        else:
            selected = unique_candidates

        for text in selected:
            global_seen.add(text)
            sample_texts.append(text)
        sample_counts_by_product[prod] = len(selected)

    return sample_texts, sample_counts_by_product


def split_train_eval(
    sample_texts: list[str],
    held_out_ratio: float = HELD_OUT_RATIO,
    random_seed: int = RANDOM_SEED,
) -> tuple[list[str], list[str]]:
    """Partition unique sampled narratives into train and held-out evaluation sets without text leakage."""
    shuffled = list(sample_texts)
    rng = random.Random(random_seed)
    rng.shuffle(shuffled)

    n_eval = int(len(shuffled) * held_out_ratio)
    eval_texts = shuffled[:n_eval]
    train_texts = shuffled[n_eval:]
    return train_texts, eval_texts


def compute_reconstruction_similarity(
    X_true: sparse.csr_matrix,
    X_pred: np.ndarray,
) -> tuple[float, int, int]:
    """Compute mean row-wise cosine similarity across all held-out documents.

    If true norm or reconstructed norm is zero, similarity is 0.0.
    Returns (mean_similarity, total_held_out, zero_vector_count).
    """
    n_docs = X_true.shape[0]
    if n_docs == 0:
        return 0.0, 0, 0

    dot_products = np.asarray(X_true.multiply(X_pred).sum(axis=1)).ravel()
    true_norms = np.sqrt(np.asarray(X_true.power(2).sum(axis=1)).ravel())
    pred_norms = np.linalg.norm(X_pred, axis=1)

    denom = true_norms * pred_norms
    valid = denom > 0

    sims = np.zeros(n_docs, dtype=float)
    sims[valid] = dot_products[valid] / denom[valid]

    zero_vector_count = int(np.sum(~valid))
    mean_sim = float(np.mean(sims))

    return round(mean_sim, 4), n_docs, zero_vector_count


def compute_topic_diversity(topic_terms: list[list[str]]) -> float:
    """Compute the proportion of unique words across the top terms of all topics."""
    all_terms: list[str] = []
    for terms in topic_terms:
        all_terms.extend(terms)
    if not all_terms:
        return 0.0
    return round(len(set(all_terms)) / len(all_terms), 4)


def fit_and_evaluate_topic_model(
    sample_texts: list[str],
    train_texts: list[str],
    eval_texts: list[str],
) -> tuple[TfidfVectorizer, NMF, dict[str, Any]]:
    """Fit TF-IDF + NMF topic model, evaluate held-out reconstruction, and refit final model."""
    # 1. Evaluation model on train portion only
    vec_eval = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_train = vec_eval.fit_transform(train_texts)

    nmf_eval = NMF(**NMF_PARAMS)
    W_train = nmf_eval.fit_transform(X_train)

    # 2. Evaluate on held-out narratives (with zero-vector handling)
    X_eval = vec_eval.transform(eval_texts)
    W_eval = nmf_eval.transform(X_eval)
    X_eval_pred = W_eval @ nmf_eval.components_

    mean_recon_sim, total_eval_docs, zero_vector_count = compute_reconstruction_similarity(
        X_eval, X_eval_pred
    )

    # 3. Refit final vectorizer and NMF model on complete fitting sample
    vec_final = TfidfVectorizer(**VECTORIZER_PARAMS)
    X_full = vec_final.fit_transform(sample_texts)

    nmf_final = NMF(**NMF_PARAMS)
    W_full = nmf_final.fit_transform(X_full)

    feature_names = vec_final.get_feature_names_out()
    top_terms_by_topic: list[list[str]] = []
    for t_idx in range(N_TOPICS):
        top_indices = np.argsort(nmf_final.components_[t_idx])[::-1][:TOP_TERMS_COUNT]
        top_terms = [feature_names[i] for i in top_indices]
        top_terms_by_topic.append(top_terms)

    topic_div = compute_topic_diversity(top_terms_by_topic)

    evaluation_metrics = {
        "fit_sample_size": len(sample_texts),
        "train_size": len(train_texts),
        "held_out_eval_size": len(eval_texts),
        "total_held_out_documents": total_eval_docs,
        "held_out_zero_vector_count": zero_vector_count,
        "vocabulary_size": len(feature_names),
        "nmf_converged": bool(nmf_final.n_iter_ < NMF_PARAMS["max_iter"]),
        "nmf_iterations": int(nmf_final.n_iter_),
        "held_out_reconstruction_similarity": mean_recon_sim,
        "topic_diversity": topic_div,
        "top_terms_by_topic": top_terms_by_topic,
    }

    return vec_final, nmf_final, evaluation_metrics


def infer_topics_for_unique_narratives(
    unique_usable_texts: list[str],
    vectorizer: TfidfVectorizer,
    nmf_model: NMF,
    batch_size: int = BATCH_SIZE,
) -> dict[str, tuple[int | None, float | None, bool]]:
    """Transform unique usable narratives in batches to produce deterministic topic assignments."""
    assignments: dict[str, tuple[int | None, float | None, bool]] = {}

    for start_idx in range(0, len(unique_usable_texts), batch_size):
        batch = unique_usable_texts[start_idx : start_idx + batch_size]
        X_batch = vectorizer.transform(batch)
        W_batch = nmf_model.transform(X_batch)

        for i, text in enumerate(batch):
            weights = W_batch[i]
            total_weight = float(np.sum(weights))
            if total_weight > 0.0:
                best_topic = int(np.argmax(weights))
                strength = float(weights[best_topic] / total_weight)
                assignments[text] = (best_topic, round(strength, 4), True)
            else:
                assignments[text] = (None, None, False)

    return assignments


def assign_topics_to_complaints(
    complaints: list[dict[str, Any]],
    text_assignments: dict[str, tuple[int | None, float | None, bool]],
) -> list[dict[str, Any]]:
    """Map unique narrative topic assignments back to all individual original complaint records."""
    assignments: list[dict[str, Any]] = []

    for c in complaints:
        cid = int(c["complaint_id"])
        cleaned = c.get("cleaned_narrative", "")

        if cleaned and cleaned in text_assignments:
            tid, strn, usable = text_assignments[cleaned]
        else:
            tid, strn, usable = None, None, False

        assignments.append({
            "complaint_id": cid,
            "topic_id": tid,
            "topic_strength": strn,
            "usable_for_nlp": usable,
        })

    return assignments


def validate_in_memory_results(
    complaints: list[dict[str, Any]],
    topic_catalog: list[dict[str, Any]],
    complaint_topics: list[dict[str, Any]],
    eval_metrics: dict[str, Any],
) -> None:
    """Validate NLP dataset integrity and consistency in-memory before database modification."""
    errors: list[str] = []

    total_complaints = len(complaints)
    if total_complaints != EXPECTED_RECORD_COUNT:
        errors.append(f"Source complaints count mismatch: {total_complaints:,} != {EXPECTED_RECORD_COUNT:,}")

    distinct_source_ids = len(set(r["complaint_id"] for r in complaints))
    if distinct_source_ids != EXPECTED_RECORD_COUNT:
        errors.append(f"Distinct source ID mismatch: {distinct_source_ids:,} != {EXPECTED_RECORD_COUNT:,}")

    total_assignments = len(complaint_topics)
    if total_assignments != EXPECTED_RECORD_COUNT:
        errors.append(f"Assignment table row mismatch: {total_assignments:,} != {EXPECTED_RECORD_COUNT:,}")

    assignment_ids = set(r["complaint_id"] for r in complaint_topics)
    if len(assignment_ids) != EXPECTED_RECORD_COUNT:
        errors.append(f"Distinct assignment complaint ID mismatch: {len(assignment_ids):,} != {EXPECTED_RECORD_COUNT:,}")

    source_ids = set(r["complaint_id"] for r in complaints)
    orphans = assignment_ids - source_ids
    if orphans:
        errors.append(f"Found {len(orphans)} orphan complaint IDs in topic assignments")

    if len(topic_catalog) != N_TOPICS:
        errors.append(f"Expected {N_TOPICS} topic catalog entries, got {len(topic_catalog)}")

    assigned = [r for r in complaint_topics if r["topic_id"] is not None]
    unassigned = [r for r in complaint_topics if r["topic_id"] is None]

    if (len(assigned) + len(unassigned)) != total_complaints:
        errors.append(
            f"Assigned ({len(assigned):,}) + unassigned ({len(unassigned):,}) != total ({total_complaints:,})"
        )

    for r in assigned:
        tid = r["topic_id"]
        strn = r["topic_strength"]
        if not (0 <= tid < N_TOPICS):
            errors.append(f"Invalid topic ID {tid} for complaint {r['complaint_id']}")
        if strn is None or not (0.0 <= strn <= 1.0):
            errors.append(f"Invalid topic strength {strn} for complaint {r['complaint_id']}")
        if not r["usable_for_nlp"]:
            errors.append(f"Assigned complaint {r['complaint_id']} marked as not usable for NLP")

    for r in unassigned:
        if r["topic_strength"] is not None:
            errors.append(f"Unassigned complaint {r['complaint_id']} has non-null topic strength")
        if r["usable_for_nlp"]:
            errors.append(f"Unassigned complaint {r['complaint_id']} marked as usable for NLP")

    if not eval_metrics.get("nmf_converged"):
        errors.append("NMF model failed to converge within maximum iterations")

    topic_div = eval_metrics.get("topic_diversity", 0.0)
    if not (0.0 < topic_div <= 1.0):
        errors.append(f"Invalid topic diversity value: {topic_div}")

    topic_counts = Counter(r["topic_id"] for r in assigned)
    if assigned:
        for tid, cnt in topic_counts.items():
            share = (cnt / len(assigned)) * 100.0
            if share > 80.0:
                errors.append(f"Topic {tid} dominates assignments with {share:.1f}% (> 80%)")

    if errors:
        raise ValueError("In-memory NLP validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


# Alias for backward compatibility
validate_nlp_results = validate_in_memory_results


def run_in_transaction_sql_checks(
    cur: Any,
    expected_assigned_count: int,
    expected_unassigned_count: int,
    expected_topic_counts: dict[int, int],
) -> dict[str, Any]:
    """Execute required database validation queries inside the active transaction before commit."""
    # 1. Source complaints count and distinct IDs
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT complaint_id) FROM arzi.complaints;")
    src_cnt, src_dist = cur.fetchone()
    if src_cnt != EXPECTED_RECORD_COUNT or src_dist != EXPECTED_RECORD_COUNT:
        raise ValueError(f"Source complaints integrity mismatch: count={src_cnt}, distinct={src_dist}")

    # 2. NLP assignment records count and distinct IDs
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT complaint_id) FROM arzi.complaint_topics;")
    asgn_cnt, asgn_dist = cur.fetchone()
    if asgn_cnt != EXPECTED_RECORD_COUNT or asgn_dist != EXPECTED_RECORD_COUNT:
        raise ValueError(f"Assignment records integrity mismatch: count={asgn_cnt}, distinct={asgn_dist}")

    # 3. Check 1-to-1 matching and orphans
    cur.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE ct.complaint_id IS NULL) AS missing_in_topics,
            COUNT(*) FILTER (WHERE c.complaint_id IS NULL) AS orphan_in_topics
        FROM arzi.complaints c
        FULL OUTER JOIN arzi.complaint_topics ct ON c.complaint_id = ct.complaint_id;
        """
    )
    missing, orphans = cur.fetchone()
    if missing != 0:
        raise ValueError(f"Found {missing} source complaint IDs missing from topic assignments")
    if orphans != 0:
        raise ValueError(f"Found {orphans} orphan complaint IDs in topic assignments")

    # 4. Exactly 10 topic catalog rows
    cur.execute("SELECT COUNT(*) FROM arzi.topic_catalog;")
    cat_cnt = cur.fetchone()[0]
    if cat_cnt != N_TOPICS:
        raise ValueError(f"Expected {N_TOPICS} topic catalog rows, got {cat_cnt}")

    # 5. Assigned topic IDs valid (0-9) and reference catalog
    cur.execute(
        """
        SELECT COUNT(*)
        FROM arzi.complaint_topics ct
        WHERE ct.topic_id IS NOT NULL AND (
            ct.topic_id < 0 OR ct.topic_id >= 10 OR NOT EXISTS (
                SELECT 1 FROM arzi.topic_catalog tc WHERE tc.topic_id = ct.topic_id
            )
        );
        """
    )
    invalid_topics = cur.fetchone()[0]
    if invalid_topics != 0:
        raise ValueError(f"Found {invalid_topics} assignments with invalid topic IDs or missing catalog reference")

    # 6. Valid strengths and unassigned conditions
    cur.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE topic_id IS NOT NULL AND (topic_strength IS NULL OR topic_strength < 0.0 OR topic_strength > 1.0)),
            COUNT(*) FILTER (WHERE topic_id IS NULL AND (topic_strength IS NOT NULL OR usable_for_nlp = TRUE)),
            COUNT(*) FILTER (WHERE topic_id IS NOT NULL),
            COUNT(*) FILTER (WHERE topic_id IS NULL)
        FROM arzi.complaint_topics;
        """
    )
    bad_assigned_str, bad_unassigned, db_assigned, db_unassigned = cur.fetchone()
    if bad_assigned_str != 0:
        raise ValueError(f"Found {bad_assigned_str} assigned records with invalid topic strength")
    if bad_unassigned != 0:
        raise ValueError(f"Found {bad_unassigned} unassigned records with non-null strength or marked usable")
    if db_assigned != expected_assigned_count:
        raise ValueError(f"Assigned count mismatch: expected {expected_assigned_count}, got {db_assigned}")
    if db_unassigned != expected_unassigned_count:
        raise ValueError(f"Unassigned count mismatch: expected {expected_unassigned_count}, got {db_unassigned}")
    if (db_assigned + db_unassigned) != EXPECTED_RECORD_COUNT:
        raise ValueError(f"Total count mismatch: {db_assigned} + {db_unassigned} != {EXPECTED_RECORD_COUNT}")

    # 7. View counts reconcile to assigned count
    cur.execute("SELECT COALESCE(SUM(complaint_count), 0) FROM arzi.v_topic_summary;")
    v_topic_sum = cur.fetchone()[0]
    if v_topic_sum != db_assigned:
        raise ValueError(f"v_topic_summary sum ({v_topic_sum}) != assigned count ({db_assigned})")

    cur.execute("SELECT COALESCE(SUM(complaint_count), 0) FROM arzi.v_topic_product_summary;")
    v_prod_sum = cur.fetchone()[0]
    if v_prod_sum != db_assigned:
        raise ValueError(f"v_topic_product_summary sum ({v_prod_sum}) != assigned count ({db_assigned})")

    cur.execute("SELECT COALESCE(SUM(complaint_count), 0) FROM arzi.v_topic_issue_summary;")
    v_issue_sum = cur.fetchone()[0]
    if v_issue_sum != db_assigned:
        raise ValueError(f"v_topic_issue_summary sum ({v_issue_sum}) != assigned count ({db_assigned})")

    # 8. Compare database per-topic counts with in-memory counts
    cur.execute(
        """
        SELECT topic_id, COUNT(*)
        FROM arzi.complaint_topics
        WHERE topic_id IS NOT NULL
        GROUP BY topic_id
        ORDER BY topic_id ASC;
        """
    )
    db_topic_counts = dict(cur.fetchall())
    for tid in range(N_TOPICS):
        expected_cnt = expected_topic_counts.get(tid, 0)
        actual_cnt = db_topic_counts.get(tid, 0)
        if actual_cnt != expected_cnt:
            raise ValueError(f"Topic {tid} count mismatch: expected {expected_cnt}, got {actual_cnt}")

    return {
        "source_complaints": src_cnt,
        "assignment_records": asgn_cnt,
        "assigned_count": db_assigned,
        "unassigned_count": db_unassigned,
        "topic_catalog_rows": cat_cnt,
        "views_reconciled": True,
        "per_topic_counts_matched": True,
    }


def load_and_validate_topic_results_in_transaction(
    conn: Any,
    topic_catalog: list[dict[str, Any]],
    complaint_topics: list[dict[str, Any]],
    expected_topic_counts: dict[int, int],
    nlp_schema_path: Path | str,
) -> dict[str, Any]:
    """Replace topic catalog and assignments and validate SQL integrity inside a single transaction."""
    schema_path = Path(nlp_schema_path)

    expected_assigned = sum(1 for r in complaint_topics if r["topic_id"] is not None)
    expected_unassigned = len(complaint_topics) - expected_assigned

    try:
        with conn.cursor() as cur:
            # 1. Apply schema and view definitions safely
            apply_schema(conn, schema_path)

            # 2. Replace topic catalog and assignments within the transaction
            cur.execute("TRUNCATE TABLE arzi.complaint_topics;")
            cur.execute("DELETE FROM arzi.topic_catalog;")

            # Insert topic catalog
            for row in topic_catalog:
                cur.execute(
                    """
                    INSERT INTO arzi.topic_catalog (topic_id, top_terms)
                    VALUES (%s, %s);
                    """,
                    (row["topic_id"], row["top_terms"]),
                )

            # Bulk COPY complaint topics
            buffer = io.StringIO()
            for r in complaint_topics:
                tid_str = str(r["topic_id"]) if r["topic_id"] is not None else "\\N"
                strn_str = (
                    f"{r['topic_strength']:.4f}"
                    if r["topic_strength"] is not None
                    else "\\N"
                )
                usable_str = "t" if r["usable_for_nlp"] else "f"
                buffer.write(f"{r['complaint_id']}\t{tid_str}\t{strn_str}\t{usable_str}\n")

            buffer.seek(0)
            copy_query = (
                "COPY arzi.complaint_topics (complaint_id, topic_id, topic_strength, usable_for_nlp) "
                "FROM STDIN WITH (FORMAT text)"
            )
            with cur.copy(copy_query) as copy:
                while chunk := buffer.read(65536):
                    copy.write(chunk)

            # 3. Execute in-transaction SQL validation checks BEFORE commit
            db_metrics = run_in_transaction_sql_checks(
                cur,
                expected_assigned_count=expected_assigned,
                expected_unassigned_count=expected_unassigned,
                expected_topic_counts=expected_topic_counts,
            )

        # 4. Commit ONLY after every check succeeds
        conn.commit()
        return db_metrics
    except Exception:
        conn.rollback()
        raise


def load_topic_results_to_db(
    topic_catalog: list[dict[str, Any]],
    complaint_topics: list[dict[str, Any]],
    nlp_schema_path: Path | str,
) -> dict[str, Any]:
    """Execute transactional load and validation with a newly acquired database connection."""
    expected_counts = Counter(
        r["topic_id"] for r in complaint_topics if r["topic_id"] is not None
    )
    with get_connection(dbname=TARGET_DB) as conn:
        return load_and_validate_topic_results_in_transaction(
            conn, topic_catalog, complaint_topics, expected_counts, nlp_schema_path
        )


def fetch_topic_summary() -> list[dict[str, Any]]:
    """Retrieve topic summary from arzi.v_topic_summary with explicit ordering."""
    with get_connection(dbname=TARGET_DB) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT topic_id, complaint_count, complaint_share_pct
                FROM arzi.v_topic_summary
                ORDER BY topic_id ASC;
                """
            )
            rows = cur.fetchall()

    return [
        {
            "topic_id": int(r[0]),
            "complaint_count": int(r[1]),
            "complaint_share_pct": float(r[2]),
        }
        for r in rows
    ]


def fetch_topic_product_summary() -> list[dict[str, Any]]:
    """Retrieve within-product topic breakdown from arzi.v_topic_product_summary with explicit ordering."""
    with get_connection(dbname=TARGET_DB) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT topic_id, product, complaint_count, product_total_complaints, within_product_share_pct
                FROM arzi.v_topic_product_summary
                ORDER BY topic_id ASC, product ASC;
                """
            )
            rows = cur.fetchall()

    return [
        {
            "topic_id": int(r[0]),
            "product": r[1],
            "complaint_count": int(r[2]),
            "product_total_complaints": int(r[3]),
            "within_product_share_pct": float(r[4]) if r[4] is not None else 0.0,
        }
        for r in rows
    ]


def fetch_topic_issue_summary() -> list[dict[str, Any]]:
    """Retrieve topic issue breakdown from arzi.v_topic_issue_summary with explicit ordering."""
    with get_connection(dbname=TARGET_DB) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT topic_id, issue, complaint_count
                FROM arzi.v_topic_issue_summary
                ORDER BY topic_id ASC, complaint_count DESC, issue ASC;
                """
            )
            rows = cur.fetchall()

    return [
        {
            "topic_id": int(r[0]),
            "issue": r[1],
            "complaint_count": int(r[2]),
        }
        for r in rows
    ]


def analyze_topic_issue_detail(
    issue_topic_rows: list[tuple[str, int, int]] | None = None,
    issue_totals: list[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Examine topic variation within the five most common structured complaint issues."""
    if issue_topic_rows is None or issue_totals is None:
        with get_connection(dbname=TARGET_DB) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT issue, COUNT(*) as cnt
                    FROM arzi.complaints
                    GROUP BY issue
                    ORDER BY cnt DESC, issue ASC
                    LIMIT 5;
                    """
                )
                issue_totals = cur.fetchall()

                top_issue_names = [r[0] for r in issue_totals]

                cur.execute(
                    """
                    SELECT c.issue, ct.topic_id, COUNT(*) AS complaint_count
                    FROM arzi.complaints c
                    JOIN arzi.complaint_topics ct ON c.complaint_id = ct.complaint_id
                    WHERE c.issue = ANY(%s) AND ct.topic_id IS NOT NULL
                    GROUP BY c.issue, ct.topic_id
                    ORDER BY c.issue ASC, complaint_count DESC;
                    """,
                    (top_issue_names,),
                )
                issue_topic_rows = cur.fetchall()

    topics_by_issue: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for iss, tid, cnt in issue_topic_rows:
        topics_by_issue[iss].append((int(tid), int(cnt)))

    results: list[dict[str, Any]] = []
    for iss, total_cnt in issue_totals:
        tot = int(total_cnt)
        topics = topics_by_issue.get(iss, [])
        top3 = topics[:3]

        top3_formatted = [
            {
                "topic_id": tid,
                "complaint_count": c,
                "share_within_issue_pct": round((c * 100.0) / tot, 2) if tot > 0 else 0.0,
            }
            for tid, c in top3
        ]

        ten_pct_count = sum(
            1 for _, c in topics if (tot > 0 and (c * 100.0 / tot) >= 10.0)
        )

        results.append({
            "issue": iss,
            "total_complaints": tot,
            "top_three_topics": top3_formatted,
            "topics_with_at_least_10pct_share": ten_pct_count,
        })

    return results


def generate_nlp_charts(
    topic_summary_rows: list[dict[str, Any]],
    topic_product_rows: list[dict[str, Any]],
    topic_catalog: list[dict[str, Any]],
    output_dir: Path | str,
    unassigned_count: int = 0,
) -> tuple[Path, Path]:
    """Render topic distribution bar chart and product-topic heatmap without text clipping."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    terms_by_topic = {r["topic_id"]: r["top_terms"] for r in topic_catalog}

    # Chart A: topic_distribution.png
    chart_a_path = out_dir / "topic_distribution.png"
    sorted_topics = sorted(topic_summary_rows, key=lambda x: x["complaint_count"], reverse=True)

    topic_labels: list[str] = []
    counts: list[int] = []
    for r in sorted_topics:
        tid = r["topic_id"]
        cnt = r["complaint_count"]
        top_terms = terms_by_topic.get(tid, [])[:4]
        terms_str = ", ".join(top_terms)
        topic_labels.append(f"Topic {tid}\n({terms_str})")
        counts.append(cnt)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_pos = np.arange(len(counts))
    bars = ax.barh(y_pos, counts, color="#2b5c8f", edgecolor="black", alpha=0.85, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topic_labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Assigned Complaint Count", fontsize=10)
    title_text = "Learned NLP Topic Distribution Across Financial Complaints"
    if unassigned_count > 0:
        title_text += f"\n(Unassigned records: {unassigned_count:,})"
    ax.set_title(title_text, fontsize=12, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    max_c = max(counts) if counts else 1
    for bar, c in zip(bars, counts):
        pct = (c * 100.0) / sum(counts) if counts else 0.0
        ax.annotate(
            f" {c:,} ({pct:.1f}%)",
            xy=(c, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            fontsize=8.5,
        )

    ax.set_xlim(0, max_c * 1.18)
    fig.tight_layout()
    fig.savefig(chart_a_path, dpi=150)
    plt.close(fig)

    # Chart B: topic_product_heatmap.png
    chart_b_path = out_dir / "topic_product_heatmap.png"
    all_products = sorted(set(r["product"] for r in topic_product_rows))
    product_idx = {p: i for i, p in enumerate(all_products)}

    matrix = np.zeros((len(all_products), N_TOPICS), dtype=float)
    for r in topic_product_rows:
        p = r["product"]
        tid = r["topic_id"]
        share = r.get("within_product_share_pct", 0.0)
        if p in product_idx and 0 <= tid < N_TOPICS:
            matrix[product_idx[p], tid] = share

    fig, ax = plt.subplots(figsize=(11, 7.5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=max(np.max(matrix), 50.0))

    ax.set_xticks(np.arange(N_TOPICS))
    ax.set_xticklabels([f"T{i}" for i in range(N_TOPICS)], fontsize=9)
    ax.set_xlabel("Learned Topic ID", fontsize=10, labelpad=8)

    short_prods = [
        re.sub(r",?\s+", " ", p).replace("Credit reporting, credit repair services, or other personal consumer reports", "Credit reporting & repair")
        .replace("Money transfer, virtual currency, or money service", "Money transfer & currency")
        .replace("Payday loan, title loan, or personal loan", "Payday & personal loan")
        for p in all_products
    ]
    ax.set_yticks(np.arange(len(all_products)))
    ax.set_yticklabels(short_prods, fontsize=8.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Within-Product Complaint Share (%)", fontsize=9.5)

    for i in range(len(all_products)):
        for j in range(N_TOPICS):
            val = matrix[i, j]
            text_color = "white" if val > 30.0 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=text_color, fontsize=7.5)

    ax.set_title(
        "Within-Product Financial Complaint Topic Distribution Heatmap\n(Cell percentage represents topic share within product complaints)",
        fontsize=11.5,
        pad=14,
    )
    fig.tight_layout()
    fig.savefig(chart_b_path, dpi=150)
    plt.close(fig)

    return chart_a_path, chart_b_path


def verify_model_metadata(
    metadata: dict[str, Any],
    expected_fingerprint: str,
    expected_dataset_count: int = EXPECTED_RECORD_COUNT,
) -> None:
    """Verify that saved model metadata strictly matches the source dataset and model configuration."""
    if not isinstance(metadata, dict):
        raise ValueError("Model metadata must be a dictionary")

    # 1. Source fingerprint
    actual_fp = metadata.get("source_data_fingerprint")
    if actual_fp != expected_fingerprint:
        raise ValueError(
            f"Source-data fingerprint mismatch: expected {expected_fingerprint}, got {actual_fp}. "
            f"Use --retrain to rebuild model for current data."
        )

    # 2. Dataset record count
    actual_cnt = metadata.get("dataset_record_count")
    if actual_cnt != expected_dataset_count:
        raise ValueError(
            f"Dataset record count mismatch: expected {expected_dataset_count:,}, got {actual_cnt}. "
            f"Use --retrain to rebuild model."
        )

    # 3. Cleaning version
    actual_clean = metadata.get("cleaning_version")
    if actual_clean != CLEANING_VERSION:
        raise ValueError(
            f"Cleaning version mismatch: expected {CLEANING_VERSION}, got {actual_clean}. "
            f"Use --retrain to rebuild model."
        )

    # 4. Sampling config
    expected_sampling = {
        "random_seed": RANDOM_SEED,
        "max_per_product": MAX_PER_PRODUCT_SAMPLE,
    }
    actual_sampling = metadata.get("sampling_config")
    if actual_sampling != expected_sampling:
        raise ValueError(
            f"Sampling configuration mismatch: expected {expected_sampling}, got {actual_sampling}. "
            f"Use --retrain to rebuild model."
        )

    # 5. Train/eval split config
    expected_split = {
        "held_out_ratio": HELD_OUT_RATIO,
        "random_seed": RANDOM_SEED,
    }
    actual_split = metadata.get("train_eval_split_config")
    if actual_split != expected_split:
        raise ValueError(
            f"Train/eval split configuration mismatch: expected {expected_split}, got {actual_split}. "
            f"Use --retrain to rebuild model."
        )

    # 6. Vectorizer config
    expected_vec = {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in VECTORIZER_PARAMS.items()
    }
    actual_vec = metadata.get("vectorizer_config")
    if actual_vec != expected_vec:
        raise ValueError(
            f"Vectorizer configuration mismatch: expected {expected_vec}, got {actual_vec}. "
            f"Use --retrain to rebuild model."
        )

    # 7. Model config
    expected_model = dict(NMF_PARAMS)
    actual_model = metadata.get("model_config")
    if actual_model != expected_model:
        raise ValueError(
            f"NMF model configuration mismatch: expected {expected_model}, got {actual_model}. "
            f"Use --retrain to rebuild model."
        )

    # 8. Topics count & top terms count
    if metadata.get("topics_count") != N_TOPICS:
        raise ValueError(
            f"Topics count mismatch: expected {N_TOPICS}, got {metadata.get('topics_count')}. "
            f"Use --retrain to rebuild model."
        )

    if metadata.get("top_terms_count") != TOP_TERMS_COUNT:
        raise ValueError(
            f"Top terms count mismatch: expected {TOP_TERMS_COUNT}, got {metadata.get('top_terms_count')}. "
            f"Use --retrain to rebuild model."
        )


def save_model_artifacts(
    vectorizer: TfidfVectorizer,
    nmf_model: NMF,
    metadata: dict[str, Any],
    output_dir: Path | str,
) -> None:
    """Save trained vectorizer, NMF model, and configuration metadata locally."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, out_dir / "vectorizer.joblib")
    joblib.dump(nmf_model, out_dir / "topic_model.joblib")

    meta_path = out_dir / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_model_artifacts(
    artifacts_dir: Path | str,
    expected_fingerprint: str,
    expected_dataset_count: int = EXPECTED_RECORD_COUNT,
) -> tuple[TfidfVectorizer, NMF, dict[str, Any]]:
    """Load local trained artifacts and verify model configuration compatibility."""
    art_dir = Path(artifacts_dir)
    vec_path = art_dir / "vectorizer.joblib"
    model_path = art_dir / "topic_model.joblib"
    meta_path = art_dir / "model_metadata.json"

    if not (vec_path.is_file() and model_path.is_file() and meta_path.is_file()):
        raise FileNotFoundError("Missing one or more required NLP model artifact files")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    # Verify comprehensive metadata compatibility
    verify_model_metadata(metadata, expected_fingerprint, expected_dataset_count)

    vectorizer: TfidfVectorizer = joblib.load(vec_path)
    nmf_model: NMF = joblib.load(model_path)

    return vectorizer, nmf_model, metadata
