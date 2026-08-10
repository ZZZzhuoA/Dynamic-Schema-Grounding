import argparse
import copy
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value):
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def normalized_tokens(value):
    return set(re.findall(r"[a-z0-9_]+", normalize_text(value)))


def token_f1(left, right):
    left_tokens = normalized_tokens(left)
    right_tokens = normalized_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sequence_similarity(left, right):
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def original_sql(record):
    return record.get("answer") or record.get("rep_answer") or ""


def correction_key(record):
    return normalize_text(record.get("db_id")), normalize_text(record.get("question"))


def original_key(record):
    return normalize_text(record.get("db_name")), normalize_text(record.get("question"))


def candidate_features(correction, original):
    question_sequence = sequence_similarity(correction.get("question"), original.get("question"))
    question_token_f1 = token_f1(correction.get("question"), original.get("question"))
    evidence_similarity = sequence_similarity(correction.get("evidence"), original.get("evidence"))
    sql_token_f1 = token_f1(correction.get("SQL"), original_sql(original))
    score = (
        0.70 * question_sequence
        + 0.15 * question_token_f1
        + 0.05 * evidence_similarity
        + 0.10 * sql_token_f1
    )
    return {
        "score": score,
        "question_sequence": question_sequence,
        "question_token_f1": question_token_f1,
        "evidence_similarity": evidence_similarity,
        "sql_token_f1": sql_token_f1,
    }


def load_review_map(path: Path | None):
    if path is None:
        return {}
    value = read_json(path)
    if isinstance(value, list):
        pairs = {}
        for item in value:
            if item.get("original_index") is not None:
                pairs[int(item["correction_index"])] = int(item["original_index"])
        return pairs
    if isinstance(value, dict):
        return {int(key): int(index) for key, index in value.items()}
    raise ValueError("review map must be a JSON object or a list of mapping records")


def rank_candidates(correction, candidate_indices, originals):
    ranked = []
    for original_index in candidate_indices:
        features = candidate_features(correction, originals[original_index])
        ranked.append({"original_index": original_index, **features})
    return sorted(ranked, key=lambda item: (-item["score"], item["original_index"]))


def compact_candidates(ranked, originals, limit=5):
    output = []
    for item in ranked[:limit]:
        original = originals[item["original_index"]]
        output.append(
            {
                **item,
                "question": original.get("question"),
                "sql": original_sql(original),
            }
        )
    return output


def register_match(matches, used_original_indices, correction_index, original_index, method, details=None):
    if correction_index in matches:
        raise ValueError(f"correction index {correction_index} was matched more than once")
    if original_index in used_original_indices:
        raise ValueError(f"original index {original_index} was matched more than once")
    matches[correction_index] = {
        "original_index": original_index,
        "match_method": method,
        "match_details": details or {},
    }
    used_original_indices.add(original_index)


def match_corrections(
    originals,
    corrections,
    review_map,
    fuzzy_question_threshold,
    fuzzy_score_margin,
):
    original_by_key = defaultdict(list)
    correction_by_key = defaultdict(list)
    original_by_db = defaultdict(list)
    for index, record in enumerate(originals):
        original_by_key[original_key(record)].append(index)
        original_by_db[normalize_text(record.get("db_name"))].append(index)
    for index, record in enumerate(corrections):
        correction_by_key[correction_key(record)].append(index)

    matches = {}
    used_original_indices = set()
    unresolved = {}

    for correction_index, original_index in sorted(review_map.items()):
        if correction_index < 0 or correction_index >= len(corrections):
            raise ValueError(f"review correction index out of range: {correction_index}")
        if original_index < 0 or original_index >= len(originals):
            raise ValueError(f"review original index out of range: {original_index}")
        correction = corrections[correction_index]
        original = originals[original_index]
        if normalize_text(correction.get("db_id")) != normalize_text(original.get("db_name")):
            raise ValueError(
                f"review mapping crosses databases: correction {correction_index} -> original {original_index}"
            )
        register_match(
            matches,
            used_original_indices,
            correction_index,
            original_index,
            "manual_review",
            candidate_features(correction, original),
        )

    for key in sorted(correction_by_key):
        correction_indices = [index for index in correction_by_key[key] if index not in matches]
        original_indices = [index for index in original_by_key.get(key, []) if index not in used_original_indices]
        if not correction_indices or not original_indices:
            continue
        if len(correction_indices) == len(original_indices):
            method = "exact_unique" if len(correction_indices) == 1 else "exact_occurrence"
            for correction_index, original_index in zip(correction_indices, original_indices):
                register_match(
                    matches,
                    used_original_indices,
                    correction_index,
                    original_index,
                    method,
                    candidate_features(corrections[correction_index], originals[original_index]),
                )
            continue
        for correction_index in correction_indices:
            available = [index for index in original_indices if index not in used_original_indices]
            ranked = rank_candidates(corrections[correction_index], available, originals)
            if not ranked:
                continue
            best = ranked[0]
            second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
            exact_sql = normalize_text(corrections[correction_index].get("SQL")) == normalize_text(
                original_sql(originals[best["original_index"]])
            )
            if exact_sql or best["score"] - second_score >= fuzzy_score_margin:
                register_match(
                    matches,
                    used_original_indices,
                    correction_index,
                    best["original_index"],
                    "exact_disambiguated",
                    {**best, "score_margin": best["score"] - second_score, "exact_sql": exact_sql},
                )
            else:
                unresolved[correction_index] = {
                    "reason": "ambiguous_exact_question",
                    "candidates": compact_candidates(ranked, originals),
                }

    fuzzy_proposals = []
    for correction_index, correction in enumerate(corrections):
        if correction_index in matches or correction_index in unresolved:
            continue
        db_id = normalize_text(correction.get("db_id"))
        available = [index for index in original_by_db.get(db_id, []) if index not in used_original_indices]
        ranked = rank_candidates(correction, available, originals)
        if not ranked:
            unresolved[correction_index] = {"reason": "database_not_found", "candidates": []}
            continue
        best = ranked[0]
        second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
        proposal = {
            "correction_index": correction_index,
            "best": best,
            "score_margin": best["score"] - second_score,
            "ranked": ranked,
        }
        fuzzy_proposals.append(proposal)

    fuzzy_proposals.sort(key=lambda item: (-item["best"]["score"], item["correction_index"]))
    for proposal in fuzzy_proposals:
        correction_index = proposal["correction_index"]
        correction = corrections[correction_index]
        db_id = normalize_text(correction.get("db_id"))
        available = [index for index in original_by_db.get(db_id, []) if index not in used_original_indices]
        ranked = rank_candidates(correction, available, originals)
        if not ranked:
            unresolved[correction_index] = {"reason": "candidate_already_used", "candidates": []}
            continue
        best = ranked[0]
        second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
        score_margin = best["score"] - second_score
        if best["question_sequence"] < fuzzy_question_threshold:
            reason = "question_similarity_below_threshold"
        elif score_margin < fuzzy_score_margin:
            reason = "fuzzy_score_margin_below_threshold"
        else:
            reason = None
        if reason is None:
            register_match(
                matches,
                used_original_indices,
                correction_index,
                best["original_index"],
                "fuzzy_question",
                {**best, "score_margin": score_margin},
            )
        else:
            unresolved[correction_index] = {
                "reason": reason,
                "thresholds": {
                    "fuzzy_question_threshold": fuzzy_question_threshold,
                    "fuzzy_score_margin": fuzzy_score_margin,
                },
                "candidates": compact_candidates(ranked, originals),
            }

    return matches, unresolved


def merge_records(originals, corrections, matches, clear_changed_hit_info=True):
    merged = copy.deepcopy(originals)
    manifest = []
    stats = Counter()
    for correction_index, match in sorted(matches.items()):
        original_index = match["original_index"]
        correction = corrections[correction_index]
        before = copy.deepcopy(merged[original_index])
        after = merged[original_index]

        corrected_question = correction.get("question")
        corrected_evidence = correction.get("evidence")
        corrected_sql = correction.get("SQL")
        if not corrected_question or not corrected_sql:
            raise ValueError(f"correction {correction_index} is missing question or SQL")

        question_changed = normalize_text(before.get("question")) != normalize_text(corrected_question)
        evidence_changed = normalize_text(before.get("evidence")) != normalize_text(corrected_evidence)
        sql_changed = normalize_text(original_sql(before)) != normalize_text(corrected_sql)

        after["question"] = corrected_question
        after["evidence"] = corrected_evidence
        after["answer"] = corrected_sql
        after["rep_answer"] = corrected_sql
        hit_info_cleared = bool(sql_changed and clear_changed_hit_info and after.get("hit_info"))
        if sql_changed and clear_changed_hit_info:
            after["hit_info"] = {}

        stats["applied"] += 1
        stats["question_changed"] += int(question_changed)
        stats["evidence_changed"] += int(evidence_changed)
        stats["sql_changed"] += int(sql_changed)
        stats["sql_unchanged"] += int(not sql_changed)
        stats["hit_info_cleared"] += int(hit_info_cleared)
        stats[f"method:{match['match_method']}"] += 1

        manifest.append(
            {
                "correction_index": correction_index,
                "original_index": original_index,
                "db_id": correction.get("db_id"),
                "match_method": match["match_method"],
                "match_details": match["match_details"],
                "changes": {
                    "question": question_changed,
                    "evidence": evidence_changed,
                    "sql": sql_changed,
                    "hit_info_cleared": hit_info_cleared,
                },
                "original": {
                    "question": before.get("question"),
                    "evidence": before.get("evidence"),
                    "sql": original_sql(before),
                    "hit_info": before.get("hit_info") or {},
                },
                "corrected": {
                    "question": corrected_question,
                    "evidence": corrected_evidence,
                    "sql": corrected_sql,
                },
            }
        )
    return merged, manifest, stats


def build_unresolved_records(corrections, unresolved):
    output = []
    for correction_index, details in sorted(unresolved.items()):
        correction = corrections[correction_index]
        output.append(
            {
                "correction_index": correction_index,
                "db_id": correction.get("db_id"),
                "question": correction.get("question"),
                "evidence": correction.get("evidence"),
                "sql": correction.get("SQL"),
                **details,
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Safely overlay a partially corrected BIRD train subset onto train_question_answer.json."
    )
    parser.add_argument(
        "--train-question-answer",
        default="Data/BIRD/bird-schema/train_question_answer.json",
    )
    parser.add_argument(
        "--corrections",
        default="Data/BIRD/processed_final_data.json",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/stage0_train_correction_merge",
    )
    parser.add_argument("--review-map", default=None)
    parser.add_argument("--fuzzy-question-threshold", type=float, default=0.90)
    parser.add_argument("--fuzzy-score-margin", type=float, default=0.05)
    parser.add_argument(
        "--retain-changed-hit-info",
        action="store_true",
        help="Retain original hit_info even when corrected SQL differs. This is unsafe by default.",
    )
    args = parser.parse_args()

    train_path = Path(args.train_question_answer)
    correction_path = Path(args.corrections)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    originals = read_json(train_path)
    corrections = read_json(correction_path)
    if not isinstance(originals, list) or not isinstance(corrections, list):
        raise ValueError("training data and corrections must both be JSON lists")

    review_map = load_review_map(Path(args.review_map) if args.review_map else None)
    matches, unresolved = match_corrections(
        originals,
        corrections,
        review_map,
        args.fuzzy_question_threshold,
        args.fuzzy_score_margin,
    )
    merged, manifest, merge_stats = merge_records(
        originals,
        corrections,
        matches,
        clear_changed_hit_info=not args.retain_changed_hit_info,
    )
    unresolved_records = build_unresolved_records(corrections, unresolved)

    merged_path = output_dir / "merged_train_question_answer.json"
    manifest_path = output_dir / "correction_manifest.jsonl"
    unresolved_path = output_dir / "unresolved_corrections.jsonl"
    review_template_path = output_dir / "review_map_template.json"
    summary_path = output_dir / "summary.json"

    write_json(merged_path, merged)
    write_jsonl(manifest_path, manifest)
    write_jsonl(unresolved_path, unresolved_records)
    write_json(
        review_template_path,
        [
            {
                "correction_index": item["correction_index"],
                "original_index": None,
                "suggested_original_index": item.get("candidates", [{}])[0].get("original_index")
                if item.get("candidates")
                else None,
                "reason": item["reason"],
            }
            for item in unresolved_records
        ],
    )

    match_method_counts = Counter(item["match_method"] for item in manifest)
    unresolved_reason_counts = Counter(item["reason"] for item in unresolved_records)
    summary = {
        "config": {
            "train_question_answer": str(train_path),
            "corrections": str(correction_path),
            "output_dir": str(output_dir),
            "review_map": args.review_map,
            "fuzzy_question_threshold": args.fuzzy_question_threshold,
            "fuzzy_score_margin": args.fuzzy_score_margin,
            "clear_changed_hit_info": not args.retain_changed_hit_info,
        },
        "original_record_count": len(originals),
        "correction_record_count": len(corrections),
        "applied_correction_count": len(matches),
        "unresolved_correction_count": len(unresolved_records),
        "merged_record_count": len(merged),
        "match_method_counts": dict(match_method_counts),
        "unresolved_reason_counts": dict(unresolved_reason_counts),
        "change_counts": {
            key: value for key, value in merge_stats.items() if not key.startswith("method:")
        },
        "outputs": {
            "merged_train_question_answer": str(merged_path),
            "correction_manifest": str(manifest_path),
            "unresolved_corrections": str(unresolved_path),
            "review_map_template": str(review_template_path),
        },
        "safety_note": (
            "Changed SQL invalidates the original hit_info. It is cleared by default so Stage 1 derives "
            "schema supervision from corrected SQL rather than stale annotations."
        ),
    }
    write_json(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
