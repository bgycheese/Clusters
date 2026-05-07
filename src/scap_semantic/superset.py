import csv
import json

import numpy as np
from numpy.linalg import norm

try:
    from .paths import (
        CLUSTER_RESULTS_FILE,
        CLUSTER_SUMMARY_FILE,
        EMBEDDINGS_FILE,
        RULE_META_FILE,
        SUPERSET_FILE,
        TRACEABILITY_FILE,
        ensure_output_dirs,
    )
except ImportError:
    from paths import (
        CLUSTER_RESULTS_FILE,
        CLUSTER_SUMMARY_FILE,
        EMBEDDINGS_FILE,
        RULE_META_FILE,
        SUPERSET_FILE,
        TRACEABILITY_FILE,
        ensure_output_dirs,
    )


TRACEABILITY_FIELDS = [
    "rank",
    "cluster_id",
    "cluster_profile_count",
    "cluster_profiles_covered",
    "canonical_rule_id",
    "canonical_rule_title",
    "rule_id",
    "rule_title",
    "rule_severity",
    "rule_profiles",
    "rule_profile_count",
    "sim_to_canonical",
    "is_canonical",
]


def load_json(path):
    with path.open() as file:
        return json.load(file)


def write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, indent=2)


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-9)


def canonicalization(rule_ids, embeddings=None, id_to_index=None):
    if embeddings is None or id_to_index is None:
        meta = load_json(RULE_META_FILE)
        embeddings = np.load(EMBEDDINGS_FILE)
        id_to_index = {rule["id"]: rule["position"] for rule in meta}

    member_embeddings = np.array([embeddings[id_to_index[rule_id]] for rule_id in rule_ids])
    centroid = member_embeddings.mean(axis=0)

    best_id = rule_ids[0]
    best_sim = -1.0
    for rule_id in rule_ids:
        embedding = embeddings[id_to_index[rule_id]]
        similarity = cosine_similarity(embedding, centroid)
        if similarity > best_sim:
            best_sim = similarity
            best_id = rule_id

    return best_id, float(best_sim)


def build_superset(embeddings, meta, cluster_summary):
    id_to_policy = {rule["id"]: rule for rule in meta}
    id_to_index = {rule["id"]: rule["position"] for rule in meta}

    superset = []
    for rank, cluster in enumerate(cluster_summary, start=1):
        rule_ids = cluster["rules"]
        canonical_id, score = canonicalization(rule_ids, embeddings, id_to_index)
        canonical_meta = id_to_policy[canonical_id]

        superset.append(
            {
                "rank": rank,
                "canonical_id": canonical_id,
                "canonical_meta": canonical_meta,
                "cluster_id": cluster["cluster_id"],
                "profile_count": cluster["profile_count"],
                "profiles_covered": cluster["profiles_covered"],
                "canonical_name": canonical_meta["title"],
                "canonical_severity": canonical_meta["severity"],
                "centroid_similarity": round(score, 4),
                "rules": rule_ids,
            }
        )

    return superset


def super_set_out():
    ensure_output_dirs()
    embeddings = np.load(EMBEDDINGS_FILE)
    meta = load_json(RULE_META_FILE)
    cluster_summary = load_json(CLUSTER_SUMMARY_FILE)

    superset = build_superset(embeddings, meta, cluster_summary)
    write_json(SUPERSET_FILE, superset)
    return superset


def build_traceability_rows(superset_data, cluster_results, embeddings, meta):
    id_to_index = {rule["id"]: rule["position"] for rule in meta}
    id_to_member = {rule["id"]: rule for rule in meta}

    rows = []
    for entry in superset_data:
        canonical_embedding = embeddings[id_to_index[entry["canonical_id"]]]
        for rule_id in entry["rules"]:
            rule_meta = id_to_member[rule_id]
            rule_embedding = embeddings[id_to_index[rule_id]]
            similarity = cosine_similarity(canonical_embedding, rule_embedding)

            rows.append(
                {
                    "rank": entry["rank"],
                    "cluster_id": entry["cluster_id"],
                    "cluster_profile_count": entry["profile_count"],
                    "cluster_profiles_covered": "|".join(entry["profiles_covered"]),
                    "canonical_rule_id": entry["canonical_id"],
                    "canonical_rule_title": entry["canonical_name"],
                    "rule_id": rule_id,
                    "rule_title": rule_meta["title"],
                    "rule_severity": rule_meta["severity"],
                    "rule_profiles": "|".join(rule_meta["profiles"]),
                    "rule_profile_count": len(rule_meta["profiles"]),
                    "sim_to_canonical": round(float(similarity), 4),
                    "is_canonical": "yes" if rule_id == entry["canonical_id"] else "no",
                }
            )

    for rule in cluster_results:
        if rule["cluster"] != -1:
            continue

        rows.append(
            {
                "rank": "noise",
                "cluster_id": -1,
                "cluster_profile_count": rule["profile_count"],
                "cluster_profiles_covered": "|".join(rule["profiles"]),
                "canonical_rule_id": rule["id"],
                "canonical_rule_title": rule["title"],
                "rule_id": rule["id"],
                "rule_title": rule["title"],
                "rule_severity": rule["severity"],
                "rule_profiles": "|".join(rule["profiles"]),
                "rule_profile_count": len(rule["profiles"]),
                "sim_to_canonical": 1.0,
                "is_canonical": "yes",
            }
        )

    return rows


def traceability():
    embeddings = np.load(EMBEDDINGS_FILE)
    meta = load_json(RULE_META_FILE)
    superset_data = load_json(SUPERSET_FILE)
    cluster_results = load_json(CLUSTER_RESULTS_FILE)

    rows = build_traceability_rows(superset_data, cluster_results, embeddings, meta)
    with TRACEABILITY_FILE.open("w", newline="") as trace_file:
        writer = csv.DictWriter(trace_file, fieldnames=TRACEABILITY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows

def main() -> None:
    super_set_out()
    traceability()


if __name__ == "__main__":
    main()
