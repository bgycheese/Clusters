import json
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
import csv
from numpy.linalg import norm
import pandas as pd
BERT_EMBEDDING = "./output/embeddings/embeddings.npy"
HDBSCAN_LABELS = "./output/clusters/labels.npy"
META = "./output/references/rule_meta.json"
SUPERSET_OUT     = "./output/superset.json"
TRACEABILITY_OUT = "./output/traceability.csv"

embeddings = np.load(BERT_EMBEDDING)
labels = np.load(HDBSCAN_LABELS)
meta = json.load(open(META))
superset = []
id_to_index = {rule["id"] : rule["position"] for rule in meta}
id_to_member = {rule["id"] : rule for rule in meta}

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-9)


def cannonization(rule_ids):
    member_embeddings = np.array([embeddings[id_to_index[rid]] for rid in rule_ids])
    centroid = member_embeddings.mean(axis=0)

    # 3. Find the rule whose embedding is closest to the centroid
    best_id = rule_ids[0]
    best_sim = -1
    for rid in rule_ids:
        emb = embeddings[id_to_index[rid]]
        sim = cosine_similarity(emb, centroid)
        if sim > best_sim:
            best_sim = sim
            best_id = rid

    return best_id, float(best_sim)
# traceability.csv
def super_set_out():
    id_to_policy = {m["id"]: m for m in meta}
    cluster_file = json.load(open("./output/cluster_summary.json"))
    for rank, cluster in enumerate(cluster_file, start=1):
        cluster_id = cluster["cluster_id"]
        rule_ids   = cluster["rules"]
        profile_list = cluster["profiles_covered"]
        canonical_id, score = cannonization(rule_ids)
        canonical_meta = id_to_policy[canonical_id]

        superset.append({
            "rank": rank,
            "canonical_id": canonical_id,
            "canonical_meta": canonical_meta,
            "cluster_id": cluster_id,
            "profile_count" : cluster["profile_count"],
            "profiles_covered" : profile_list,
            "canonical_name" : canonical_meta["title"],
            "canonical_severity" : canonical_meta["severity"],
            "centroid_similarity" : round(score, 4),
            "rules" : rule_ids
        })

    json.dump(superset, open(SUPERSET_OUT, "w"), indent=2)



def traceability():
    rows = []
    data = json.load(open(SUPERSET_OUT))
    for entry in data:
        original_embeddings = embeddings[id_to_index[entry["canonical_id"]]]
        for rule in entry["rules"]:
            rule_meta = id_to_member[rule]
            rule_embeddings = embeddings[id_to_index[rule]]
            sim = cosine_similarity(original_embeddings, rule_embeddings)

            rows.append({
                "rank" : entry["rank"],
                "cluster_id" : entry["cluster_id"],
                "cluster_profile_count" : entry["profile_count"],
                "cluster_profiles_covered" : "|".join(entry["profiles_covered"]),
                "canonical_rule_id" : entry["canonical_id"],
                "canonical_rule_name" : entry["canonical_name"],
                "member_id" : rule,
                "member_title" : rule_meta["title"],
                "member_severity": rule_meta["severity"],
                "member_profiles":   "|".join(rule_meta["profiles"]) if rule_meta["profiles"] != [] else np.nan,
                "profile_count_rule": len(rule_meta["profiles"]) if rule_meta["profiles"] != [] else 0,
                "sim_to_canonical" : round(sim,4),
                "is_canonical" : "yes" if rule ==  entry["canonical_id"] else "no"
            })
    noise = json.load(open("./output/cluster_results.json"))
    for rule in noise:
        if rule["cluster"] == -1:
            rows.append({
                "rank": "noise",
                "cluster_id": -1,
                "cluster_profile_count": rule["profile_count"],
                "cluster_profiles_covered": "|".join(rule["profiles"]),
                "canonical_rule_name":   rule["title"],
                "member_id":    rule["id"],
                "member_title":      rule["title"],
                "member_severity":   rule["severity"],
                "member_profiles":   "|".join(rule["profiles"]) if rule["profiles"] != [] else np.nan,
                "profile_count_rule" : len(rule["profiles"]) if rule["profiles"] != [] else 0,
                "sim_to_canonical":  1.0,
                "is_canonical":      "yes"
            })

    with open(TRACEABILITY_OUT, "w") as trace_file:
        traceability_write = csv.DictWriter(trace_file, fieldnames=rows[0].keys())
        traceability_write.writeheader()
        traceability_write.writerows(rows)


def terminal_output():
    df = pd.read_csv("./output/traceability.csv", iterator=False)
    # top_policies = (
    #     df.groupby(["cluster_id", "profile_count", ])["profile_count"]
    #     .nunique()
    #     .sort_values(ascending=True)
    #     .head(20)
    # )
    print(
        df[["rank", "cluster_id", "profile_count_rule", "member_title"]]
        .dropna()
        .sort_values("profile_count_rule", ascending=False)
        .head()
        .to_string()
    )

def main():
    super_set_out()
    traceability()
    terminal_output()
if __name__ == '__main__':
     main()