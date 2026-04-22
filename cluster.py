import numpy as np
import json
import hdbscan
from collections import defaultdict

REDUCED_FILE = "./output/clusters/reduced.npy"
RESULTS_OUT = "./output/cluster_results.json"
SUMMARY_OUT = "./output/cluster_summary.json"

# ── Step 1: Load embeddings and meta ──────────────────────────────────────────
print("Loading Data")
reduced_data = np.load(REDUCED_FILE)


print(f"Number of rules in reduced file: {reduced_data.shape[0]}, number of dimensions: {reduced_data.shape[1]}")

print("Running HDBSCAN clustering...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,
    min_samples=3,
    metric="euclidean"
)

labels = clusterer.fit_predict(reduced_data)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
n_non_noise = len(list(labels)) - n_noise
print(f"  Found {n_clusters} clusters of {n_non_noise} rules, {n_noise} noise points")

# ── Step 4: Build cluster_results.json ────────────────────────────────────────
meta = json.load(open("./output/references/rule_meta.json"))
results = []
for i, rule in enumerate(meta):
    results.append({
        "profile_count": len(rule["profiles"]),
        "id": rule["id"],
        "title": rule["rule"],
        "severity": rule["severity"],
        "profiles": rule.get("profiles", []),
        "cluster": int(labels[i]),
    })

# add some user input on the state of data at this point
results.sort(key=lambda x: (x["cluster"], -x["profile_count"]))
json.dump(results, open(RESULTS_OUT, "w"), indent=2)

# ── Step 5: Build cluster_summary.json ────────────────────────────────────────
clusters = defaultdict(list)
for r in results:
    clusters[r["cluster"]].append(r)

summary = []

for cluster_id, members in sorted(clusters.items()):
    if cluster_id == -1:
        continue

    all_profiles = set()
    for m in members:
        for p in m["profiles"]:
            all_profiles.add(p)

    summary.append({
        "cluster_id": cluster_id,
        "profile_count": len(all_profiles),
        "profiles_covered": sorted(all_profiles),
        "rules": [m["id"] for m in members],
    })

summary.sort(key=lambda x: x["profile_count"], reverse=True)
json.dump(summary, open(SUMMARY_OUT, "w"), indent=2)