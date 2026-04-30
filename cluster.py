import numpy as np
import json
import hdbscan
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples

# import umap
# import matplotlib.cm as cm
# from sklearn.manifold import TSNE
# import seaborn as sns


REDUCED_FILE = "./output/clusters/reduced.npy"
RESULTS_OUT = "./output/cluster_results.json"
SUMMARY_OUT = "./output/cluster_summary.json"

meta = json.load(open("./output/references/rule_meta.json"))
rules = json.load(open("../OpenScap_Dataset_RHEL9/output/policies.json"))
# ── Step 1: Load embeddings and meta ──────────────────────────────────────────
print("Loading Data")
data = np.load(REDUCED_FILE)
print(f"Number of rules in reduced file: {data.shape[0]}, number of dimensions: {data.shape[1]}")
print("Running hdbscan clustering...")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=18,
    min_samples=14,
    metric="euclidean",
    approx_min_span_tree=False,
    gen_min_span_tree=True
)
# ---- Step 2 Fit HDBSCAN and persist labels for visualize.py
labels = clusterer.fit_predict(data)
np.save("./output/clusters/labels.npy", labels)
print("Saved labels.npy")
# ---- Step 3 Cluster Evaluation against Group metric
def cluster_evaluation():
    global unique_clusters, sil_avg, sil_vals, labels_clean
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"{n_clusters} clusters and {n_noise} of noise points.")
    mask = labels != -1
    X_clean = data[mask]
    labels_clean = labels[mask]
    sil_avg  = silhouette_score(X_clean, labels_clean, metric="euclidean")
    sil_vals = silhouette_samples(X_clean, labels_clean)
    lowest_score = list(sil_vals).index(min(sil_vals))
    print(lowest_score, sil_vals.shape)
    unique_clusters = sorted(set(labels_clean))
    n_clusters = len(unique_clusters)



    print(f"Silhouette Score: {sil_avg:.4f}")
    y_true = np.load("./output/clusters/y_true.npy")
    score_ira = adjusted_rand_score(y_true, labels)
    print(f"Adjusted Rand Score: {score_ira:.4f}")

    return {
        "method" : "cysecbert_scan",
        "n_clusters" : n_clusters,
        "n_noise" : n_noise,
        "silhoutte"  : sil_avg,
        "ari" : score_ira
    }

# ── Draw the silhouette plot ─────────────────────────────────────────────────
def cluster_stability():
    fig, ax = plt.subplots(figsize=(8, 14))

    y_lower = 10
    for cid in unique_clusters:
        ith_sil_vals = sil_vals[labels_clean == cid]  # ← scores for cluster cid
        ith_sil_vals.sort()  # ← sort low to high
        size_j = ith_sil_vals.shape[0]  # ← number of rules in cluster
        y_upper = y_lower + size_j  # ← reserve vertical space

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_sil_vals,
            alpha=0.7
        )

        ax.text(-0.04, y_lower + 0.3 * size_j, str(cid))
        y_lower = y_upper + 10  # ← gap before next cluster

    ax.axvline(x=sil_avg, color="red", linestyle="--")
    ax.set_title(f"HDBSCAN Silhouette — avg = {sil_avg:.4f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Rules grouped by cluster")
    ax.set_xlim((-0.2, 1))



# ── Step 5: Build cluster_results.json ────────────────────────────────────────
results = []
def produce_clean(max=1358):
    cnt = 0
    while cnt <= max:
        yield cnt
        cnt += 1

j = produce_clean()
for i, rule in enumerate(meta):
    results.append({
        "cluster": int(labels[i]),
        "id": rule["id"],
        "profile_count": len(rule["profiles"]),
        "title": rule["title"],
        "severity": rule["severity"],
        "profiles": rule.get("profiles", []),
        "silhouette score for cluster": 0 if int(labels[i]) == -1 else f"{sil_vals[next(j)]}",
    })

# add some user input on the state of data at this point
results.sort(key=lambda x: (x["cluster"], x["profile_count"]))
# ── Step 5: Build cluster_summary.json ────────────────────────────────────────
json.dump(results, open(RESULTS_OUT, "w"), indent=2)
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
#  Dendrogram Visualization
plt.figure(figsize=(12,8))
clusterer.condensed_tree_.plot()
plt.ylim(0, 15)   # zoom into lower density range
plt.title("HDBSCAN Condensed Tree (Zoomed)")
plt.ylabel("Lambda (density level)")
plt.xlabel("Policies")
plt.tight_layout()
plt.savefig("./output/visuals/condensed_tree.png")

cluster_stability()