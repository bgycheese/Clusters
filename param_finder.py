import numpy as np
import json
import hdbscan
from sklearn.metrics import silhouette_score, adjusted_rand_score
import itertools
import matplotlib.pyplot as plt

data = np.load("./output/clusters/reduced.npy")
rules = json.load(open("../OpenScap_Dataset_RHEL9/output/policies.json"))
meta  = json.load(open("./output/references/rule_meta.json"))

# ── Define the Ground truth by building y_true using second-level group (index -1) ──────────────────────────
id_to_group = {}
for r in rules:
    groups = r.get("groups", [])
    if len(groups) >= 2:
        id_to_group[r["id"]] = groups[-1]   # Lowest level — 152 unique groups
    elif len(groups) == 1:
        id_to_group[r["id"]] = groups[0]   # fallback to top level
    else:
        id_to_group[r["id"]] = "ungrouped"

unique_groups = list(set(id_to_group.values()))
group_to_int  = {g: i for i, g in enumerate(unique_groups)}

y_true = np.array([ group_to_int[id_to_group[m["id"]]] for m in meta])
np.set_printoptions(threshold=1467)
print(f"y_true built: {len(y_true)} labels, {len(unique_groups)} unique groups")

# ── Define parameter grid ─────────────────────────────────────────────────────
min_cluster_sizes = [i for i in range(10,30)]
min_samples_vals  = [i for i in range(5,20)]
results_grid   = []

for mcs, ms in (itertools.product(min_cluster_sizes, min_samples_vals)):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        metric="euclidean",
        approx_min_span_tree=True,
        gen_min_span_tree=True
    )
    labels = clusterer.fit_predict(data)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = list(labels).count(-1)

    mask = labels != -1

    if n_clusters < 2 or mask.sum() < 2:
        sil = -1.0
        ari = -1.0
    else:
        sil = silhouette_score(data[mask], labels[mask], metric="euclidean")
        ari = adjusted_rand_score(y_true, labels)
    results_grid.append({
        "min_cluster_size": mcs,
        "min_samples":      ms,
        "n_clusters":       n_clusters,
        "n_noise":          n_noise,
        "silhouette":       round(sil, 4),
        "ari":              round(ari, 4)
    })


results_grid.sort(key=lambda k: (k["ari"] + k["silhouette"]), reverse=True)

json.dump(results_grid, open("./output/references/param_search.json", "w"), indent=2)

for i in results_grid:
    print(f"mcs={i["min_cluster_size"]:>2}  ms={i["min_samples"]:>2}  →  clusters={i["n_clusters"]:>3}  noise={i["n_noise"]:>3}  sil={i["silhouette"]:.4f}  ari={i.get("ari"):.4f}")

np.save("./output/clusters/y_true.npy", y_true)