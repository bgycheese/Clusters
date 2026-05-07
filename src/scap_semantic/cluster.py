import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score

try:
    from .paths import (
        CLUSTER_RESULTS_FILE,
        CLUSTER_SUMMARY_FILE,
        CONDENSED_TREE_PLOT,
        LABELS_FILE,
        REDUCED_FILE,
        RULE_META_FILE,
        Y_TRUE_FILE,
        ensure_output_dirs,
    )
except ImportError:
    from paths import (
        CLUSTER_RESULTS_FILE,
        CLUSTER_SUMMARY_FILE,
        CONDENSED_TREE_PLOT,
        LABELS_FILE,
        REDUCED_FILE,
        RULE_META_FILE,
        Y_TRUE_FILE,
        ensure_output_dirs,
    )


def load_json(path):
    with path.open() as file:
        return json.load(file)


def write_json(path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, indent=2)


def run_clustering(data, min_cluster_size=18, min_samples=14):
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        approx_min_span_tree=False,
        gen_min_span_tree=True,
    )
    labels = clusterer.fit_predict(data)
    return clusterer, labels


def _silhouette_details(labels, data):
    mask = labels != -1
    labels_clean = labels[mask]
    unique_clusters = sorted(set(labels_clean))

    if len(unique_clusters) < 2 or mask.sum() < 2:
        return -1.0, np.array([]), labels_clean, unique_clusters

    data_clean = data[mask]
    sil_avg = silhouette_score(data_clean, labels_clean, metric="euclidean")
    sil_vals = silhouette_samples(data_clean, labels_clean)
    return float(sil_avg), sil_vals, labels_clean, unique_clusters


def cluster_evaluation(labels=None, data=None, y_true_path=Y_TRUE_FILE):
    if data is None:
        data = np.load(REDUCED_FILE)
    if labels is None:
        labels = np.load(LABELS_FILE)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(list(labels).count(-1))
    print(f"{n_clusters} clusters and {n_noise} noise points.")

    sil_avg, sil_vals, _, unique_clusters = _silhouette_details(labels, data)
    if sil_vals.size:
        lowest_score = int(np.argmin(sil_vals))
        print(lowest_score, sil_vals.shape)
    print(f"Silhouette Score: {sil_avg:.4f}")

    ari = None
    if y_true_path.exists():
        y_true = np.load(y_true_path)
        ari = float(adjusted_rand_score(y_true, labels))
        print(f"Adjusted Rand Score: {ari:.4f}")

    return {
        "method": "cysecbert_scan",
        "n_clusters": len(unique_clusters),
        "n_noise": n_noise,
        "silhoutte": sil_avg,
        "ari": ari,
    }


def build_cluster_results(meta, labels, silhouette_values):
    if len(meta) != len(labels):
        raise ValueError(f"metadata length {len(meta)} does not match labels length {len(labels)}")

    silhouette_by_index = np.zeros(len(labels), dtype=float)
    non_noise_mask = labels != -1
    if silhouette_values.size:
        silhouette_by_index[non_noise_mask] = silhouette_values

    results = []
    for index, rule in enumerate(meta):
        results.append(
            {
                "cluster": int(labels[index]),
                "id": rule["id"],
                "profile_count": len(rule["profiles"]),
                "title": rule["title"],
                "severity": rule["severity"],
                "profiles": rule.get("profiles", []),
                "silhouette score for cluster": float(silhouette_by_index[index]),
            }
        )

    results.sort(key=lambda item: (item["cluster"], item["profile_count"]))
    return results


def build_cluster_summary(results):
    clusters = defaultdict(list)
    for result in results:
        clusters[result["cluster"]].append(result)

    summary = []
    for cluster_id, members in sorted(clusters.items()):
        if cluster_id == -1:
            continue

        all_profiles = set()
        for member in members:
            all_profiles.update(member["profiles"])

        summary.append(
            {
                "cluster_id": cluster_id,
                "profile_count": len(all_profiles),
                "profiles_covered": sorted(all_profiles),
                "rules": [member["id"] for member in members],
            }
        )

    summary.sort(key=lambda item: item["profile_count"], reverse=True)
    return summary


def plot_condensed_tree(clusterer, output_path=CONDENSED_TREE_PLOT) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    clusterer.condensed_tree_.plot()
    plt.ylim(0, 15)
    plt.title("HDBSCAN Condensed Tree (Zoomed)")
    plt.ylabel("Lambda (density level)")
    plt.xlabel("Policies")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def cluster_stability(labels=None, data=None):
    import matplotlib.pyplot as plt

    if data is None:
        data = np.load(REDUCED_FILE)
    if labels is None:
        labels = np.load(LABELS_FILE)

    sil_avg, sil_vals, labels_clean, unique_clusters = _silhouette_details(labels, data)

    fig, ax = plt.subplots(figsize=(8, 14))
    y_lower = 10
    for cluster_id in unique_clusters:
        cluster_sil_vals = sil_vals[labels_clean == cluster_id]
        cluster_sil_vals.sort()
        cluster_size = cluster_sil_vals.shape[0]
        y_upper = y_lower + cluster_size

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_vals, alpha=0.7)
        ax.text(-0.04, y_lower + 0.3 * cluster_size, str(cluster_id))
        y_lower = y_upper + 10

    ax.axvline(x=sil_avg, color="red", linestyle="--")
    ax.set_title(f"HDBSCAN Silhouette - avg = {sil_avg:.4f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Rules grouped by cluster")
    ax.set_xlim((-0.2, 1))
    return fig, ax


def main() -> None:
    ensure_output_dirs()
    meta = load_json(RULE_META_FILE)

    print("Loading data")
    data = np.load(REDUCED_FILE)
    print(f"Number of rules in reduced file: {data.shape[0]}, number of dimensions: {data.shape[1]}")
    print("Running HDBSCAN clustering...")

    clusterer, labels = run_clustering(data)
    np.save(LABELS_FILE, labels)
    print(f"Saved {LABELS_FILE}")

    _, silhouette_values, _, _ = _silhouette_details(labels, data)
    cluster_evaluation(labels, data)

    results = build_cluster_results(meta, labels, silhouette_values)
    write_json(CLUSTER_RESULTS_FILE, results)

    summary = build_cluster_summary(results)
    write_json(CLUSTER_SUMMARY_FILE, summary)

    plot_condensed_tree(clusterer)


if __name__ == "__main__":
    main()
