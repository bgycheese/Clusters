import itertools
import json

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import adjusted_rand_score, silhouette_score

try:
    from paths import PARAM_SEARCH_FILE, POLICIES_FILE, REDUCED_FILE, RULE_META_FILE, Y_TRUE_FILE, ensure_output_dirs, EMBEDDINGS_FILE
except ImportError:
    from paths import PARAM_SEARCH_FILE, POLICIES_FILE, REDUCED_FILE, RULE_META_FILE, Y_TRUE_FILE, ensure_output_dirs, EMBEDDINGS_FILE


def load_json(path):
    with path.open() as file:
        return json.load(file)


def build_ground_truth(rules, meta):
    id_to_group = {}
    for rule in rules:
        groups = rule.get("groups", [])
        if len(groups) >= 2:
            id_to_group[rule["id"]] = groups[-1]
        elif len(groups) == 1:
            id_to_group[rule["id"]] = groups[0]
        else:
            id_to_group[rule["id"]] = "ungrouped"

    unique_groups = sorted(set(id_to_group.values()))
    group_to_int = {group: index for index, group in enumerate(unique_groups)}
    y_true = np.array([group_to_int[id_to_group[rule["id"]]] for rule in meta])
    return y_true, unique_groups


def search_hdbscan_params(data, y_true, min_cluster_sizes=range(10, 30), min_samples_values=range(5, 20)):
    import hdbscan

    results = []
    for min_cluster_size, min_samples in itertools.product(min_cluster_sizes, min_samples_values):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            approx_min_span_tree=True,
            gen_min_span_tree=True
        )
        labels = clusterer.fit_predict(data)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(list(labels).count(-1))
        mask = labels != -1

        if n_clusters < 2 or mask.sum() < 2:
            silhouette = -1.0
            ari = -1.0
        else:
            silhouette = silhouette_score(data[mask], labels[mask], metric="euclidean")
            ari = adjusted_rand_score(y_true, labels)
            # score = calinski_harabasz_score(data, labels) TODO:add this metric perhaps and compare them with ARI and Sil_Score
        results.append(
            {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette": round(float(silhouette), 4),
                "ari": round(float(ari), 4),
            }
        )

    results.sort(key=lambda result: result["ari"] + result["silhouette"], reverse=True)
    return results


def main() -> None:
    ensure_output_dirs()
    data = np.load(EMBEDDINGS_FILE)
    # data = np.load(REDUCED_FILE)
    rules = load_json(POLICIES_FILE)
    meta = load_json(RULE_META_FILE)

    y_true, unique_groups = build_ground_truth(rules, meta)
    print(f"y_true built: {len(y_true)} labels, {len(unique_groups)} unique groups")
    dimensions = list(range(10,51))
    n_neighbors = list(range(15,31))
    max_ari = -1
    max_sil = -1
    for dimension, n_neighbor in itertools.product(dimensions, n_neighbors):
        import umap

        new_data = umap.UMAP(
            n_components=dimension,
            n_neighbors=n_neighbor,
            min_dist=0.0,
            random_state=42,
            metric="euclidean"
        ).fit_transform(data)
        print(f"{dimension, n_neighbor} values ", new_data.shape)
        results = search_hdbscan_params(new_data, y_true)
        for result in results:
            if result["silhouette"] and result["ari"] > 0.7:
                if result["silhouette"] > max_sil or result["ari"] > max_ari:
                    max_sil = result["silhouette"]
                    max_ari = result["ari"]
                    print(
                        f"mcs={result['min_cluster_size']:>2}  "
                        f"ms={result['min_samples']:>2}  "
                        f"clusters={result['n_clusters']:>3}  "
                        f"noise={result['n_noise']:>3}  "
                        f"sil={result['silhouette']:.4f}  " 
                        f"ari={result['ari']:.4f}"
                    )


    with PARAM_SEARCH_FILE.open("w") as file:
        json.dump(results, file, indent=2)

    for result in results:
        print(
            f"mcs={result['min_cluster_size']:>2}  "
            f"ms={result['min_samples']:>2}  "
            f"clusters={result['n_clusters']:>3}  "
            f"noise={result['n_noise']:>3}  "
            f"sil={result['silhouette']:.4f}  "
            f"ari={result['ari']:.4f}"
        )

    np.save(Y_TRUE_FILE, y_true)


if __name__ == "__main__":
    main()
