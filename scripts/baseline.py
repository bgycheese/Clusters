import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from rapidfuzz import fuzz, process
from sklearn.metrics import adjusted_rand_score, silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scap_semantic.cluster import cluster_evaluation
from paths import LABELS_FILE, REDUCED_FILE, RULE_META_FILE, Y_TRUE_FILE


def load_json(path):
    with path.open() as file:
        return json.load(file)


def compare_fuzzy_search(threshold=75):
    data = np.load(REDUCED_FILE)
    y_true = np.load(Y_TRUE_FILE)
    rules = load_json(RULE_META_FILE)
    titles = [rule["title"] for rule in rules]

    similarity_matrix = process.cdist(
        titles,
        titles,
        scorer=fuzz.token_set_ratio,
        score_cutoff=threshold,
        dtype=np.uint8,
    )
    # print(similarity_matrix.shape)

    graph = nx.Graph()
    graph.add_nodes_from(range(len(titles)))
    i_idx, j_idx = np.where(
        (similarity_matrix >= threshold)
        & (np.arange(len(titles))[:, None] < np.arange(len(titles)))
    )

    for source, target in zip(i_idx, j_idx):
        graph.add_edge(int(source), int(target))

    components = list(nx.connected_components(graph))
    print(f"Fuzzy clusters: {sum(1 for component in components if len(component) > 1)}")
    print(f"Singletons (noise): {sum(1 for component in components if len(component) == 1)}")

    labels = np.full(len(titles), fill_value=-99, dtype=int)
    for index, component in enumerate(components):
        if len(component) > 1:
            for element in component:
                labels[element] = index
        else:
            labels[next(iter(component))] = -1

    assert (labels != -99).all(), "some rules were not assigned a label"

    mask = labels != -1
    if mask.sum() > 1 and len(set(labels[mask])) > 1:
        silhouette = silhouette_score(data[mask], labels[mask], metric="euclidean")
    else:
        silhouette = -1.0
    ari = adjusted_rand_score(y_true, labels)

    n_clusters = sum(1 for component in components if len(component) > 1)
    n_noise = int(list(labels).count(-1))
    print(data.shape, y_true.shape)

    print(f"Fuzzy baseline - clusters: {n_clusters}, noise: {n_noise}")
    print(f"Silhouette: {silhouette:.4f}, ARI: {ari:.4f}")

    return {
        "method": "fuzzy_token_set_ratio",
        "threshold": threshold,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette": round(float(silhouette), 4),
        "ari": round(float(ari), 4),
    }


def main():
    print(compare_fuzzy_search())
    print(cluster_evaluation(np.load(LABELS_FILE), np.load(REDUCED_FILE)))


if __name__ == "__main__":
    main()