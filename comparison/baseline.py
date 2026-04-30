import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
from rapidfuzz import fuzz, process
import networkx as nx
from .cluster import cluster_evaluation

def compare_fuzzy_search():
    X = np.load("../output/clusters/reduced.npy")
    y_true = np.load("../output/clusters/y_true.npy")
    rules = json.load(open("../output/references/rule_meta.json"))
    titles = [r["title"] for r in rules]
    threshold = 75
    similarity_matrix = process.cdist(titles,titles,
                                      scorer=fuzz.token_set_ratio,
                                      score_cutoff=threshold,
                                      dtype=np.uint8)
    print(similarity_matrix.shape)  # (1467, 1467)
    # Figure out what is happening here
    G = nx.Graph()
    G.add_nodes_from(range(len(titles)))  # one node per rule
    # numpy trick: get coordinates of all cells above threshold
    i_idx, j_idx = np.where((similarity_matrix >= threshold) & (np.arange(len(titles))[:, None] < np.arange(len(titles))))
    # the second condition (i < j) avoids adding both (i,j) and (j,i) — the matrix is symmetric

    for i, j in zip(i_idx, j_idx):
        G.add_edge(int(i), int(j))

    # extract clusters
    components = list(nx.connected_components(G))
    # nx.draw_spring(G, with_labels=True)

    print(f"Fuzzy clusters: {sum(1 for c in components if len(c) > 1)}")
    print(f"Singletons (noise): {sum(1 for c in components if len(c) == 1)}")

    labels = np.full(1467, fill_value=-99, dtype=int)

    for index , c in enumerate(components):
        if len(c) > 1:
            for elem in c:
                labels[elem] = index
        else:
            labels[c.pop()] = -1

    # np.set_printoptions(threshold=1467)
    # print(labels)
    assert (labels != -99).all(), "some rules were not assigned a label"

    mask = labels != -1
    sil = silhouette_score(X[mask], labels[mask], metric="euclidean") if mask.sum() > 1 and len(set(labels[mask])) > 1 else -1.0
    ari = adjusted_rand_score(y_true, labels)

    n_clusters = sum(1 for c in components if len(c) > 1)
    n_noise    = list(labels).count(-1)
    print(X.shape, y_true.shape)

    print(f"Fuzzy baseline — clusters: {n_clusters}, noise: {n_noise}")
    print(f"Silhouette: {sil:.4f}, ARI: {ari:.4f}")

    return {
        "method":     "fuzzy_token_set_ratio",
        "threshold":  threshold,
        "n_clusters": n_clusters,
        "n_noise":    n_noise,
        "silhouette": round(sil, 4),
        "ari":        round(ari, 4)
    }

def main():
    print(compare_fuzzy_search())
    print(cluster_evaluation())


if __name__ == '__main__':
    main()
