import numpy as np

try:
    from .paths import LABELS_FILE, UMAP_2D_COORDS_FILE, UMAP_CLUSTERS_PLOT, ensure_output_dirs
except ImportError:
    from paths import LABELS_FILE, UMAP_2D_COORDS_FILE, UMAP_CLUSTERS_PLOT, ensure_output_dirs


def main() -> None:
    import matplotlib.pyplot as plt

    ensure_output_dirs()
    coords = np.load(UMAP_2D_COORDS_FILE)
    labels = np.load(LABELS_FILE)

    print(f"Loaded {coords.shape[0]} points with {labels.shape[0]} cluster labels")

    unique_clusters = sorted(set(labels) - {-1})
    n_clusters = len(unique_clusters)

    cmap = plt.get_cmap("tab20", max(n_clusters, 1))
    cluster_to_color = {cluster_id: cmap(index) for index, cluster_id in enumerate(unique_clusters)}

    fig, ax = plt.subplots(figsize=(14, 10))

    noise_mask = labels == -1
    n_noise = int(noise_mask.sum())
    ax.scatter(
        coords[noise_mask, 0],
        coords[noise_mask, 1],
        c="red",
        s=200,
        alpha=1,
        label=f"Noise ({n_noise})",
    )

    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=cluster_to_color[cluster_id],
            s=20,
            alpha=0.8,
        )

    ax.set_title(f"HDBSCAN Clusters via UMAP 2D - {n_clusters} clusters, {n_noise} noise points")
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")

    fig.savefig(UMAP_CLUSTERS_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {UMAP_CLUSTERS_PLOT}")

if __name__ == "__main__":
    main()
