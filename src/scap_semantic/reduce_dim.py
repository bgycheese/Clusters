import json

import numpy as np

try:
    from paths import (
        EMBEDDINGS_FILE,
        REDUCED_FILE,
        RULE_META_FILE,
        UMAP_2D_COORDS_FILE,
        UMAP_PLOT_2D,
        UMAP_PLOT_3D,
        ensure_output_dirs,
    )
except ImportError:
    from paths import (
        EMBEDDINGS_FILE,
        REDUCED_FILE,
        RULE_META_FILE,
        UMAP_2D_COORDS_FILE,
        UMAP_PLOT_2D,
        UMAP_PLOT_3D,
        ensure_output_dirs,
    )


def load_json(path):
    with path.open() as file:
        return json.load(file)


def fit_umap(data, n_components, n_neighbors, metric, min_dist=0.0, n_epochs=500):
    import umap

    return umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_epochs=n_epochs,
        random_state=42,
        metric=metric,
    ).fit_transform(data)


def severity_colors(meta):
    color = [i for i, j in enumerate(meta)]
    return list(color)


def save_visualizations(embeddings, meta, n_neighbors) -> None:
    import matplotlib.pyplot as plt

    colors = severity_colors(meta)

    visualized_3d = fit_umap(
        embeddings,
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
    )
    print(f"UMAP 3D done. New shape: {visualized_3d.shape}")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(visualized_3d[:, 0], visualized_3d[:, 1], visualized_3d[:, 2], c=colors, s=100)
    plt.title("SCAP Rules - UMAP 3D")
    fig.savefig(UMAP_PLOT_3D, dpi=150)
    plt.close(fig)

    visualized_2d = fit_umap(
        embeddings,
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
    )
    print(f"UMAP 2D done. New shape: {visualized_2d.shape}")
    np.save(UMAP_2D_COORDS_FILE, visualized_2d)

    fig2 = plt.figure(figsize=(12, 8))
    plt.scatter(visualized_2d[:, 0], visualized_2d[:, 1], c=colors, s=20, alpha=0.7)
    plt.title("SCAP Rules - UMAP 2D")
    fig2.savefig(UMAP_PLOT_2D, dpi=150)
    plt.close(fig2)


def main() -> None:
    ensure_output_dirs()
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"The number of rules {embeddings.shape[0]} and dimensions per rule {embeddings.shape[1]}")

    n_neighbors = max(2, int(0.01 * embeddings.shape[0]))

    reduced = fit_umap(
        embeddings,
        n_components=10,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
    )
    print(f"UMAP {reduced.shape[1]}D done. New shape: {reduced.shape}")
    # TODO: See if the seconfd reduction makes sense
    # reduced_10 = fit_umap(
    #     reduced_100,
    #     n_components=10,
    #     n_neighbors=15,
    #     min_dist=0.0,
    #     metric="euclidean"
    # )
    # print(f"UMAP 10D done. New shape: {reduced_10.shape}")
    np.save(REDUCED_FILE, reduced)
    print(f"Saved {REDUCED_FILE}")

    meta = load_json(RULE_META_FILE)
    save_visualizations(embeddings, meta, n_neighbors)


if __name__ == "__main__":
    main()
