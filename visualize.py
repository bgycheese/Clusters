import numpy as np
import matplotlib.pyplot as plt

# ── Load the 2D UMAP coordinates and the HDBSCAN cluster labels ──────────────
coords = np.load("./output/clusters/umap_2d_coords.npy")   # shape: (1467, 2)
labels = np.load("./output/clusters/labels.npy")          # shape: (1467,)

print(f"Loaded {coords.shape[0]} points with {labels.shape[0]} cluster labels")

# ── Build one color per cluster (excluding noise) ────────────────────────────
unique_clusters = sorted(set(labels) - {-1})
n_clusters = len(unique_clusters)

cmap = plt.get_cmap("tab20", max(n_clusters, 1))
cluster_to_color = {cid: cmap(i) for i, cid in enumerate(unique_clusters)}

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10))

# noise first so colored cluster dots sit on top of grey
noise_mask = labels == -1
n_noise = int(noise_mask.sum())
ax.scatter(
    coords[noise_mask, 0],
    coords[noise_mask, 1],
    c="red",
    s=200,
    alpha=1,
    label=f"Noise ({n_noise})"
)

# each cluster on top in its own color
for cid in unique_clusters:
    mask = labels == cid
    ax.scatter(
        coords[mask, 0],
        coords[mask, 1],
        color=cluster_to_color[cid],
        s=20,
        alpha=0.8
    )

ax.set_title(f"HDBSCAN Clusters via UMAP 2D — {n_clusters} clusters, {n_noise} noise points")
ax.set_xlabel("UMAP dimension 1")
ax.set_ylabel("UMAP dimension 2")

fig.savefig("./output/visuals/umap_clusters.png", dpi=150, bbox_inches="tight")
print("Saved umap_clusters.png")
plt.show()
