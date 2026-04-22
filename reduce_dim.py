import numpy as np
import json
import umap
import matplotlib.pyplot as plt
import os

# ── Load embeddings and metadata
embeddings = np.load("./output/embeddings/embeddings.npy")       # shape: (1467, 768)

print(f"The number of rules {embeddings.shape[0]} and the number of dimensions per rule {embeddings.shape[1]}")

# ── n_neighbours value set to 1% of the dataset row size  ───────────────────────────────────────────────────────────
n_neighbors = int( 0.01 * embeddings.shape[0] )
# ── Initiate the model hyperparams ───────────────────────────────────────────────────────────
reducer = umap.UMAP(
    n_components=100,
    n_neighbors=n_neighbors,
    min_dist=0.0,
    n_epochs=500,
    random_state=42,
    metric="cosine"
)
# ── Reduce ───────────────────────────────────────────────────────────
reduced_d_100 = reducer.fit_transform(embeddings)   # shape: (1467, 100)
print(f"UMAP done. New shape: {reduced_d_100.shape}")
reduced_d_10 = umap.UMAP(
    n_components=10,
    n_neighbors=n_neighbors,
    min_dist=0.0,
    n_epochs=500,
    random_state=42,
    metric="euclidean").fit_transform(reduced_d_100)   # shape: (1467, 10)

# ── Report the shape ───────────────────────────────────────────────────────────
print(f"UMAP done. New shape: {reduced_d_10.shape}")
# ── Save the result ───────────────────────────────────────────────────────────
os.makedirs("./output/clusters", exist_ok=True)
np.save("./output/clusters/reduced.npy", reduced_d_10)
print("Saved reduced.npy")
# Visualization ───────────────────────────────────────────────────────────

visualized = umap.UMAP(
    n_components=3,
    n_neighbors=n_neighbors,
    min_dist=0.1,
    metric="cosine",
    n_epochs=500    ,
    random_state=42
    ).fit_transform(embeddings)

print(f"UMAP done. New shape: {visualized.shape}")
os.makedirs("./output/references", exist_ok=True)
meta = json.load(open("./output/references/rule_meta.json"))


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
severity_colors = {"high": "red", "medium": "orange", "low": "green", "unknown": "grey"}
colors = [severity_colors.get(m["severity"], "grey") for m in meta]
ax.scatter(visualized[:,0], visualized[:,1], visualized[:,2], c=colors, s=100)
plt.title("SCAP Rules — UMAP 3D")
os.makedirs("./output/visuals", exist_ok=True)
fig.savefig("./output/visuals/umap_plot_3d.png", dpi=150)

visualized_2d = umap.UMAP(
    n_components=2,
    n_neighbors=n_neighbors,
    min_dist=0.1,
    metric="cosine",
    n_epochs=500,
    random_state=42
).fit_transform(embeddings)

print(f"UMAP 2D done. New shape: {visualized_2d.shape}")

fig2 = plt.figure(figsize=(12, 8))
plt.scatter(visualized_2d[:,0], visualized_2d[:,1], c=colors, s=20, alpha=0.7)
plt.title("SCAP Rules — UMAP 2D")
fig2.savefig("./output/visuals/umap_plot_2d.png", dpi=150)
plt.show()