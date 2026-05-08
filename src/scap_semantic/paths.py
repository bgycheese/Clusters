from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
PHASE1_OUTPUT_DIR = PROJECT_ROOT.parent / "OpenScap_Dataset_RHEL9" / "output"

POLICIES_FILE = PHASE1_OUTPUT_DIR / "policies.json"

EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
REFERENCES_DIR = OUTPUT_DIR / "references"
CLUSTERS_DIR = OUTPUT_DIR / "clusters"
VISUALS_DIR = OUTPUT_DIR / "visuals"

EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.npy"
RULE_META_FILE = REFERENCES_DIR / "rule_meta.json"
PARAM_SEARCH_FILE = REFERENCES_DIR / "param_search.json"

REDUCED_FILE = CLUSTERS_DIR / "reduced.npy"
LABELS_FILE = CLUSTERS_DIR / "labels.npy"
Y_TRUE_FILE = CLUSTERS_DIR / "y_true.npy"
UMAP_2D_COORDS_FILE = CLUSTERS_DIR / "umap_2d_coords.npy"

CLUSTER_RESULTS_FILE = OUTPUT_DIR / "cluster_results.json"
CLUSTER_SUMMARY_FILE = OUTPUT_DIR / "cluster_summary.json"
SUPERSET_FILE = OUTPUT_DIR / "superset.json"
TRACEABILITY_FILE = OUTPUT_DIR / "traceability.csv"

CONDENSED_TREE_PLOT = VISUALS_DIR / "condensed_tree.png"
UMAP_PLOT_2D = VISUALS_DIR / "umap_plot_2d.png"
UMAP_PLOT_3D = VISUALS_DIR / "umap_plot_3d.png"
UMAP_CLUSTERS_PLOT = VISUALS_DIR / "umap_clusters.png"


def ensure_output_dirs() -> None:
    for directory in (EMBEDDINGS_DIR, REFERENCES_DIR, CLUSTERS_DIR, VISUALS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
