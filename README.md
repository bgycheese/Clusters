# SCAP Compliance Rule Analysis — Phase 2

Semantic clustering of RHEL 9 security compliance rules using CySecBERT embeddings, UMAP dimensionality reduction, and HDBSCAN clustering. Built on top of the dataset extracted in Phase 1 (`OpenScap_Dataset_RHEL9/`).

---

## What This Does

Takes 1,467 SCAP security rules from 19 compliance profiles (CIS, DISA STIG, HIPAA, PCI-DSS, NIST 800-171, ANSSI, etc.) and automatically identifies which rules across different frameworks are semantically equivalent — rules that say the same thing in different words.

**Core thesis question:** Can semantic similarity find cross-profile rule equivalences that naive string matching cannot?

---

## Pipeline

Run the scripts in this order:

```
embed.py → reduce_dim.py → cluster.py
```

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `embed.py` | `policies.json` | `embeddings.npy`, `rule_meta.json` |
| 2 | `reduce_dim.py` | `embeddings.npy` | `reduced.npy`, `umap_plot_2d.png`, `umap_plot_3d.png` |
| 3 | `cluster.py` | `reduced.npy` | `cluster_results.json`, `cluster_summary.json` |

---

## Scripts

### `embed.py` — Semantic Embedding
Loads all 1,467 rules from `policies.json` and encodes each rule's `title + description` using **CySecBERT** (`markusbayer/CySecBERT`) — a BERT model pre-trained on cybersecurity text. Each rule becomes a 768-dimensional vector where similar meaning produces similar vectors.

- Model: CySecBERT via `sentence-transformers`
- Device: MPS (Apple Silicon GPU)
  - **MPS device** (Apple Silicon) assumed. Change `device='mps'` to `device='cpu'` or `device='cuda'` in `embed.py` if running on a different machine
- Batch size: 16
- Output shape: `(1467, 768)`

### `reduce_dim.py` — Dimensionality Reduction
Compresses the 768-dimensional embeddings down to 10 dimensions using a two-stage UMAP reduction (768 → 100 → 10). This preserves neighbourhood structure while making the data suitable for density-based clustering. Also produces 2D and 3D UMAP projections for visualisation.

- Stage 1: UMAP 768 → 100 (`metric=cosine`)
- Stage 2: UMAP 100 → 10 (`metric=euclidean`)
- Visualisation: UMAP → 2D and 3D, coloured by severity
- `n_neighbors` set to 1% of dataset size (~14)

### `cluster.py` — HDBSCAN Clustering
Groups the reduced embeddings into clusters of semantically equivalent rules using HDBSCAN. Rules that do not belong to any cluster are labelled as noise (`-1`).

- Algorithm: HDBSCAN
- `min_cluster_size=5`, `min_samples=3`, `metric=euclidean`
- Result: **137 clusters**, **71 noise points** from 1,467 rules
- Top cluster spans all **19 compliance profiles**

---

## Output Structure

```
output/
├── embeddings/
│   └── embeddings.npy          # (1467, 768) — CySecBERT vectors
├── references/
│   └── rule_meta.json          # Rule ID index (id, title, severity, profiles)
├── clusters/
│   └── reduced.npy             # (1467, 10) — UMAP-reduced embeddings
├── visuals/
│   ├── umap_plot_2d.png        # 2D scatter coloured by severity
│   └── umap_plot_3d.png        # 3D scatter coloured by severity
├── cluster_results.json        # Every rule with its cluster label
└── cluster_summary.json        # Per-cluster summary sorted by profile coverage
```

### `cluster_results.json`
One entry per rule. Each rule is assigned a cluster ID (`-1` = noise).

```json
{
  "profile_count": 2,
  "id": "xccdf_org.ssgproject.content_rule_prefer_64bit_os",
  "title": "Prefer to use a 64-bit Operating System when supported",
  "severity": "medium",
  "profiles": ["anssi_bp28_enhanced", "anssi_bp28_high"],
  "cluster": 12
}
```

### `cluster_summary.json`
One entry per cluster, sorted by how many compliance profiles that cluster spans (descending). The first entry is the most universally shared security requirement across all frameworks.

```json
{
  "cluster_id": 62,
  "profile_count": 19,
  "profiles_covered": ["anssi_bp28_enhanced", "cis_server_l1", "stig", ...],
  "rules": ["xccdf_...rule_a", "xccdf_...rule_b"]
}
```

---

## Results

| Metric | Value |
|---|---|
| Total rules | 1,467 |
| Compliance profiles | 19 |
| Clusters found | 137 |
| Rules in clusters | 1,396 |
| Noise (unique rules) | 71 |
| Top cluster profile coverage | 19 / 19 profiles |

---

## Requirements

```
sentence-transformers
umap-learn
hdbscan
numpy
matplotlib
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

> **Note:** `embed.py` will prompt for a HuggingFace token at runtime for faster model downloads. Get one free at [huggingface.co](https://huggingface.co).

---

## Dependencies

- **Phase 1 dataset** (`../OpenScap_Dataset_RHEL9/output/policies.json`) must exist before running `embed.py`
