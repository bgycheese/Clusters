try:
    from paths import EMBEDDINGS_FILE, POLICIES_FILE, PROJECT_ROOT, RULE_META_FILE, ensure_output_dirs, TRACEABILITY_FILE
except ImportError:
    from paths import EMBEDDINGS_FILE, POLICIES_FILE, PROJECT_ROOT, RULE_META_FILE, ensure_output_dirs

def terminal_output(view_value=5, col=None) -> None:
    if col is None:
        col = ["rank", "cluster_id", "cluster_profile_count", "rule_title"]
    import pandas as pd
    ensure_output_dirs()
    df = pd.read_csv(TRACEABILITY_FILE)
    print(
        df[col]
        .dropna()
        .sort_values(by=["rule_profile_count",'cluster_profile_count' ], ascending=[False, False])
        .head(view_value)
        .to_string()
    )

def main() -> None:
    import os
    os.system('python3 src/scap_semantic/embed.py')
    os.system('python3 src/scap_semantic/reduce_dim.py')
    os.system('python3 src/scap_semantic/cluster.py')
    os.system('python3 src/scap_semantic/superset.py')
    terminal_output(20, ["rank", "cluster_id", "cluster_profile_count", "rule_title", "sim_to_canonical", "rule_profile_count"] )
if __name__ == "__main__":
    main()