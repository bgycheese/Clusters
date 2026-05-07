try:
    from src.scap_semantic.paths import EMBEDDINGS_FILE, POLICIES_FILE, PROJECT_ROOT, RULE_META_FILE, ensure_output_dirs, TRACEABILITY_FILE
except ImportError:
    from paths import EMBEDDINGS_FILE, POLICIES_FILE, PROJECT_ROOT, RULE_META_FILE, ensure_output_dirs

def terminal_output(view_value=5, col=["rank", "cluster_id", "cluster_profile_count", "rule_title"]) -> None:
    import pandas as pd
    ensure_output_dirs()
    df = pd.read_csv(TRACEABILITY_FILE)
    print(
        df[col]
        .dropna()
        .sort_values("rule_profile_count", ascending=False)
        .head(view_value)
        .to_string()
    )

def main() -> None:
    terminal_output(20, ["rank", "cluster_id", "cluster_profile_count", "rule_title", "sim_to_canonical", "rule_profile_count"] )

if __name__ == "__main__":
    main()