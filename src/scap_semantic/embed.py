import json
import os
import time

import numpy as np
from dotenv import load_dotenv

try:
    from paths import EMBEDDINGS_FILE, POLICIES_FILE, PROJECT_ROOT, RULE_META_FILE, ensure_output_dirs
except ImportError:
    from paths import EMBEDDINGS_FILE, POLICIES_FILE, PROJECT_ROOT, RULE_META_FILE, ensure_output_dirs


# MODEL_NAME = "markusbayer/CySecBERT"
MODEL_NAME = "cisco-ai/SecureBERT2.0-biencoder"
BATCH_SIZE = 32
DEFAULT_DEVICE = "mps"


def load_rules(path=POLICIES_FILE):
    with path.open() as file:
        return json.load(file)


def build_rule_texts(rules):
    return [f"{rule['title']}. {rule['description']}" for rule in rules]


def build_rule_meta(rules):
    return [
        {
            "position": index,
            "id": rule["id"],
            "title": rule["title"],
            "severity": rule["severity"],
            "profiles": rule["profiles"],
        }
        for index, rule in enumerate(rules)
    ]


def main() -> None:
    from sentence_transformers import SentenceTransformer

    load_dotenv(PROJECT_ROOT / ".env")
    if "HF_TOKEN" not in os.environ:
        print("You can set your HF_TOKEN for higher download speed")

    device = os.getenv("SENTENCE_TRANSFORMERS_DEVICE", DEFAULT_DEVICE)
    model = SentenceTransformer(MODEL_NAME, device=device)
    # TODO: Consider what to do with the truncated sentences
    # print(model.max_seq_length)

    rules = load_rules()
    texts = build_rule_texts(rules)

    # for batch_size in [16, 32, 64, 128, 256, 512]:
    #     start_t = time.time()
    #     model.encode(texts, batch_size=batch_size,show_progress_bar=True)
    #     duration = time.time() - start_t
    #     print(
    #         f"Batch size: {batch_size}, Duration: {duration:.2f} seconds; {len(texts) / duration:.2f} sentences per second")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True
    )
    meta = build_rule_meta(rules)

    ensure_output_dirs()
    np.save(EMBEDDINGS_FILE, embeddings)
    with RULE_META_FILE.open("w") as file:
        json.dump(meta, file, indent=2)


if __name__ == "__main__":
    main()
