import numpy as np
import os, json
from sentence_transformers import SentenceTransformer
# device mps is for MacOS, cuda = GPU, and device=cpu
# can remove the token afterwards
os.environ["HF_TOKEN"] = str(input("Your HF TOKEN for faster download/upload: "))

model_1 = SentenceTransformer('markusbayer/CySecBERT', device='mps')

# BEST for a laptop specification
BATCH_SIZE = 16

rules = json.load(open("../OpenScap_Dataset_RHEL9/output/policies.json"))
texts = [f"{r['title']}.{r['description']}" for r in rules]

embeddings = model_1.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, prompt="Find semantically similar elements and clusterize them")
meta = [{"id": r['id'], "rule": r['title'], "severity": r['severity'], "profiles": r['profiles']} for r in rules]

os.makedirs("./output/embeddings", exist_ok=True)

np.save("./output/embeddings/embeddings.npy", embeddings)
os.makedirs("./output/references", exist_ok=True)
json.dump(meta, open("./output/references/rule_meta.json", "w"), indent=2)