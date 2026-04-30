import numpy as np
import os, json
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, dotenv_values
# device mps is for MacOS, cuda = GPU, and device=cpu
# can remove the token afterwards
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("You can set your HF_TOKEN for higher download speed")

model_1 = SentenceTransformer('markusbayer/CySecBERT', device='mps')
print(model_1.max_seq_length) # 512
# TODO: implement a solution for sentences that are too long.
# BEST for a MAC laptop specification
BATCH_SIZE = 16

rules = json.load(open("../OpenScap_Dataset_RHEL9/output/policies.json"))
texts = [f"{r['title']}.{r['description']}" for r in rules]


embeddings = model_1.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, prompt="Find semantically similar elements and clusterize them")
meta = [{"position" : index , "id": r['id'], "title": r['title'], "severity": r['severity'], "profiles": r["profiles"]} for index,r in enumerate(rules)]

os.makedirs("./output/embeddings", exist_ok=True)

np.save("./output/embeddings/embeddings.npy", embeddings)
os.makedirs("./output/references", exist_ok=True)
json.dump(meta, open("./output/references/rule_meta.json", "w"), indent=2)