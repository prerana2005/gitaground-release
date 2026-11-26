"""expanded scores for the 400 new passages with the STORED verse-document vectors: E recovered from
results/embed_scores.npz by least squares (expanded = U @ E.T); checks the fit reproduces the stored scores."""
import os
import numpy as np
R = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama"
SP = os.path.dirname(os.path.abspath(__file__))
old = np.load(os.path.join(R, "results/embed_scores.npz")); ou = np.load(os.path.join(R, "results/unit_embeddings.npz"))
held = np.load(os.path.join(R, "results/embed_scores_heldout.npz")); hu = np.load(os.path.join(R, "results/unit_embeddings_units_heldout_v1.npz"))
U = ou["emb"].astype(np.float64)
Et, *_ = np.linalg.lstsq(U, old["expanded"].astype(np.float64), rcond=None)
print("rank U", np.linalg.matrix_rank(U), " fit max|diff| on 1390:", abs(U @ Et - old["expanded"]).max())
hi = {u: i for i, u in enumerate(hu["unit_ids"])}; hk = [hi[u] for u in held["unit_ids"]]
print("held-out 494 (not used in fit) max|diff|:", abs(hu["emb"][hk] @ Et - held["expanded"]).max())
new = np.load(os.path.join(SP, "embed_scores_new400.npz"))
full = np.load(os.path.join(os.path.dirname(SP), "new4e2/unit_embeddings_full.npz"))
ix = {u: i for i, u in enumerate(full["unit_ids"])}
Un = full["emb"][[ix[u] for u in new["unit_ids"]]].astype(np.float64)
ex = (Un @ Et).astype(np.float32)
print("new 400: stored-vector vs recomputed-vector max|diff|", abs(ex - new["expanded"]).max())
np.savez_compressed(os.path.join(SP, "embed_scores_new400.npz"), unit_ids=new["unit_ids"], refs=new["refs"],
                    trans=new["trans"], expanded=ex)
print("rewrote embed_scores_new400.npz with stored verse vectors")
