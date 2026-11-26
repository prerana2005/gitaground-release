"""embed_scores for the 400 video_11-14 passages (expanded + trans), same code as scripts/embed_scores.py,
unit vectors from new4e2/unit_embeddings_full.npz. Checks the verse vectors reproduce results/embed_scores.npz."""
import os, sys
import numpy as np
R = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama"
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "scripts"))
os.chdir(R)
from embeddings import embed_texts
from verse_database import VERSES_BG_CH2
from verse_translations import VERSE_TRANSLATIONS
refs = [v["ref"] for v in VERSES_BG_CH2]
flat = [(i, t) for i, r in enumerate(refs) for t in VERSE_TRANSLATIONS[r]]
T = np.asarray(embed_texts([t for _, t in flat]), dtype=np.float32)
docs = ["\n".join([v["devanagari"], v["kannada"]] + list(VERSE_TRANSLATIONS[v["ref"]])) for v in VERSES_BG_CH2]
E = np.asarray(embed_texts(docs), dtype=np.float32)
def score(U):
    sims = U @ T.T
    tr = np.full((len(U), len(refs)), -1.0, dtype=np.float32)
    for c, (i, _) in enumerate(flat):
        tr[:, i] = np.maximum(tr[:, i], sims[:, c])
    return tr, (U @ E.T).astype(np.float32)
old = np.load(os.path.join(R, "results/embed_scores.npz")); ou = np.load(os.path.join(R, "results/unit_embeddings.npz"))
assert list(old["unit_ids"]) == list(ou["unit_ids"]) and list(old["refs"]) == refs
tr, ex = score(ou["emb"])
print("check vs stored: max |diff| expanded %.2e trans %.2e" % (abs(ex - old["expanded"]).max(), abs(tr - old["trans"]).max()))
full = np.load(os.path.join(os.path.dirname(SP), "new4e2/unit_embeddings_full.npz"))
ids = list(full["unit_ids"]); keep = [i for i, u in enumerate(ids) if u not in set(old["unit_ids"])]
assert len(keep) == 400, len(keep)
tr, ex = score(full["emb"][keep])
np.savez_compressed(os.path.join(SP, "embed_scores_new400.npz"), unit_ids=np.array([ids[i] for i in keep]),
                    refs=np.array(refs), trans=tr, expanded=ex)
print("wrote embed_scores_new400.npz", ex.shape, sorted({ids[i].rsplit('_', 1)[0] for i in keep}))
