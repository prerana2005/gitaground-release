"""Cue inputs for video_11-14 without touching repo caches:
1. BGE-M3 embeddings of the 400 new passages (embeddings.embed_texts, as embed_scores.py) -> combined cache in scratch
2. step2_features.main() run unchanged except: the 'original' lecture set = the 51 lectures of the old units_v2
   (so existing profiles are not changed; the new lectures use all originals, as new-speaker lectures do),
   embedding cache + output path in scratch. Then checks every existing row equals results/step2_features.npz."""
import csv, os, sys, re
import numpy as np
REPO = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama"
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
OLD_GRID = os.path.join(os.path.dirname(SP), "units_v2_before_video11-14.csv")
old_rows = list(csv.DictReader(open(OLD_GRID, encoding="utf-8")))
new_rows = list(csv.DictReader(open(os.path.join(REPO, "data/units_v2.csv"), encoding="utf-8")))
old_ids = {r["unit_id"] for r in old_rows}
add = [r for r in new_rows if r["unit_id"] not in old_ids]
assert [r["unit_id"] for r in new_rows[:len(old_rows)]] == [r["unit_id"] for r in old_rows]
EMB = os.path.join(SP, "unit_embeddings_full.npz")
if not os.path.exists(EMB):
    from embeddings import embed_texts
    z = np.load(os.path.join(REPO, "results/unit_embeddings.npz"))
    U = np.asarray(embed_texts([r["text"] for r in add]), dtype=np.float32)
    np.savez_compressed(EMB, unit_ids=np.array([r["unit_id"] for r in new_rows]),
                        emb=np.concatenate([z["emb"], U]))
    print("embedded", len(add), "new passages")
import step2_features as s2
import embed_scores
src = open(s2.__file__, encoding="utf-8").read()
fn = re.search(r"\ndef main\(\):.*?(?=\n\ndef |\nif __name__)", src, re.S).group(0)
fn = fn.replace('original = {u["video_file"] for u in grid}', 'original = {u["video_file"] for u in grid} - NEWV')
fn = fn.replace("    report(units, row, feats, refs)", "")
assert "- NEWV" in fn
g = vars(s2)
g["NEWV"] = {r["video_file"] for r in add}
g["OUT"] = os.path.join(SP, "step2_features_full.npz")
real_cache = s2.unit_cache
g["unit_cache"] = lambda p: EMB if os.path.basename(p) == "units_v2.csv" else real_cache(p)
sys.argv = ["step2_features.py"]
exec(fn, g)
g["main"]()
a, b = np.load(os.path.join(REPO, "results/step2_features.npz")), np.load(g["OUT"])
ida, idb = list(a["unit_ids"]), list(b["unit_ids"])
pos = [idb.index(u) for u in ida]
worst = {k: float(np.max(np.abs(a[k].astype(np.float64) - b[k][pos].astype(np.float64)))) for k in a if k not in ("unit_ids", "refs")}
print("existing rows", len(ida), "of", len(idb), "max abs diff per feature:", worst)
