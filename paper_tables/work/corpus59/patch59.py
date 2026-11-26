"""Shared patches so the repo scripts score the full 59-lecture corpus (repo files untouched):
FEATS -> scratch step2_features_full.npz (existing rows identical), E2 English test += video_11-14,
BGE-M3 margin scores for the 400 new passages, teacher P_off for the new passages in the Cue models,
outputs written here. NEW=False reproduces the earlier 42-lecture runs (regression check)."""
import os, sys, json, copy, pickle
import numpy as np
R = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama"
SP = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(SP)
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "scripts"))
os.chdir(R)
NEWE2 = ["video_11", "video_12", "video_13", "video_14"]
NEW = "--old" not in sys.argv
if "--old" in sys.argv: sys.argv.remove("--old")
import evalkit, step3_scorer, step4_viterbi, heldout_eval
FE = os.path.join(W, "new4e2", "step2_features_full.npz")
step3_scorer.FEATS = step4_viterbi.FEATS = FE
_rs = evalkit.read_split
def read_split(language, split, *a, **k):
    out = _rs(language, split, *a, **k)
    return out + NEWE2 if NEW and language == "en" and split in ("test", "all") else out
evalkit.read_split = read_split
_ls = heldout_eval.load_scores
def load_scores(rep):
    row, mats, refs = _ls(rep)
    z = np.load(os.path.join(SP, "embed_scores_new400.npz"))
    assert list(z["refs"]) == list(refs)
    for i, u in enumerate(z["unit_ids"]):
        row[u] = (len(mats), i)
    mats.append(z[rep])
    return row, mats, refs
heldout_eval.load_scores = load_scores
def teacher_soft():
    soft = {}
    for line in open(os.path.join(W, "new4e2", "silver", "v2_window_low_gpt-5-mini.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        soft.setdefault(r["unit_id"], min(0.98, max(0.02, (r["score"] + 5) / 10)))
    return soft
SOFT = teacher_soft()
_pl = step4_viterbi.predict_locked
def predict_locked(lecture_units, alpha="a05", path=step4_viterbi.OUT, model=None):
    if model is None:
        model = pickle.load(open(path, "rb"))[alpha]
    if model.get("p_off") is not None:
        m2 = copy.copy(model); m2["p_off"] = dict(model["p_off"])
        for us in lecture_units.values():
            for u in us:
                if u not in m2["p_off"]:
                    m2["p_off"][u] = SOFT[u]
        model = m2
    return _pl(lecture_units, alpha, model=model)
step4_viterbi.predict_locked = predict_locked
def patch(mod):
    """re-bind the names a script imported with `from x import y`"""
    for k, v in dict(FEATS=FE, read_split=read_split, load_scores=load_scores, predict_locked=predict_locked).items():
        if hasattr(mod, k):
            setattr(mod, k, v)
def run_main(mod, out_old, out_new):
    """exec mod.main with its output path redirected here"""
    import re
    src = open(mod.__file__, encoding="utf-8").read()
    fn = re.search(r"\ndef main\(.*?(?=\n\n\nif __name__)", src, re.S).group(0)
    assert out_old in fn, out_old
    fn = fn.replace(out_old, out_new)
    g = vars(mod); g["SP"] = SP; g["TAG"] = "59" if NEW else "old"
    exec(fn, g)
    mod.main()
