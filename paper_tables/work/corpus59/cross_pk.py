"""Strict F1 and Pk/WD (as freeze_e6) on E2 (23 KN, 11 EN) for the cross-ablation decoders: Local 73 (step4_locked),
Local 145 (local145.pkl), Cue 73 (v2_teacher73), Cue 145 (v2_teacher145)."""
import os, numpy as np
import patch59
from evalkit import load_gold
from exp_e6_ablation import segmentation
from freeze_grounding import f1_strict, strict_counts
from step4_viterbi import OFF, OUT
lectures, gold, _ = load_gold()
refs = list(np.load(patch59.FE)["refs"])
R = os.path.join(patch59.R, "results")
for name, path in (("Local 73 (GitaGround-Local)", OUT), ("Local 145", os.path.join(patch59.SP, "local145.pkl")),
                   ("Cue 73", os.path.join(R, "v2_teacher73.pkl")), ("Cue 145 (GitaGround-Cue)", os.path.join(R, "v2_teacher145.pkl"))):
    cells = []
    for L in ("kn", "en"):
        vids = patch59.read_split(L, "test")
        dec = patch59.predict_locked({v: lectures[v] for v in vids}, "a05", path=path)
        pred = {u: ([] if st == OFF else [refs[st]]) + [refs[j] for j in extra] for v in vids for u, (st, extra, _, _) in zip(lectures[v], dec[v])}
        f = f1_strict([sum(x) for x in zip(*(strict_counts(gold, pred, lectures[v]) for v in vids))])
        pk, wd = segmentation(gold, pred, lectures, vids, primary=True)
        cells.append(f"{L} ({len(vids)}) strict {f:.3f} Pk {pk:.3f} WD {wd:.3f}")
    print(f"{name:28} " + "   ".join(cells))
