"""E7 label-source table on the 59-lecture corpus: the app59.py index (all 2,583 passages, or the 50 test lectures,
2,209 passages) with its labels swapped per source. Verse ranking / profiles / abstention frozen as in every E7 run;
retrieval code unchanged. Local = step4_locked decoded on all 59 (checked against results/step5_index_all.csv on the
1,884 passages it has), p_primary = its decoded probability (or 1). LLM labels carry no probability: p_primary = 1.
Usage: labelsrc59.py <questions csv in data/> all|testonly"""
import sys, os, csv
from collections import defaultdict
import patch59
R, W = patch59.R, patch59.W
import numpy as np
import gitaground
OLD = os.path.join(W, "units_v2_before_video11-14.csv")
_uc = gitaground.unit_cache
gitaground.UNITS = OLD
gitaground.unit_cache = lambda p: os.path.join(R, "results", "unit_embeddings.npz") if p == OLD else _uc(p)
import freeze_e7 as fe
fe.TEST_Q = os.path.join(R, "data", sys.argv[1])
TESTONLY = sys.argv[2] == "testonly"
from evalkit import load_gold, load_csv, UNITS, UNITS_HELDOUT
from gitaground import GitaGround
from step4_viterbi import OFF
from gemini_runs_55 import TUNE
lectures, gold, _ = load_gold()
g = GitaGround("all")
dec = patch59.predict_locked({v: lectures[v] for v in lectures}, "a05")
FZ = np.load(patch59.FE); refs = list(FZ["refs"]); FROW = {u: i for i, u in enumerate(FZ["unit_ids"])}
loc, locp = {}, {}
for v in lectures:
    for u, (st, extra, p_state, _) in zip(lectures[v], dec[v]):
        sec = sorted(set(j for j in np.flatnonzero(FZ["anchor"][FROW[u]]) if j != st) | set(extra))   # as step5_index
        loc[u] = (["OFF"] if st == OFF else [refs[st]]) + [refs[j] for j in sec]
        locp[u] = p_state
old = {r["unit_id"]: r for r in csv.DictReader(open(os.path.join(R, "results", "step5_index_all.csv"), encoding="utf-8"))}
bad = [u for u, r in old.items() if r["primary_verse"] != loc[u][0] or r["secondary_verses"] != ";".join(loc[u][1:])
       or abs(float(r["p_primary"]) - round(locp[u], 4)) > 1e-4]
print(f"Local check vs step5_index_all.csv: {len(old)} passages, {len(bad)} differ")
def col(path, c):
    return {r["unit_id"]: [x for x in r[c].split(";") if x] for r in csv.DictReader(open(path, encoding="utf-8"))}
N = os.path.join(W, "new4e2")
src = {"Gemini low": col(f"{N}/all59_gemini_low.csv", "gemini"), "3L + Gemini low": col(f"{N}/all59_gemini_low.csv", "improved_3l"),
       "Gemini high": col(f"{N}/all59_gemini.csv", "gemini"), "3L + Gemini high": col(f"{N}/all59_gemini.csv", "improved_3l"),
       "Sol": col(f"{N}/all59_sol.csv", "sol"), "3L + Sol": col(f"{N}/all59_sol.csv", "improved_3l")}
grid = load_csv(UNITS) + load_csv(UNITS_HELDOUT)
fz = np.load(os.path.join(N, "unit_embeddings_full.npz")); hz = np.load(_uc(UNITS_HELDOUT))
ids = [u["unit_id"] for u in grid]
assert list(fz["unit_ids"]) + list(hz["unit_ids"]) == ids
emb = np.concatenate([fz["emb"], hz["emb"]]); row = {u: i for i, u in enumerate(ids)}
vids = [v for v in sorted(lectures) if not (TESTONLY and v in TUNE["kn"] + TUNE["en"])]
units = [u for v in vids for u in lectures[v]]
for n, d in src.items():
    assert all(u in d for u in units), n
vf = {u["unit_id"]: u["video_file"] for u in grid}
P = np.stack([emb[row[u]] for u in units]); P = P / np.linalg.norm(P, axis=1, keepdims=True)
print(f"index: {len(units)} passages, {len(vids)} lectures ({'50 test lectures only' if TESTONLY else 'all 59'})")
def use(labels, pp):
    g.index = [dict(unit_id=u, video_file=vf[u], primary_verse=(labels[u][0] if labels[u] else "OFF"),
                    secondary_verses=";".join(labels[u][1:])) for u in units]
    g.P, g.pp = P, np.array([pp(u) for u in units])
    g.by_verse = defaultdict(list)
    for k, r in enumerate(g.index):
        if r["primary_verse"] != "OFF":
            g.by_verse[r["primary_verse"]].append(k)
relevant_of = lambda exp: {k for k, u in enumerate(units) if set(gold[u]) & exp}
qs = fe.test_questions(); ta = [q for q in qs if q["expected"]]
qe = {q["text"]: g.embed([q["text"]])[0] for q in qs}
def metrics(get):
    m = defaultdict(float); cl = [0, 0]
    for q in ta:
        rel = relevant_of(q["expected"])
        if not rel: continue
        ks, topv = get(q)
        m["n"] += 1; m["hit5"] += bool(set(ks) & rel); m["p5"] += len(set(ks) & rel) / 5; m["r5"] += len(set(ks) & rel) / len(rel)
        for k in ks:
            if topv and g.index[k]["primary_verse"] == topv:
                cl[1] += 1; cl[0] += topv in gold[g.index[k]["unit_id"]]
    return m, cl
rows = [("GitaGround-Local (decoded p)", loc, lambda u: locp[u]), ("GitaGround-Local (p = 1)", loc, lambda u: 1.0)] + \
       [(n, d, lambda u: 1.0) for n, d in src.items()]
for name, lab, pp in rows:
    use(lab, pp)
    m, cl = metrics(lambda q: g.retrieve(q["text"]))
    print(f"  {name:30} Hit@5 {m['hit5'] / m['n']:.3f}  P@5 {m['p5'] / m['n']:.3f}  R@5 {m['r5'] / m['n']:.3f}  (n {int(m['n'])})"
          f"  | primary claims correct {cl[0]}/{cl[1]} = {cl[0] / cl[1]:.3f}")
m, _ = metrics(lambda q: (list(np.argsort(-(P @ qe[q["text"]]))[:5]), None))
print(f"  {'vector search':30} Hit@5 {m['hit5'] / m['n']:.3f}  P@5 {m['p5'] / m['n']:.3f}  R@5 {m['r5'] / m['n']:.3f}")
