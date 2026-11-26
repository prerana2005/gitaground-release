"""E7 with the app index as it would be deployed: GitaGround-3L + GPT-6.1 Sol labels on ALL 59 lectures (2,583 passages).
Verse ranking, meaning profiles and the abstention rule are the frozen ones (51-lecture grid, as every E7 run);
retrieval code (GitaGround.retrieve) unchanged; only the passage index (labels + embeddings) is replaced.
Vector search searches the same 2,583 passages; its abstention threshold is re-chosen on the DEV questions on that index
(as e7_label_sources.py). LLM labels carry no probability, so p_primary = 1.
Usage: app59.py <questions csv in data/>"""
import sys, os, csv, copy
from collections import defaultdict
R = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama"
SP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import numpy as np
import gitaground
OLD = os.path.join(SP, "units_v2_before_video11-14.csv")
_uc = gitaground.unit_cache
gitaground.UNITS = OLD
gitaground.unit_cache = lambda p: os.path.join(R, "results", "unit_embeddings.npz") if p == OLD else _uc(p)
import freeze_e7 as fe
fe.TEST_Q = os.path.join(R, "data", sys.argv[1])
from evalkit import load_gold, load_csv, UNITS, UNITS_HELDOUT
from gitaground import GitaGround
lectures, gold, _ = load_gold()
g = GitaGround("all")
n51 = len(g.index)
# ---- the 59-lecture 3L + Sol index
lab = {r["unit_id"]: [x for x in r["improved_3l"].split(";") if x]
       for r in csv.DictReader(open(os.path.join(SP, "new4e2", "all59_sol.csv"), encoding="utf-8"))}
grid = load_csv(UNITS) + load_csv(UNITS_HELDOUT)
emb = np.concatenate([np.load(os.path.join(SP, "new4e2", "unit_embeddings_full.npz"))["emb"],
                      np.load(_uc(UNITS_HELDOUT))["emb"]])
ids = [u["unit_id"] for u in grid]
assert list(np.load(os.path.join(SP, "new4e2", "unit_embeddings_full.npz"))["unit_ids"]) + list(np.load(_uc(UNITS_HELDOUT))["unit_ids"]) == ids
row = {u: i for i, u in enumerate(ids)}
units = [u for v in sorted(lectures) for u in lectures[v]]
assert len(units) == 2583 and all(u in lab for u in units)
g.index = [dict(unit_id=u, video_file=next(x["video_file"] for x in [grid[row[u]]]),
                primary_verse=(lab[u][0] if lab[u] else "OFF"), secondary_verses=";".join(lab[u][1:])) for u in units]
g.P = np.stack([emb[row[u]] for u in units]) / np.linalg.norm(np.stack([emb[row[u]] for u in units]), axis=1, keepdims=True)
g.pp = np.ones(len(g.index))
g.by_verse = defaultdict(list)
for k, r in enumerate(g.index):
    if r["primary_verse"] != "OFF":
        g.by_verse[r["primary_verse"]].append(k)
print(f"index: {n51} passages (frozen 51-lecture) -> {len(g.index)} passages, {len({r['video_file'] for r in g.index})} lectures, "
      f"labels GitaGround-3L + Sol; {sum(r['primary_verse'] == 'OFF' for r in g.index)} passages OFF")
relevant_of = lambda exp: {k for k, r in enumerate(g.index) if set(gold[r["unit_id"]]) & exp}
qs = fe.test_questions()
qe = {q["text"]: g.embed([q["text"]])[0] for q in qs}
dq = fe.dev_questions()
ans = [q for q in dq if q["expected"] and relevant_of(q["expected"])]; un = [q for q in dq if not q["expected"]]
a = [float((g.P @ g.embed([q["text"]])[0]).max()) for q in ans]; u_ = [float((g.P @ g.embed([q["text"]])[0]).max()) for q in un]
vals = sorted(set(a + u_)); cands = [(x + y) / 2 for x, y in zip(vals, vals[1:])]
accs = [fe.balanced(a, u_, t) for t in cands]; top = max(accs); tied = [t for t, c in zip(cands, accs) if c == top]
thr = tied[len(tied) // 2]
ta = [q for q in qs if q["expected"]]; tu = [q for q in qs if not q["expected"]]
vs = {q["text"]: float((g.P @ qe[q["text"]]).max()) for q in qs}
ours_abst = {q["text"]: g.abstains(q["text"]) for q in qs}
print(f"abstention  GitaGround (profile rule, frozen): unanswerable refused {sum(bool(ours_abst[q['text']]) for q in tu)}/{len(tu)}, "
      f"answerable wrongly refused {sum(bool(ours_abst[q['text']]) for q in ta)}/{len(ta)}")
print(f"            vector search (passage cosine < {thr:.4f}, dev): unanswerable refused {sum(vs[q['text']] < thr for q in tu)}/{len(tu)}, "
      f"answerable wrongly refused {sum(vs[q['text']] < thr for q in ta)}/{len(ta)}")
hm = defaultdict(lambda: [0, 0])
for q in ta:
    ranked = [r for r, _ in g.rank_verses(q["text"], qe[q["text"]])]
    hm[q["mode"]][0] += ranked[0] in q["expected"]; hm[q["mode"]][1] += 1
print("verse Hit@1:", {m: f"{h}/{n} = {h / n:.3f}" for m, (h, n) in hm.items()},
      f" all {sum(h for h, _ in hm.values())}/{sum(n for _, n in hm.values())}")
def metrics(get):
    m = defaultdict(float); bym = defaultdict(lambda: [0.0, 0]); cl = [0, 0]
    for q in ta:
        rel = relevant_of(q["expected"])
        if not rel: continue
        ks, topv = get(q)
        m["n"] += 1; m["hit5"] += bool(set(ks) & rel); p = len(set(ks) & rel) / 5; m["p5"] += p; m["r5"] += len(set(ks) & rel) / len(rel)
        bym[q["mode"]][0] += p; bym[q["mode"]][1] += 1
        for k in ks:
            if topv and g.index[k]["primary_verse"] == topv:
                cl[1] += 1; cl[0] += topv in gold[g.index[k]["unit_id"]]
    return m, bym, cl
for name, get in (("GitaGround-3L + Sol index", lambda q: g.retrieve(q["text"])),
                  ("vector search (same 2,583 passages)", lambda q: (list(np.argsort(-(g.P @ qe[q["text"]]))[:5]), None))):
    m, bym, cl = metrics(get)
    print(f"  {name:38} Hit@5 {m['hit5'] / m['n']:.3f}  P@5 {m['p5'] / m['n']:.3f}  R@5 {m['r5'] / m['n']:.3f}  (n {int(m['n'])})  "
          + "  ".join(f"{k} P@5 {s / c:.3f}" for k, (s, c) in bym.items())
          + (f"  | primary claims correct {cl[0]}/{cl[1]} = {cl[0] / cl[1]:.3f}" if cl[1] else ""))
