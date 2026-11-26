"""E7 with the index labels swapped: GitaGround-Local (frozen), Sol, Sol cascade, Gemini high, Gemini cascade.
Retrieval code (gitaground.GitaGround.retrieve) unchanged; only primary/secondary verses (and p_primary) are replaced.
LLM labels carry no probability, so p_primary = 1 for them; 'Local, p=1' isolates that effect.
Vector-search abstention: passage-cosine threshold chosen on the DEV questions only (best balanced accuracy, midpoint
of the best range, as freeze_e7 --calibrate)."""
import sys, os, csv, copy
from collections import defaultdict
sys.path.insert(0, "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama/scripts")
sys.path.insert(0, "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama")
import numpy as np
from evalkit import load_gold
import gitaground
OLD = "/private/tmp/claude-501/-Users-maitreyi-Documents-Capstone/36440e69-87ca-48a6-a581-537a9d90af22/scratchpad/units_v2_before_video11-14.csv"     # the frozen 51-lecture grid (index built before video_11-14 were added)
_uc = gitaground.unit_cache
gitaground.UNITS = OLD
gitaground.unit_cache = lambda p: "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama/results/unit_embeddings.npz" if p == OLD else _uc(p)
import freeze_e7 as fe
fe.TEST_Q = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama/data/" + sys.argv[1]
TESTONLY = len(sys.argv) > 2 and sys.argv[2] == "testonly"
from gitaground import GitaGround
S = "/private/tmp/claude-501/-Users-maitreyi-Documents-Capstone/36440e69-87ca-48a6-a581-537a9d90af22/scratchpad"
R = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama/results"
_, gold, _ = load_gold()
g = GitaGround("all")
if TESTONLY:
    from gemini_runs_55 import TUNE
    keep = [k for k, r in enumerate(g.index) if r["video_file"] not in TUNE["kn"] + TUNE["en"]]
    g.index = [g.index[k] for k in keep]; g.P = g.P[keep]
    print(f"index restricted to test lectures: {len(g.index)} passages, {len({r['video_file'] for r in g.index})} lectures")
print("questions:", fe.TEST_Q)
base = copy.deepcopy(g.index)
units = [r["unit_id"] for r in base]
def col(path, c):
    return {r["unit_id"]: [x for x in r[c].split(";") if x] for r in csv.DictReader(open(path, encoding="utf-8"))}
src = {"Gemini low": col(f"{R}/improved3l_55.csv", "gemini"), "3L + Gemini low": col(f"{R}/improved3l_55.csv", "improved_3l"),
       "Gemini high": col(f"{S}/improved3l_high55.csv", "gemini"), "3L + Gemini": col(f"{S}/improved3l_high55.csv", "improved_3l"),
       "Sol": col(f"{R}/improved3l_sol51.csv", "sol"), "3L + Sol": col(f"{R}/improved3l_sol51.csv", "improved_3l")}
for n, d in src.items():
    miss = [u for u in units if u not in d]
    assert not miss, (n, len(miss))
def use(labels=None, p1=False):
    g.index = copy.deepcopy(base)
    for r in g.index:
        if labels is not None:
            l = labels[r["unit_id"]]
            r["primary_verse"] = l[0] if l else "OFF"
            r["secondary_verses"] = ";".join(l[1:])
    g.pp = np.ones(len(g.index)) if (p1 or labels is not None) else np.array([float(r["p_primary"]) for r in g.index])
    g.by_verse = defaultdict(list)
    for k, r in enumerate(g.index):
        if r["primary_verse"] != "OFF":
            g.by_verse[r["primary_verse"]].append(k)
relevant_of = lambda exp: {k for k, r in enumerate(base) if set(gold[r["unit_id"]]) & exp}
qs = fe.test_questions()
emb = {q["text"]: g.embed([q["text"]])[0] for q in qs}
# vector-search abstention threshold on DEV questions
dq = fe.dev_questions()
ans = [q for q in dq if q["expected"] and relevant_of(q["expected"])]; un = [q for q in dq if not q["expected"]]
a = [float((g.P @ g.embed([q["text"]])[0]).max()) for q in ans]; u = [float((g.P @ g.embed([q["text"]])[0]).max()) for q in un]
vals = sorted(set(a + u)); cands = [(x + y) / 2 for x, y in zip(vals, vals[1:])]
accs = [fe.balanced(a, u, t) for t in cands]; top = max(accs); tied = [t for t, c in zip(cands, accs) if c == top]
thr = tied[len(tied) // 2]
print(f"vector-search abstention: passage cosine < {thr:.4f} (dev balanced accuracy {top:.3f}; {len(ans)} answerable, {len(un)} unanswerable dev)")
vs = {q["text"]: float((g.P @ emb[q["text"]]).max()) for q in qs}
tu = [q for q in qs if not q["expected"]]; ta = [q for q in qs if q["expected"]]
print(f"  test: unanswerable refused {sum(vs[q['text']] < thr for q in tu)}/{len(tu)}, answerable wrongly refused {sum(vs[q['text']] < thr for q in ta)}/{len(ta)}")
# Hit@1 by mode (index-independent)
hm = defaultdict(lambda: [0, 0])
for q in ta:
    ranked = [r for r, _ in g.rank_verses(q["text"], emb[q["text"]])]
    hm[q["mode"]][0] += ranked[0] in q["expected"]; hm[q["mode"]][1] += 1
print("  verse Hit@1 by mode:", {m: f"{h}/{n} = {h / n:.3f}" for m, (h, n) in hm.items()},
      " non-shloka:", f"{sum(h for m, (h, n) in hm.items() if 'shloka' not in m)}/{sum(n for m, (h, n) in hm.items() if 'shloka' not in m)}")
def run(name):
    m = defaultdict(float); claims = [0, 0]; bym = defaultdict(lambda: [0.0, 0])
    for q in ta:
        rel = relevant_of(q["expected"])
        if not rel: continue
        ks, topv = g.retrieve(q["text"])
        m["n"] += 1; m["hit5"] += bool(set(ks) & rel); p = len(set(ks) & rel) / 5; m["p5"] += p
        bym[q["mode"]][0] += p; bym[q["mode"]][1] += 1
        for k in ks:
            r = g.index[k]
            if r["primary_verse"] == topv:
                claims[1] += 1; claims[0] += topv in gold[r["unit_id"]]
    sim_vec = None
    print(f"  {name:22} Hit@5 {m['hit5'] / m['n']:.3f}  P@5 {m['p5'] / m['n']:.3f}  (n {int(m['n'])})  "
          + "  ".join(f"{k} P@5 {s / c:.3f}" for k, (s, c) in bym.items())
          + f"  | primary claims correct {claims[0]}/{claims[1]} = {claims[0] / max(claims[1], 1):.3f}")
use(); run("GitaGround-Local")
use(p1=True); run("GitaGround-Local, p=1")
for n, d in src.items():
    use(d); run(n)
# plain vector P@5 for reference
m = [0, 0, 0]
for q in ta:
    rel = relevant_of(q["expected"])
    if not rel: continue
    vec = list(np.argsort(-(g.P @ emb[q["text"]]))[:5]); m[0] += 1; m[1] += bool(set(vec) & rel); m[2] += len(set(vec) & rel) / 5
print(f"  {'plain vector search':22} Hit@5 {m[1] / m[0]:.3f}  P@5 {m[2] / m[0]:.3f}")
