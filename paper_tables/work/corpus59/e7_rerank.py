"""E7 lexical and hybrid baselines on the deployed 59-lecture index (same 2,583 passages and the same relevance rule as
e7s50/app59.py): Okapi BM25 (k1 = 1.2, b = 0.75; lower-cased whitespace/punctuation tokens, so Kannada script and
romanized Sanskrit stay whole) and a hybrid that fuses the BM25 and dense (BGE-M3) rankings by reciprocal rank fusion
(RRF, k = 60). No parameter is tuned. GitaGround and vector search are re-computed as a consistency check.
Adds hybrid + reranker: the hybrid's top 50 passages re-scored by the BGE-reranker-v2-m3 cross-encoder (max 512 tokens),
top 5 kept. No parameter is tuned.
Usage: e7_rerank.py <questions csv in data/>"""
import sys, os, csv, re, math
from collections import Counter, defaultdict
import numpy as np
R = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama"
W = "/Users/maitreyi/Documents/Capstone/review/paper_v7_work"
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import gitaground
OLD = os.path.join(W, "units_v2_before_video11-14.csv")
_uc = gitaground.unit_cache
gitaground.UNITS = OLD
gitaground.unit_cache = lambda p: os.path.join(R, "results", "unit_embeddings.npz") if p == OLD else _uc(p)
import freeze_e7 as fe
fe.TEST_Q = os.path.join(R, "data", sys.argv[1])
from evalkit import load_gold, load_csv, UNITS, UNITS_HELDOUT
from gitaground import GitaGround
lectures, gold, _ = load_gold()
g = GitaGround("all")
lab = {r["unit_id"]: [x for x in r["improved_3l"].split(";") if x]
       for r in csv.DictReader(open(os.path.join(W, "new4e2", "all59_sol.csv"), encoding="utf-8"))}
grid = load_csv(UNITS) + load_csv(UNITS_HELDOUT)
fz = np.load(os.path.join(W, "new4e2", "unit_embeddings_full.npz")); hz = np.load(_uc(UNITS_HELDOUT))
ids = [u["unit_id"] for u in grid]; assert list(fz["unit_ids"]) + list(hz["unit_ids"]) == ids
emb = np.concatenate([fz["emb"], hz["emb"]]); row = {u: i for i, u in enumerate(ids)}
text = {u["unit_id"]: u["text"] for u in grid}; vf = {u["unit_id"]: u["video_file"] for u in grid}
units = [u for v in sorted(lectures) for u in lectures[v]]; assert len(units) == 2583
g.index = [dict(unit_id=u, video_file=vf[u], primary_verse=(lab[u][0] if lab[u] else "OFF"), secondary_verses=";".join(lab[u][1:])) for u in units]
P = np.stack([emb[row[u]] for u in units]); g.P = P / np.linalg.norm(P, axis=1, keepdims=True)
g.pp = np.ones(len(units)); g.by_verse = defaultdict(list)
for k, r in enumerate(g.index):
    if r["primary_verse"] != "OFF": g.by_verse[r["primary_verse"]].append(k)
# ---- BM25
TOK = re.compile(r"[^\s.,;:!?\"'()\[\]{}|।॥\-]+")
tok = lambda s: TOK.findall(s.lower())
docs = [tok(text[u]) for u in units]; N = len(docs); avg = sum(map(len, docs)) / N
df = Counter(t for d in docs for t in set(d)); tf = [Counter(d) for d in docs]; dl = [len(d) for d in docs]
idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
K1, B = 1.2, 0.75
def bm25(q):
    s = np.zeros(N)
    for t in set(tok(q)):
        if t not in idf: continue
        for k in range(N):
            f = tf[k].get(t)
            if f: s[k] += idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * dl[k] / avg))
    return s
def rrf(*scores, k=60):
    out = np.zeros(N)
    for s in scores:
        rank = np.empty(N); rank[np.argsort(-s)] = np.arange(1, N + 1); out += 1 / (k + rank)
    return out
relevant_of = lambda exp: {k for k, u in enumerate(units) if set(gold[u]) & exp}
qs = fe.test_questions(); ta = [q for q in qs if q["expected"]]
def metrics(get):
    m = defaultdict(float); bym = defaultdict(lambda: [0.0, 0])
    for q in ta:
        rel = relevant_of(q["expected"])
        if not rel: continue
        ks = get(q)
        m["n"] += 1; m["hit5"] += bool(set(ks) & rel); p = len(set(ks) & rel) / 5; m["p5"] += p; m["r5"] += len(set(ks) & rel) / len(rel)
        bym[q["mode"]][0] += p; bym[q["mode"]][1] += 1
    return m, bym
dense = {q["text"]: g.P @ g.embed([q["text"]])[0] for q in ta}
lex = {q["text"]: bm25(q["text"]) for q in ta}
from sentence_transformers import CrossEncoder
import torch
ce = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="mps" if torch.backends.mps.is_available() else "cpu")
DEPTH = 50
def rerank(q):
    cand = list(np.argsort(-rrf(lex[q["text"]], dense[q["text"]]), kind="stable")[:DEPTH])
    sc = ce.predict([(q["text"], text[units[k]]) for k in cand], batch_size=16)
    return [cand[i] for i in np.argsort(-np.asarray(sc), kind="stable")[:5]]
rr = {q["text"]: rerank(q) for q in ta}
print("questions:", sys.argv[1], "| index", N, "passages | BM25 vocabulary", len(idf))
for name, get in (("GitaGround (3L + Sol index)", lambda q: g.retrieve(q["text"])[0]),
                  ("vector search (BGE-M3)", lambda q: list(np.argsort(-dense[q["text"]])[:5])),
                  ("BM25", lambda q: list(np.argsort(-lex[q["text"]], kind="stable")[:5])),
                  ("hybrid BM25 + dense (RRF)", lambda q: list(np.argsort(-rrf(lex[q["text"]], dense[q["text"]]), kind="stable")[:5])),
                  ("hybrid + reranker (top 50)", lambda q: rr[q["text"]])):
    m, bym = metrics(get)
    print(f"  {name:30} Hit@5 {m['hit5'] / m['n']:.3f}  P@5 {m['p5'] / m['n']:.3f}  R@5 {m['r5'] / m['n']:.3f}  (n {int(m['n'])})  "
          + "  ".join(f"{k} P@5 {s / c:.3f}" for k, (s, c) in bym.items()))
