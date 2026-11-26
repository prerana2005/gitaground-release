"""Every number for paper v7, from stored labels only (no API call).
Test = 50 lectures: E2 Kannada 23, E2 English 11 (7 + video_11-14), E3 Kannada 9, E3 English 7 (ISKCON, Chinmaya 03/04/06-09).
LLM labels + improved 3L + old 3L from new4e2/all59_{gemini_low,gemini,sol}.csv; Cue/conf/P_off from improved3l_55.cue_all
with the same patches as allsys.py; the improved 3L is re-implemented here and checked against the CSV row by row."""
import csv, os, re, sys
import numpy as np
R = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama"
SP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = os.path.join(SP, "new4e2")
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "scripts"))
from evalkit import load_gold, read_split
from freeze_grounding import f1_near, f1_strict, near_counts, paired, strict_counts, theme_counts
from recall_by_type import GROUPS, recall_ci, hits
from baselines import carry
import gemini_runs_55, new4_gemini3l, step4_viterbi
import improved3l_55 as m
NEWE2 = ["video_11", "video_12", "video_13", "video_14"]
NEW4 = gemini_runs_55.NEW4; TUNE = gemini_runs_55.TUNE
FE = os.path.join(N, "step2_features_full.npz")
step4_viterbi.FEATS = m.FEATS = FE
new4_gemini3l.SILVER = os.path.join(N, "silver")
_g = gemini_runs_55.groups
m.groups = lambda: _g() + [("New E2 English (video_11-14)", NEWE2)]
src = open(m.__file__, encoding="utf-8").read()
fn = re.search(r"\ndef cue_all\(.*?(?=\n\n\ndef )", src, re.S).group(0).replace(
    "for v in NEW4 for u in lectures[v]", "for v in NEW4 + NEWE2 for u in lectures[v]")
g = vars(m); g["NEWE2"] = NEWE2; exec(fn, g)
lectures, gold, _ = load_gold()
cue, conf, poff, anchor = m.cue_all(lectures)

E2K = read_split("kn", "test"); E2E = read_split("en", "test") + NEWE2
E3K = read_split("kn", "heldout"); E3E = [v for v in read_split("en", "heldout") if v not in NEW4] + NEW4
TEST = E2K + E2E + E3K + E3E
assert (len(E2K), len(E2E), len(E3K), len(E3E), len(TEST)) == (23, 11, 9, 7, 50), (len(E2K), len(E2E), len(E3K), len(E3E))
TUN = TUNE["kn"] + TUNE["en"]
assert not set(TEST) & set(TUN)
EN = set(E2E + E3E + TUNE["en"])
lang = {u: ("en" if v in EN else "kn") for v in lectures for u in lectures[v]}
vid = {u: v for v in lectures for u in lectures[v]}
GRP = [("E2 Kannada", E2K), ("E2 English", E2E), ("E3 Kannada", E3K), ("E3 English", E3E), ("All 50 test", TEST)]

LL = {}
for name, f, col in (("Gemini low", "all59_gemini_low.csv", "gemini"), ("Gemini high", "all59_gemini.csv", "gemini"), ("Sol", "all59_sol.csv", "sol")):
    rows = {r["unit_id"]: r for r in csv.DictReader(open(os.path.join(N, f), encoding="utf-8"))}
    sp = lambda s: [x for x in s.split(";") if x]
    LL[name] = dict(llm={u: sp(r[col]) for u, r in rows.items()}, csv3l={u: sp(r["improved_3l"]) for u, r in rows.items()},
                    conf3l={u: sp(r["old_3l"]) for u, r in rows.items()}, csvcue={u: sp(r["cue"]) for u, r in rows.items()})
    assert all(set(LL[name]["csvcue"][u]) == set(cue[u]) for u in rows)
# keep rule results (from the all59 reports, identical to the earlier 55/51 reports): English fill kept only for Gemini low
EN_FILL = {"Gemini low": True, "Gemini high": False, "Sol": False}
def system(name, veto, fill_kn=True, en_fill=None, gate=lambda u: True):
    llm = LL[name]["llm"]; en_fill = EN_FILL[name] if en_fill is None else en_fill
    out = {}
    for u, p in llm.items():
        if not p and cue[u] and conf[u] >= 0.5 and ((lang[u] == "kn" and fill_kn) or (lang[u] == "en" and en_fill)):
            p = cue[u]
        if lang[u] == "en" and veto(u) and gate(u):
            p = []
        out[u] = p
    return out
V = {"decoder OFF + conf 5": lambda u: not cue[u] and poff[u] >= 0.95,
     "conf 5 only": lambda u: poff[u] >= 0.95,
     "decoder OFF only": lambda u: not cue[u],
     "decoder OFF + conf >= 4": lambda u: not cue[u] and poff[u] >= 0.85,
     "conf >= 4 only": lambda u: poff[u] >= 0.85}
for name in LL:
    P = system(name, V["decoder OFF + conf 5"])
    bad = [u for u in LL[name]["csv3l"] if set(P[u]) != set(LL[name]["csv3l"][u])]
    assert not bad, (name, len(bad), bad[:3])
    LL[name]["3l"] = P
print("re-implemented 3L == stored 3L for all 3 LLMs on all", len(P), "passages")
z = np.load(FE); refs = list(z["refs"]); row = {u: i for i, u in enumerate(z["unit_ids"])}
anchors = {u: [refs[j] for j in np.flatnonzero(z["anchor"][row[u]])] for u in row}
carry_p = {}
for v in lectures:
    carry_p.update(carry(lectures[v], anchors))
S = {"Plain Carry": carry_p, "Cue": cue}
for name in LL:
    S[name] = LL[name]["llm"]; S["conf. cascade + " + name] = LL[name]["conf3l"]; S["3L + " + name] = LL[name]["3l"]
us_of = lambda vs: [u for v in vs for u in lectures[v]]
pool = lambda f, c, p, vs: f([sum(x) for x in zip(*(c(gold, p, lectures[v]) for v in vs))])
f1 = lambda p, vs: pool(f1_strict, strict_counts, p, vs)
nr = lambda p, vs: pool(f1_near, near_counts, p, vs)
th = lambda p, vs: pool(f1_strict, theme_counts, p, vs)
fv = lambda p, vs: sum(bool(p[u]) for u in us_of(vs) if not gold[u])
mv = lambda p, vs: sum(not p[u] for u in us_of(vs) if gold[u])
def ident(p, vs):
    b = [u for u in us_of(vs) if gold[u] and p[u]]; return sum(set(p[u]) == set(gold[u]) for u in b) / len(b)
det = lambda p, vs: (len(us_of(vs)) - fv(p, vs) - mv(p, vs)) / len(us_of(vs))
ok = lambda p, u: set(p) == set(gold[u])
out = []; P_ = out.append
def boot_ratio(a_p, b_p, vs, seed=0):
    a = np.array([fv(a_p, [v]) for v in vs]); b = np.array([fv(b_p, [v]) for v in vs])
    idx = np.random.default_rng(seed).integers(0, len(vs), (10000, len(vs)))
    r = b[idx].sum(1) / a[idx].sum(1)
    return b.sum() / a.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)


for n in LL:
    base = f1(system(n, V["decoder OFF + conf 5"]), TEST)
    print(n, "3L %.5f" % base, " | ".join(f"{k}: {f1(system(n, V[k]), TEST):.5f}" for k in V),
          "| nofill %.5f" % f1(system(n, V["decoder OFF + conf 5"], fill_kn=False), TEST), "| LLM %.5f" % f1(S[n], TEST))
