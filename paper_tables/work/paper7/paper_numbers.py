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

P_("0. CORPUS (lectures, passages, digressions, share, verse mentions = gold (passage, verse) pairs)")
for gname, vs in [("Tuning KN", TUNE["kn"]), ("Tuning EN", TUNE["en"]), ("E2 KN", E2K), ("E2 EN orig 7", read_split("en", "test")),
                  ("E2 EN new 4", NEWE2), ("E2 EN 11", E2E), ("E3 KN", E3K), ("E3 EN orig 3", E3E[:3]), ("E3 EN Chinmaya new 4", NEW4),
                  ("E3 EN 7", E3E), ("TEST 50", TEST), ("ALL 59", TUN + TEST)]:
    us = us_of(vs); d = sum(not gold[u] for u in us); vm = sum(len(gold[u]) for u in us)
    P_(f"  {gname:22} {len(vs):>3} lect {len(us):>5} psg  dig {d:>4} ({d / len(us):.1%})  verse psg {len(us) - d:>5}  mentions {vm}")
P_("\n1. STRICT / +-1 / THEME, per group")
for gname, vs in GRP:
    P_(f"  -- {gname} ({len(vs)} lect, {len(us_of(vs))} psg, {sum(not gold[u] for u in us_of(vs))} dig)")
    for k, p in S.items():
        P_(f"     {k:34} {f1(p, vs):.3f} {nr(p, vs):.3f} {th(p, vs):.3f}   det {det(p, vs):.3f} FV {fv(p, vs):>4} missed {mv(p, vs):>4} ident {ident(p, vs):.3f}")
P_("\n2. PAIRED strict (10,000) diff [CI] W/T/L (groups of 5+)")
pairs = [(f"3L + {n}", n) for n in LL] + [(f"3L + {n}", f"conf. cascade + {n}") for n in LL] + \
        [("Cue", "Plain Carry"), ("Sol", "Gemini high"), ("Gemini high", "Gemini low")] + [(n, "Cue") for n in LL]
for a, b in pairs:
    cells = []
    for gname, vs in GRP:
        d, ci, _, wtl, _ = paired(gold, lectures, S[a], S[b], vs, 10000)
        cells.append(f"{gname}: {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {wtl[0]}/{wtl[1]}/{wtl[2]}")
    P_(f"  {a} - {b}:  " + " | ".join(cells))
P_("\n3. FALSE-VERSE RATIO 3L / LLM (lecture bootstrap 10,000), and veto decided / right vs LLM right")
for n in LL:
    for gname, vs in GRP:
        r = boot_ratio(S[n], S["3L + " + n], vs)
        dec = [u for u in us_of(vs) if lang[u] == "en" and V["decoder OFF + conf 5"](u)]
        P_(f"  {n:12} {gname:12} FV {fv(S[n], vs):>4} -> {fv(S['3L + ' + n], vs):>4}  x{r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]  "
           f"missed {mv(S[n], vs):>4} -> {mv(S['3L + ' + n], vs):>4} | veto decides {len(dec)}: right {sum(ok([], u) for u in dec)} vs LLM {sum(ok(S[n][u], u) for u in dec)}")
P_("\n4. KANNADA FILL on test: fires, right, LLM right")
for n in LL:
    us = [u for u in us_of(TEST) if lang[u] == "kn" and not S[n][u] and cue[u] and conf[u] >= 0.5]
    ue = [u for u in us_of(TEST) if lang[u] == "en" and not S[n][u] and cue[u] and conf[u] >= 0.5 and not V["decoder OFF + conf 5"](u)]
    P_(f"  {n:12} KN fill fires {len(us)} right {sum(ok(cue[u], u) for u in us)} (LLM {sum(ok(S[n][u], u) for u in us)})"
       + (f" | EN fill fires {len(ue)} right {sum(ok(cue[u], u) for u in ue)} (LLM {sum(ok(S[n][u], u) for u in ue)})" if EN_FILL[n] else ""))
P_("\n5. ABLATIONS (English veto variants; tuning EN evidence: decides, right vs LLM right; TEST 50: strict / FV / missed, diff vs LLM)")
tu_en = us_of(TUNE["en"])
Z = z["z_a05"]
for n in LL:
    llm = S[n]
    zl = {u: max((Z[row[u], refs.index(r)] for r in llm[u] if r in refs), default=-9) for u in llm}
    nrate = {v: np.mean([not llm[u] for u in lectures[v]]) for v in lectures}
    variants = [(k, dict(veto=f)) for k, f in V.items()] + [
        ("3L without Kannada fill", dict(veto=V["decoder OFF + conf 5"], fill_kn=False)),
        ("3L + profile gate z<2", dict(veto=V["decoder OFF + conf 5"], gate=lambda u: zl[u] < 2)),
        ("3L + lecture gate NONE>=0.3", dict(veto=V["decoder OFF + conf 5"], gate=lambda u: nrate[vid[u]] >= 0.3))]
    P_(f"  == {n}: alone {f1(llm, TEST):.3f} / {fv(llm, TEST)} / {mv(llm, TEST)}")
    for k, kw in variants:
        Pv = system(n, **kw)
        gate = kw.get("gate", lambda u: True)
        fires = [u for u in tu_en if kw["veto"](u) and gate(u)]
        d, ci, _, wtl, _ = paired(gold, lectures, Pv, llm, TEST, 10000)
        P_(f"     {k:30} tuning {len(fires):>3}: {sum(ok([], u) for u in fires):>3} vs {sum(ok(llm[u], u) for u in fires):>3} | "
           f"test {f1(Pv, TEST):.3f} / {fv(Pv, TEST):>3} / {mv(Pv, TEST):>3}  {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {wtl[0]}/{wtl[1]}/{wtl[2]} | E2 EN {f1(Pv, E2E):.3f} E3 EN {f1(Pv, E3E):.3f}")
P_("\n6. ORACLE REPAIRS (fix digressions: gold digression -> NONE; fix identity: both give a verse -> gold set)")
for gname, vs in [("E2 English", E2E), ("E3 English", E3E), ("All 50 test", TEST)]:
    for k in ["Plain Carry", "Cue"] + list(LL):
        p = S[k]; us = us_of(vs)
        fixd = {u: ([] if not gold[u] else p[u]) for u in us}
        fixi = {u: (list(gold[u]) if gold[u] and p[u] else p[u]) for u in us}
        both = {u: (list(gold[u]) if gold[u] and p[u] else [] if not gold[u] else p[u]) for u in us}
        a = f1(p, vs)
        P_(f"  {gname:12} {k:12} as is {a:.3f}  fix dig {f1(fixd, vs):.3f} ({f1(fixd, vs) - a:+.3f})  fix id {f1(fixi, vs):.3f} ({f1(fixi, vs) - a:+.3f})  both {f1(both, vs):.3f}")
P_("\n7. RECALL BY MENTION TYPE (recited T1+T2 / non-recited T3-T5); n")
for gname, vs in GRP[:4]:
    n_ = [recall_ci(gold, {}, lectures, vs, ts)[1] for _, ts in GROUPS]
    P_(f"  {gname} n={n_}  " + "  ".join(f"{k}: {recall_ci(gold, S[k], lectures, vs, GROUPS[0][1])[0]:.3f}/{recall_ci(gold, S[k], lectures, vs, GROUPS[1][1])[0]:.3f}"
                                     for k in ["Plain Carry", "Cue"] + [x for n in LL for x in (n, '3L + ' + n)]))
P_("\n8. PER SPEAKER (E3), strict")
SPK = [("Sumith", [v for v in E3K if "sumith" in v]), ("Brahmanyacharya", [v for v in E3K if "brahmanya" in v]),
       ("ISKCON", [v for v in E3E if "iskcon" in v]), ("Chinmaya 03-04", [v for v in E3E if v in ("en_chinmaya_day03", "en_chinmaya_day04")]),
       ("Chinmaya 06-09", NEW4), ("Chinmaya all 6", [v for v in E3E if "chinmaya" in v])]
for sname, vs in SPK:
    us = us_of(vs); d = sum(not gold[u] for u in us)
    P_(f"  {sname:16} {len(vs)} lect dig {d}/{len(us)} ({d / len(us):.1%})  " + "  ".join(f"{k}: {f1(S[k], vs):.3f}" for k in ["Plain Carry", "Cue"] + [x for n in LL for x in (n, '3L + ' + n)]))
P_("\n9. TRADE-OFF points on TEST 50 (missed, FV) per LLM")
for n in LL:
    P_(f"  {n}: alone ({mv(S[n], TEST)},{fv(S[n], TEST)}) | decoder-gated conf5/conf>=4/any: " + " ".join(
        f"({mv(system(n, V[k]), TEST)},{fv(system(n, V[k]), TEST)})" for k in ("decoder OFF + conf 5", "decoder OFF + conf >= 4", "decoder OFF only"))
       + " | cue-only conf5/conf>=4: " + " ".join(f"({mv(system(n, V[k]), TEST)},{fv(system(n, V[k]), TEST)})" for k in ("conf 5 only", "conf >= 4 only")))
txt = "\n".join(out); print(txt)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_numbers.txt"), "w").write(txt + "\n")
