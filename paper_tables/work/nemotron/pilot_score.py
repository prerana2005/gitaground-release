"""Pilot on the 3 English TUNING lectures (292 passages, 119 digressions): Nemotron 3 Ultra vs Sol / Gemini low / high,
GitaGround-Cue (leave-one-lecture-out, as improved3l_55.cue_all) and GitaGround-3L on each LLM with the FROZEN veto
(Cue off-verse and P_off >= 0.95) and fill (LLM NONE, Cue conf >= 0.5); keep rule for Nemotron computed here, the others'
as selected (English fill kept only for Gemini low). Nothing is tuned on these answers."""
import csv, os, sys
R = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama"
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "scripts"))
os.chdir(R)
from evalkit import load_gold
from freeze_grounding import f1_strict, strict_counts
from llm_frontier import DIR, parse
from step4_viterbi import run
import improved3l_55 as m
lectures, gold, _ = load_gold()
EN = sys.argv[1:] or ["video_01", "video_04", "video_07"]
teacher = {r["unit_id"]: min(0.98, max(0.02, float(r["soft_target"]))) for r in csv.DictReader(open(m.SILVER_LABELS, encoding="utf-8"))}
info = {}
order, _, _, _ = run(alphas=("a05",), verbose=False, resume=True, off_override=teacher, info=info)
lolo = dict(dict(order)["Viterbi (alpha 0.5)"])
import numpy as np
refs = list(np.load(m.FEATS)["refs"])
cue, conf, poff = {}, {}, {}
for v in EN:
    pv, po = info["probs_a05"][v]
    for k, u in enumerate(lectures[v]):
        cue[u], poff[u] = lolo[u], float(po[k])
        conf[u] = (1 - po[k]) * pv[k, refs.index(cue[u][0])] / pv[k].sum() if cue[u] else po[k]
N = "/Users/maitreyi/Documents/Capstone/review/paper_v7_work/new4e2"
def col(f, c):
    return {r["unit_id"]: [x for x in r[c].split(";") if x] for r in csv.DictReader(open(os.path.join(N, f), encoding="utf-8"))}
llms = {"Gemini low": col("all59_gemini_low.csv", "gemini"), "Gemini high": col("all59_gemini.csv", "gemini"), "Sol": col("all59_sol.csv", "sol")}
nem, probs = {}, []
DRY = not all(os.path.exists(os.path.join(DIR, "answers_openrouter_nemotron-3-ultra_high", "tuning", f"{v}.txt")) for v in EN)
for v in ([] if DRY else EN):
    ids = [(r["passage_id"], r["unit_id"]) for r in csv.DictReader(open(os.path.join(DIR, "tuning", f"{v}_ids.csv"), encoding="utf-8"))]
    p, pr = parse(os.path.join(DIR, "answers_openrouter_nemotron-3-ultra_high", "tuning", f"{v}.txt"), ids)
    probs.append(f"{v}: missing {len(pr['missing'])} unparsed {len(pr['unparsed'])} outside Ch2 {pr['outside_ch2']}")
    nem.update({u: p.get(u, []) for u in lectures[v]})
if DRY:
    print("DRY RUN: no Nemotron answers yet; Nemotron = Sol labels, only to check the script")
    nem = dict(llms["Sol"])
llms["Nemotron 3 Ultra"] = nem
units = [u for v in EN for u in lectures[v]]
right = lambda lab, u: set(lab.get(u, [])) == set(gold[u])
veto = [u for u in units if not cue[u] and poff[u] >= m.VETO]
def stats(lab):
    tp, fp, fn = [sum(x) for x in zip(*(strict_counts(gold, lab, lectures[v]) for v in EN))]
    dig = [u for u in units if not gold[u]]; ver = [u for u in units if gold[u]]
    both = [u for u in ver if lab.get(u)]
    det = sum((not gold[u]) == (not lab.get(u)) for u in units) / len(units)
    return (f"strict {f1_strict((tp, fp, fn)):.3f}  det {det:.3f}  FV {sum(bool(lab.get(u)) for u in dig):3}/{len(dig)}  "
            f"missed {sum(not lab.get(u) for u in ver):3}/{len(ver)}  ident {sum(set(lab[u]) == set(gold[u]) for u in both) / max(len(both), 1):.3f}")
print("answer check:", "; ".join(probs))
print(f"\nveto decides {len(veto)} passages: Cue right {sum(right(cue, u) or (not gold[u]) for u in veto)}  (veto output = NONE)")
vr = sum(not gold[u] for u in veto)
print(f"  veto right {vr}; LLM right on the same: " + ", ".join(f"{n} {sum(right(l, u) for u in veto)}" for n, l in llms.items()))
keep_fill = {"Gemini low": True, "Gemini high": False, "Sol": False}
fills = {n: [u for u in units if u not in veto and not l[u] and cue[u] and conf[u] >= m.FILL] for n, l in llms.items()}
fr = {n: (sum(right(cue, u) for u in fs), sum(right(llms[n], u) for u in fs), len(fs)) for n, fs in fills.items()}
keep_fill["Nemotron 3 Ultra"] = fr["Nemotron 3 Ultra"][0] >= fr["Nemotron 3 Ultra"][1]
print("  fill decides (Cue right vs LLM right / n): " + ", ".join(f"{n} {a} vs {b} / {c}" for n, (a, b, c) in fr.items())
      + f"  -> Nemotron fill {'KEEP' if keep_fill['Nemotron 3 Ultra'] else 'drop'}")
nveto = sum(right(nem, u) for u in veto)
print(f"  Nemotron veto keep rule: {vr} vs {nveto} -> {'KEEP' if vr >= nveto else 'drop'}")
old = {n: col(f, "old_3l") for n, f in (("Gemini low", "all59_gemini_low.csv"), ("Gemini high", "all59_gemini.csv"), ("Sol", "all59_sol.csv"))}
l2 = [u for u in units if conf[u] >= 0.9]          # old 3L layer 2 (English): Cue conf >= 0.9 -> Cue label, else the LLM
a, b = sum(right(cue, u) for u in l2), sum(right(nem, u) for u in l2)
print(f"  old 3L (confidence routing) layer Cue conf >= 0.9 decides {len(l2)}: Cue right {a} vs Nemotron {b} -> {'KEEP' if a >= b else 'drop'}")
old["Nemotron 3 Ultra"] = {u: (cue[u] if (a >= b and u in l2) else nem[u]) for u in units}
print(f"\nEnglish tuning lectures {EN}, {len(units)} passages, {sum(not gold[u] for u in units)} digressions:")
print(f"  {'GitaGround-Cue (LOLO)':24} " + stats(cue))
for n, l in llms.items():
    tl = {u: ([] if u in veto else (cue[u] if (keep_fill[n] and u in fills[n]) else l[u])) for u in units}
    print(f"  {n:24} " + stats(l)); print(f"  {'  old 3L (conf. routing)':24} " + stats(old[n])); print(f"  {'  improved 3L':24} " + stats(tl))
