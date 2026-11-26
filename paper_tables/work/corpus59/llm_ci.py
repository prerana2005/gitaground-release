"""Strict [1,000-resample lecture CI], +-1, theme for the three LLMs on E2 KN (23) and E2 EN (11), from all59_*.csv"""
import csv, os
import patch59
from evalkit import load_gold, read_split
from freeze_grounding import f1_near, f1_strict, near_counts, strict_counts, theme_counts, with_ci
lectures, gold, _ = load_gold()
N = os.path.join(patch59.W, "new4e2")
def col(f, c):
    return {r["unit_id"]: [x for x in r[c].split(";") if x] for r in csv.DictReader(open(os.path.join(N, f), encoding="utf-8"))}
for name, f, c in (("Gemini low", "all59_gemini_low.csv", "gemini"), ("Gemini high", "all59_gemini.csv", "gemini"), ("Sol", "all59_sol.csv", "sol")):
    p = col(f, c); out = []
    for L in ("kn", "en"):
        vs = read_split(L, "test")
        cells = []
        for counts, f1 in ((strict_counts, f1_strict), (near_counts, f1_near), (theme_counts, f1_strict)):
            pt, ci = with_ci([counts(gold, p, lectures[v]) for v in vs], f1, 1000)
            cells.append(f"{pt:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if f1 is f1_strict and counts is strict_counts else ""))
        out.append(f"{L} ({len(vs)}): " + " / ".join(cells))
    print(f"{name:12} " + "   ".join(out))
