"""R@5 ceiling on the 59-lecture index: mean over scored answerable queries of min(5, |relevant|) / |relevant|;
plus median number of relevant passages per query."""
import os, sys, statistics as st
import patch59
import freeze_e7 as fe
from evalkit import load_gold
lectures, gold, _ = load_gold()
units = [u for v in sorted(lectures) for u in lectures[v]]
for q in ("e7_questions_150.csv", "e7_questions_scholar50.csv"):
    fe.TEST_Q = os.path.join(patch59.R, "data", q)
    qs = [x for x in fe.test_questions() if x["expected"]]
    rel = [sum(1 for u in units if set(gold[u]) & x["expected"]) for x in qs]
    rel = [r for r in rel if r]
    print(f"{q}: {len(rel)} scored; relevant passages per query median {st.median(rel)} min {min(rel)} max {max(rel)}; "
          f"R@5 ceiling {sum(min(5, r) / r for r in rel) / len(rel):.3f}")
