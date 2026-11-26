"""Recited (T1, T2) vs non-recited (T3-T5) share of gold verse mentions per Table II group."""
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    src = open("paper_numbers.py").read()
    exec(src[:src.index('P_("0. CORPUS')])
from collections import Counter
for gname, vs in [("Tuning KN", TUNE["kn"]), ("Tuning EN", TUNE["en"]), ("E2 KN", E2K), ("E2 EN", E2E), ("E3 KN", E3K),
                  ("E3 EN", E3E), ("Test 50", TEST), ("All 59", TUN + TEST)]:
    c = Counter(t for u in us_of(vs) for t in gold[u].values())
    n = sum(c.values()); rec = c["T1"] + c["T2"]
    print(f"{gname:10} mentions {n:5}  recited {rec:4} ({rec / n:.1%})  non-recited {n - rec:4} ({(n - rec) / n:.1%})  {dict(sorted(c.items()))}")
