"""Data for the Phase-3 report figures (no API call, stored labels only):
(1) Strict F1 and false verses with 95% lecture-bootstrap intervals (1,000 resamples) for each LLM alone and with
    GitaGround-3L on the 50 test lectures;
(2) per-passage gold / Sol / 3L + Sol labels for the English test lectures, to pick and draw a timeline.
Run from this folder; writes report_figdata.json and prints a summary."""
import io, contextlib, json, random
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    src = open("paper_numbers.py").read()
    exec(src[:src.index('P_("0. CORPUS')])
from freeze_grounding import with_ci

def ci_fv(p, vs, n_boot=1000, seed=0):
    per = [fv(p, [v]) for v in vs]
    rng = random.Random(seed)
    draws = sorted(sum(per[rng.randrange(len(per))] for _ in per) for _ in range(n_boot))
    return sum(per), (draws[int(0.025 * n_boot)], draws[min(int(0.975 * n_boot), n_boot - 1)])

out = {"ci": {}, "timeline": {}}
for n in LL:
    for tag, p in (("llm", S[n]), ("3l", S["3L + " + n])):
        pt, ci = with_ci([strict_counts(gold, p, lectures[v]) for v in TEST], f1_strict, 1000)
        f, fci = ci_fv(p, TEST)
        out["ci"][f"{n}|{tag}"] = dict(strict=round(pt, 3), strict_ci=[round(ci[0], 3), round(ci[1], 3)], fv=f, fv_ci=list(fci))
        print(f"{n:12} {tag:4} strict {pt:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  FV {f} [{fci[0]}, {fci[1]}]")

for v in E2E + E3E:
    us = lectures[v]
    a, b = S["Sol"], S["3L + Sol"]
    out["timeline"][v] = dict(gold=[gold[u] for u in us], sol=[a[u] for u in us], l3=[b[u] for u in us])
    print(f"{v:22} n={len(us):3} digr={sum(not gold[u] for u in us):3} FV sol {fv(a, [v]):3} -> 3L {fv(b, [v]):3}  "
          f"missed {mv(a, [v]):3} -> {mv(b, [v]):3}  strict {f1(a, [v]):.3f} -> {f1(b, [v]):.3f}")
json.dump(out, open("report_figdata.json", "w"))
