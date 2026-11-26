"""Sensitivity of the veto threshold on the 50 test lectures (no selection here; 0.95 was selected on tuning):
for each grid value t, veto = decoder off-verse and P_off >= t; everything else as the frozen 3L. Strict F1, false
and missed verses, paired diff vs the LLM alone (10,000 lecture resamples)."""
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    src = open("paper_numbers.py").read()
    exec(src[:src.index('P_("0. CORPUS')])
GRID = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99]
for n in LL:
    llm = S[n]
    print(f"== {n}: alone {f1(llm, TEST):.3f} / FV {fv(llm, TEST)} / missed {mv(llm, TEST)}")
    for t in GRID:
        P = system(n, lambda u, t=t: not cue[u] and poff[u] >= t)
        fires = sum(1 for u in us_of(TEST) if lang[u] == "en" and not cue[u] and poff[u] >= t)
        d, ci, _, wtl, _ = paired(gold, lectures, P, llm, TEST, 10000)
        print(f"   t={t:<5} vetoes {fires:4}  strict {f1(P, TEST):.3f}  FV {fv(P, TEST):4}  missed {mv(P, TEST):4}  "
              f"E2EN {f1(P, E2E):.3f}  diff {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {wtl[0]}/{wtl[1]}/{wtl[2]}")
