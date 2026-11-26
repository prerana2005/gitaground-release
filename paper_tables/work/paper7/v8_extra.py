"""Extra numbers for paper v8: per-lecture 3L - LLM on English test lectures; oracle repairs for the 3L rows;
short-passage check."""
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    src = open("paper_numbers.py").read()
    exec(src[:src.index('P_("0. CORPUS')])
from evalkit import load_csv, UNITS, UNITS_HELDOUT
print("== per-lecture strict F1, 3L - LLM, English test lectures (passages whose label changed; FV removed; verse passages lost)")
for n in LL:
    rows = []
    for v in E2E + E3E:
        a, b = S[n], S["3L + " + n]
        d = f1(b, [v]) - f1(a, [v])
        ch = sum(set(a[u]) != set(b[u]) for u in lectures[v])
        fvr = sum(bool(a[u]) and not b[u] for u in lectures[v] if not gold[u])
        lost = sum(ok(a[u], u) and not ok(b[u], u) for u in lectures[v])
        rows.append((v, d, ch, fvr, lost, sum(not gold[u] for u in lectures[v]) / len(lectures[v])))
    W = [r for r in rows if r[1] > 1e-9]; L = [r for r in rows if r[1] < -1e-9]
    print(f"  {n}: wins {len(W)} mean {sum(r[1] for r in W) / max(len(W), 1):+.3f}, losses {len(L)} mean {sum(r[1] for r in L) / max(len(L), 1):+.3f}")
    for r in sorted(rows, key=lambda r: r[1]):
        print(f"     {r[0]:20} {r[1]:+.3f}  changed {r[2]:3}  FV removed {r[3]:3}  right->wrong {r[4]:3}  digr share {r[5]:.2f}")
print("== oracle repairs incl. 3L rows")
for gname, vs in [("E2 English", E2E), ("All 50 test", TEST)]:
    for k in ["Plain Carry", "Cue"] + [x for n in LL for x in (n, "3L + " + n)]:
        p = S[k]; us = us_of(vs)
        fixd = {u: ([] if not gold[u] else p[u]) for u in us}
        fixi = {u: (list(gold[u]) if gold[u] and p[u] else p[u]) for u in us}
        a = f1(p, vs)
        print(f"  {gname:12} {k:18} as is {a:.3f} fixdig {f1(fixd, vs):.3f} ({f1(fixd, vs) - a:+.3f}) fixid {f1(fixi, vs):.3f} ({f1(fixi, vs) - a:+.3f})  det {det(p, vs):.3f} FV {fv(p, vs)} miss {mv(p, vs)} ident {ident(p, vs):.3f}")
grid = {u["unit_id"]: u for u in load_csv(UNITS) + load_csv(UNITS_HELDOUT)}
short = [(u, int(grid[u]["n_words"]), lectures[vid[u]].index(u) == len(lectures[vid[u]]) - 1) for v in lectures for u in lectures[v] if int(grid[u]["n_words"]) < 50]
print("== passages < 50 words:", len(short), "of which lecture-final:", sum(s[2] for s in short), short[:8])
