import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    src = open("paper_numbers.py").read()
    exec(src[:src.index('P_("0. CORPUS')])
for n in LL:
    k = "conf. cascade + " + n; p = S[k]; us = us_of(TEST)
    fixd = {u: ([] if not gold[u] else p[u]) for u in us}; fixi = {u: (list(gold[u]) if gold[u] and p[u] else p[u]) for u in us}
    a = f1(p, TEST)
    print(f"All 50 {k:26} as is {a:.3f} fixdig {f1(fixd, TEST):.3f} ({f1(fixd, TEST) - a:+.3f}) fixid {f1(fixi, TEST):.3f} ({f1(fixi, TEST) - a:+.3f})")
