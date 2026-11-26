"""E3 English false-verse ratio CI (10,000 lecture resamples, seed 0 as boot_ratio); resamples where the LLM has
0 false verses (ratio 0/0) are undefined and dropped. Per-lecture FV counts printed."""
import io, contextlib, numpy as np
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    src = open("paper_numbers.py").read()
    exec(src[:src.index('P_("0. CORPUS')])
for n in LL:
    a = np.array([fv(S[n], [v]) for v in E3E]); b = np.array([fv(S["3L + " + n], [v]) for v in E3E])
    idx = np.random.default_rng(0).integers(0, len(E3E), (10000, len(E3E)))
    A, B = a[idx].sum(1), b[idx].sum(1); ok_ = A > 0
    r = B[ok_] / A[ok_]
    print(f"{n:12} per lecture LLM FV {list(a)} 3L FV {list(b)}  ratio x{b.sum() / a.sum():.2f} "
          f"[{np.percentile(r, 2.5):.2f}, {np.percentile(r, 97.5):.2f}]  ({(~ok_).sum()} of 10000 resamples undefined)")
print("E3E order:", E3E)
