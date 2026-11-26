import io, contextlib
from collections import Counter
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    src = open("paper_numbers.py").read()
    exec(src[:src.index('P_("0. CORPUS')])
for n in LL:
    a, b = S[n], S["3L + " + n]
    lost = [u for u in lectures["video_10"] if ok(a[u], u) and not ok(b[u], u)]
    print(n, len(lost), Counter(v for u in lost for v in gold[u]), Counter(t for u in lost for t in gold[u].values()))
print("video_10 gold verses:", Counter(v for u in lectures["video_10"] for v in gold[u]))
