"""Winnowed word-5-gram fingerprints: lets a rebuilt transcript be aligned to the released grid
without the grid carrying any text. Hashes are 32-bit and sampled (~1 in 4 positions), so the
text cannot be read back from them."""
import hashlib, re
N, WINDOW = 5, 4
_PUNCT = ".,;:!?\"'()[]{}|।॥-–—‘’“”"

def words(text):
    return [w for w in (t.strip(_PUNCT) for t in text.lower().split()) if w]

def _h(gram):
    return int.from_bytes(hashlib.sha1(" ".join(gram).encode("utf-8")).digest()[:4], "big")

def grams(ws):
    return [(i, _h(ws[i:i + N])) for i in range(len(ws) - N + 1)]

def winnow(ws):
    g = grams(ws); out = {}
    for s in range(max(1, len(g) - WINDOW + 1)):
        win = g[s:s + WINDOW]
        if win:
            i, h = min(win, key=lambda t: (t[1], -t[0]))
            out[i] = h
    return sorted(out.items())
