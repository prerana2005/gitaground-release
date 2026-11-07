"""Align a rebuilt transcript to the released passage grid (offsets matched by text alignment).

  python align_to_grid.py --text normalized/video_01.txt --video video_01 \
      --grid ../../passage_grid/units_v2.csv --fp ../../passage_grid/fingerprints_units_v2.csv --out aligned_video_01.csv

Each passage votes for where it starts in your transcript using its fingerprints; the winning offset
gives char_start/char_end. `coverage` is the share of fingerprints found at that offset: below ~0.5
treat the passage as unaligned and do not apply labels to it."""
import argparse, csv, re, statistics
from collections import Counter
from fingerprint import words, grams

csv.field_size_limit(10**9)
ap = argparse.ArgumentParser()
for k in ["--text", "--video", "--grid", "--fp", "--out"]: ap.add_argument(k, required=True)
ap.add_argument("--tol", type=int, default=6, help="word-offset tolerance for voting")
a = ap.parse_args()

text = open(a.text, encoding="utf-8").read()
spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
ws, wspan = [], []
for s, e, tok in spans:
    w = words(tok)
    if w: ws.append(w[0]); wspan.append((s, e))
index = {}
for i, h in grams(ws): index.setdefault(h, []).append(i)

grid = {r["unit_id"]: r for r in csv.DictReader(open(a.grid, encoding="utf-8")) if r["video_file"] == a.video}
rows, prev = [], 0
for r in csv.DictReader(open(a.fp, encoding="utf-8")):
    g = grid.get(r["unit_id"])
    if not g: continue
    fps = [(int(x.split(":")[0]), int(x.split(":")[1], 16)) for x in r["fingerprints"].split(";") if x]
    votes = Counter()
    for off, h in fps:
        for pos in index.get(h, []):
            if pos - off >= prev - a.tol: votes[pos - off] += 1
    if not votes:
        rows.append({"unit_id": r["unit_id"], "char_start": "", "char_end": "", "coverage": 0.0}); continue
    best = max(votes, key=lambda s: sum(c for t, c in votes.items() if abs(t - s) <= a.tol))
    hit = sum(c for t, c in votes.items() if abs(t - best) <= a.tol)
    start = max(best, 0); end = min(start + int(r["n_words_norm"]), len(wspan)) - 1
    rows.append({"unit_id": r["unit_id"], "char_start": wspan[start][0], "char_end": wspan[end][1],
                 "coverage": round(hit / max(len(fps), 1), 3)})
    if hit / max(len(fps), 1) >= 0.5: prev = start + 1
with open(a.out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["unit_id", "char_start", "char_end", "coverage"]); w.writeheader(); w.writerows(rows)
print(f"{sum(1 for r in rows if r['coverage'] >= 0.5)}/{len(rows)} passages aligned (coverage >= 0.5)")
