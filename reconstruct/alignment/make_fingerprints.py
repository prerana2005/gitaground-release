"""(Used to build passage_grid/fingerprints_*.csv from the original units; needs passage text.)"""
import csv, sys
from fingerprint import words, winnow
csv.field_size_limit(10**9)
src, dst = sys.argv[1:3]
with open(dst, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["unit_id", "n_words_norm", "fingerprints"])
    for r in csv.DictReader(open(src, encoding="utf-8")):
        ws = words(r["text"]); fp = winnow(ws)
        w.writerow([r["unit_id"], len(ws), ";".join(f"{i}:{h:08x}" for i, h in fp)])
