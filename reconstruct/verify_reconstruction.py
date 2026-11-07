"""Compare a rebuilt units CSV (with text) against the released grid (hashes only)."""
import argparse, csv, hashlib, sys
csv.field_size_limit(10**9)
ap = argparse.ArgumentParser(); ap.add_argument("--units", required=True); ap.add_argument("--grid", required=True)
a = ap.parse_args()
grid = {r["unit_id"]: r for r in csv.DictReader(open(a.grid, encoding="utf-8"))}
ok = bad = 0; seen = set()
for r in csv.DictReader(open(a.units, encoding="utf-8")):
    g = grid.get(r["unit_id"]); seen.add(r["unit_id"])
    h = hashlib.sha256(r["text"].encode("utf-8")).hexdigest()
    if g and h == g["text_sha256"] and r["char_start"] == g["char_start"] and r["char_end"] == g["char_end"]: ok += 1
    else: bad += 1
missing = len(set(grid) - seen)
print(f"match={ok} mismatch={bad} missing={missing}")
sys.exit(0 if bad == 0 and missing == 0 else 1)
