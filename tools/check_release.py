"""Pre-release checks: placeholders left, lectures without a source URL, stray text/audio files."""
import csv, sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent; problems = []
for f in [".zenodo.json", "CITATION.cff", "README.md"]:
    if "TODO" in (root / f).read_text(encoding="utf-8"): problems.append(f"{f}: TODO placeholder")
for r in csv.DictReader(open(root / "reconstruct/sources.csv", encoding="utf-8")):
    if not r["source_url"]: problems.append(f"sources.csv: no source_url for {r['video_file']}")
    if not r["retrieved_at"]: problems.append(f"sources.csv: no retrieved_at for {r['video_file']}")
for p in root.rglob("*"):
    if p.suffix.lower() in {".mp3", ".mp4", ".wav", ".m4a"} or p.name in {"normalized", "transcripts"}:
        problems.append(f"forbidden under Route (a): {p.relative_to(root)}")
print("\n".join(problems) or "OK"); sys.exit(1 if problems else 0)
