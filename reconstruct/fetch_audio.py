"""Download lecture audio listed in sources.csv (rows with a source_url). Output is git-ignored."""
import csv, subprocess, sys
from pathlib import Path
here = Path(__file__).parent
out = here / "audio"; out.mkdir(exist_ok=True)
for r in csv.DictReader(open(here / "sources.csv", encoding="utf-8")):
    if not r["source_url"]:
        print("no URL for", r["video_file"]); continue
    subprocess.run([sys.executable, "-m", "yt_dlp", "-x", "--audio-format", "mp3",
                    "-o", str(out / (r["video_file"] + ".%(ext)s")), r["source_url"]], check=False)
