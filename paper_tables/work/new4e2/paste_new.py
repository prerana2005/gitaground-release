"""Whole-lecture paste files for video_11-14, built exactly as the 42 (new4_gemini3l.paste_text = llm_frontier
format); first rebuilds 30_video_10 and stops unless byte-identical to the saved file."""
import csv, os, sys
REPO = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from evalkit import UNITS, UNITS_HELDOUT, load_csv, load_gold
from llm_frontier import DIR
from llm_pilot import catalogue
from new4_gemini3l import paste_text
NEW = ["video_11", "video_12", "video_13", "video_14"]
OUT = os.path.join(DIR, "new4_e2")
lectures, _, _ = load_gold()
units = {u["unit_id"]: u for u in load_csv(UNITS) + load_csv(UNITS_HELDOUT)}
cat = catalogue()
for n, v in ((30, "video_10"), (24, "video_02")):
    t, _ = paste_text(v, lectures, units, cat)
    assert t == open(os.path.join(DIR, f"{n}_{v}_PASTE_THIS.txt"), encoding="utf-8").read(), v
print("video_10 and video_02 rebuilt byte-identical")
os.makedirs(OUT, exist_ok=True)
for n, v in enumerate(NEW, start=47):
    t, ids = paste_text(v, lectures, units, cat)
    open(os.path.join(OUT, f"{n}_{v}_PASTE_THIS.txt"), "w", encoding="utf-8").write(t)
    with open(os.path.join(OUT, f"{v}_ids.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("passage_id", "unit_id")] + ids)
    print(v, len(ids), "passages", len(t), "chars")
