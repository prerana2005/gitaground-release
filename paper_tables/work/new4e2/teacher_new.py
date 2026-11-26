"""gpt-5-mini teacher, config B (window, low) as silver_v2.run, on video_11-14 only.
Answers go to a scratch copy of results/silver/v2_window_low_gpt-5-mini.jsonl (repo file not changed)."""
import os, shutil, sys
REPO = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama"
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
NEW = ["video_11", "video_12", "video_13", "video_14"]
d = os.path.join(SP, "silver"); os.makedirs(d, exist_ok=True)
dst = os.path.join(d, "v2_window_low_gpt-5-mini.jsonl")
if not os.path.exists(dst):
    shutil.copy(os.path.join(REPO, "results/silver/v2_window_low_gpt-5-mini.jsonl"), dst)
import silver_v2
silver_v2.OUT_DIR = d
silver_v2.read_split = lambda L, s: NEW if L == "en" else []
silver_v2.run("window", "low", "gpt-5-mini", 6, split="dev")
