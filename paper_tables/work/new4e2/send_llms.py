"""Gemini 3.1 Pro high on video_11-14; GPT-6.1 Sol high on video_11-14 + en_chinmaya_day06-09.
Same send functions and settings as the 42; answers into the existing answer folders (new file names only)."""
import os, sys, threading
REPO = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
D = "/Users/maitreyi/Documents/Capstone/review/llm_frontier"
E2 = [(v, f"{D}/new4_e2/{n}_{v}_PASTE_THIS.txt") for n, v in enumerate(["video_11", "video_12", "video_13", "video_14"], 47)]
CH = [(v, f"{D}/new4/{n}_{v}_PASTE_THIS.txt") for n, v in enumerate(
    ["en_chinmaya_day06", "en_chinmaya_day07", "en_chinmaya_day08", "en_chinmaya_day09"], 43)]
assert all(os.path.exists(p) for _, p in E2 + CH)
import llm_frontier_gemini_api as g
import llm_frontier_api as o
g.paste_files = lambda split=None: E2
o.paste_files = lambda split=None: E2 + CH
t = threading.Thread(target=g.send, args=("gemini-3.1-pro-preview", "high", "heldout"))
t.start()
o.send("gpt-6.1-sol", "high", "heldout")
t.join()
