import os, sys
REPO = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
D = "/Users/maitreyi/Documents/Capstone/review/llm_frontier"
E2 = [(v, f"{D}/new4_e2/{n}_{v}_PASTE_THIS.txt") for n, v in enumerate(["video_11", "video_12", "video_13", "video_14"], 47)]
CH = [(v, f"{D}/new4/{n}_{v}_PASTE_THIS.txt") for n, v in enumerate(
    ["en_chinmaya_day06", "en_chinmaya_day07", "en_chinmaya_day08", "en_chinmaya_day09"], 43)]
import llm_frontier_api as o
o.paste_files = lambda split=None: E2 + CH
o.send("gpt-6.1-sol", "high", "heldout")
