import os, sys
REPO = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
D = "/Users/maitreyi/Documents/Capstone/review/llm_frontier"
E2 = [(v, f"{D}/new4_e2/{n}_{v}_PASTE_THIS.txt") for n, v in enumerate(["video_11", "video_12", "video_13", "video_14"], 47)]
import llm_frontier_gemini_api as g
g.paste_files = lambda split=None: E2
g.send("gemini-3.1-pro-preview", "low", "heldout")
