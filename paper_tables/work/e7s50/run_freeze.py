"""freeze_e7.py --index all --questions test, unchanged, on data/e7_questions_scholar50.csv; LLM route cache and the
output file redirected to this folder (results/freeze_e7_all_test.txt and results/e7_llm_routes.json untouched)."""
import builtins, os, sys
R = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama"
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "scripts"))
import gitaground
OLD = os.path.join(os.path.dirname(D), "units_v2_before_video11-14.csv")   # the frozen 51-lecture grid
_uc = gitaground.unit_cache
gitaground.UNITS = OLD
gitaground.unit_cache = lambda p: os.path.join(R, "results", "unit_embeddings.npz") if p == OLD else _uc(p)
import freeze_e7 as fe
Q = sys.argv[1] if len(sys.argv) > 1 else "e7_questions_scholar50.csv"
TAG = "scholar50" if "scholar50" in Q else "check150"
fe.TEST_Q = os.path.join(R, "data", Q)
fe.LLM_CACHE = os.path.join(D, "e7_llm_routes.json")
def redirect(path, mode="r", *a, **k):
    if "w" in mode and os.path.basename(str(path)).startswith("freeze_e7_"):
        path = os.path.join(D, f"freeze_e7_all_{TAG}.txt")
    return builtins.open(path, mode, *a, **k)
fe.open = redirect
fe.input = lambda *a: "yes"   # the script asks before a test run
from dotenv import load_dotenv
load_dotenv("/Users/maitreyi/Documents/Capstone/RAG-pipeline/.env", override=True)
sys.argv = ["freeze_e7.py", "--index", "all", "--questions", "test"]
fe.main()
