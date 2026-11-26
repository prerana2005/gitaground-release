"""run_freeze.py, but GPT-4o-mini routing is not called (OpenAI credits exhausted): every route = NONE, so the
GPT-4o-mini row of this output is NOT a result and is not reported; all other rows are unaffected."""
import runpy, sys
sys.argv = ["run_freeze.py", "e7_questions_scholar50.csv"]
import os
sys.path.insert(0, "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama"); sys.path.insert(0, "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama/scripts")
import freeze_e7
freeze_e7.llm_routes = lambda texts: {t: "NONE" for t in texts}
runpy.run_path("/private/tmp/claude-501/-Users-maitreyi-Documents-Capstone/36440e69-87ca-48a6-a581-537a9d90af22/scratchpad/e7s50/run_freeze.py", run_name="__main__")
