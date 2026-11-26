"""gpt-5-mini end to end (scripts/llm_e2e.py, unchanged prompt, effort low) on the 4 new E2 English lectures, then the
score on E2 with 11 English lectures. Answers go to review/llm_e2e/gpt-5-mini/ like the other 42; csv/txt written here.
Usage: run_mini.py run | score [--old]"""
import re, sys
import patch59
import llm_e2e
llm_e2e.read_split = patch59.read_split
step = sys.argv[1]
if step == "run":
    llm_e2e.all_lectures = lambda: list(patch59.NEWE2)
    llm_e2e.run("gpt-5-mini", "low", "all")
else:
    src = open(llm_e2e.__file__, encoding="utf-8").read()
    fn = re.search(r"\ndef score\(.*?(?=\n\n\ndef main)", src, re.S).group(0)
    a = 'os.path.join(HERE, "results", f"llm_e2e_{folder(model)}'
    assert fn.count(a) == 2
    fn = fn.replace(a, 'os.path.join(SP, f"llm_e2e_" + TAG + f"_{folder(model)}')
    g = vars(llm_e2e); g["SP"] = patch59.SP; g["TAG"] = "59" if patch59.NEW else "old"
    exec(fn, g)
    llm_e2e.score("gpt-5-mini")
