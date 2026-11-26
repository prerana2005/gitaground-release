"""improved3l_55.main on ALL lectures: the 55 + video_11-14 (new E2 English), Sol now also on the 4 Chinmaya.
Changes vs improved3l_55 (repo files untouched): FEATS = scratch step2_features_full.npz (existing rows identical),
teacher P_off for the new passages from the scratch teacher file, new groups/sets, outputs to scratch.
Usage: allsys.py gemini|sol"""
import builtins, csv, os, re, sys
R = "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama"
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "scripts"))
from evalkit import load_csv, load_gold
from llm_frontier import DIR, parse
import gemini_runs_55, new4_gemini3l, step4_viterbi
import improved3l_55 as m
NEWE2 = ["video_11", "video_12", "video_13", "video_14"]
E2DIR = os.path.join(DIR, "new4_e2")
FE = os.path.join(SP, "step2_features_full.npz")
step4_viterbi.FEATS = m.FEATS = FE
new4_gemini3l.SILVER = os.path.join(SP, "silver")
_g = gemini_runs_55.groups
groups = lambda: _g() + [("New E2 English (video_11-14)", NEWE2)]
m.groups = groups
lectures, _, _ = load_gold()
ids = lambda v: [(r["passage_id"], r["unit_id"]) for r in load_csv(os.path.join(E2DIR if v in NEWE2 else new4_gemini3l.NEW_DIR, f"{v}_ids.csv"))]
problems = []
def labels(folder, vs):
    out = {}
    for v in vs:
        p, pr = parse(os.path.join(folder, f"{v}.txt"), ids(v))
        problems.append(f"{os.path.basename(folder)} {v}: missing {len(pr['missing'])} unparsed {len(pr['unparsed'])} outside Ch2 {pr['outside_ch2']}")
        out.update({u: p.get(u, []) for u in lectures[v]})       # missing passage = NONE, as the 42
    return out
which = sys.argv[1]
if which in ("gemini", "gemini_low"):
    lvl = "high" if which == "gemini" else "low"
    base = os.path.join(os.path.dirname(SP), "gemini_high_55.csv") if lvl == "high" else os.path.join(R, "results/gemini_runs/gita_data.csv")
    gcsv = os.path.join(SP, f"gemini_{lvl}_59.csv")
    extra = labels(os.path.join(DIR, f"answers_api_gemini-3.1-pro-preview_{lvl}"), NEWE2)
    with open(gcsv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["unit_id", "video_file", "refs"])
        for r in csv.DictReader(open(base, encoding="utf-8")):
            w.writerow([r["unit_id"], r["video_file"], r["refs"]])
        for v in NEWE2:
            for u in lectures[v]:
                w.writerow([u, v, ";".join(extra[u])])
else:
    gcsv = None
    _ll = m.llm_labels
    sol_extra = labels(os.path.join(DIR, "answers_gpt-6.1-sol"), gemini_runs_55.NEW4 + NEWE2)
    m.llm_labels = lambda *a: {**_ll(*a), **sol_extra}
src = open(m.__file__, encoding="utf-8").read()
for fname in ("cue_all", "main"):
    fn = re.search(rf"\ndef {fname}\(.*?(?=\n\n\ndef |\n\n\nif __name__)", src, re.S).group(0)
    reps = [("for v in NEW4 for u in lectures[v]", "for v in NEW4 + NEWE2 for u in lectures[v]")] if fname == "cue_all" else [
        ('G = [(g, vs) for g, vs in groups() if not (which == "sol" and vs == NEW4)]', "G = groups()"),
        ('    assert len(allv) == (55 if which == "gemini" else 51), len(allv)\n', ""),
        ('("All 42 test", test),', '("All 42 test", test), ("42 test + new E2 4 (46)", test + NEWE2), '
         '("All test: 42 + Chinmaya 4 + new E2 4 (50)", test + NEW4 + NEWE2), '
         '("All E2 English test", G[3][1] + NEWE2), ("All E3 English", G[5][1] + NEW4),'),
        ('stem = "improved3l_55" if which == "gemini" else "improved3l_sol51"', 'stem = "all59_" + OUTTAG'),
        ('os.path.join(HERE, "results", f"{stem}', 'os.path.join(SP, f"{stem}')]
    for a, b in reps:
        assert a in fn, (fname, a)
        fn = fn.replace(a, b)
    g = vars(m); g.update(NEWE2=NEWE2, SP=SP, OUTTAG=which)
    exec(fn, g)
m.main("sol" if which == "sol" else "gemini", gcsv)
print("\nanswer check:\n  " + "\n  ".join(problems))
