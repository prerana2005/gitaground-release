"""freeze_grounding --split test --v2 on E2 with the 11 English lectures, plus the Local 145-state decoder
(step4_viterbi.run(resume=True) fit on the 9 tuning lectures, as v2_teacher145 without the teacher P_off).
Usage: grounding59.py [--old]"""
import os, sys
import patch59
import step4_viterbi, freeze_grounding as fg
patch59.patch(fg)
L145 = os.path.join(patch59.SP, "local145.pkl")
if not os.path.exists(L145):
    step4_viterbi.run(alphas=("a05",), verbose=False, save=L145, resume=True)
# v2_local73 (student logit) is not in the paper and has no student score for new passages: dropped in the 59 run
fg.V2 = tuple(x for x in fg.V2 if not (patch59.NEW and x[1] == "v2_local73")) + (("local scorer 145", L145[:-4]),)
if patch59.NEW:
    fg.COMPARE = [c for c in fg.COMPARE if "v2 local 73 (old+student)" not in c]
sys.argv = [sys.argv[0], "--split", "test", "--v2"]
patch59.run_main(fg, 'out = os.path.join(RES, f"freeze_{\'v2\' if args.v2 else \'grounding\'}_{args.split}.txt")',
                 'out = os.path.join(SP, f"freeze_v2_test_{TAG}.txt")')
