"""freeze_e6 --split test on E2 with the 11 English lectures. Usage: e6_59.py [--old]"""
import sys
import patch59
import freeze_e6
patch59.patch(freeze_e6)
sys.argv = [sys.argv[0], "--split", "test"]
patch59.run_main(freeze_e6, 'out = os.path.join(RES, f"freeze_e6_{args.split}.txt")',
                 'out = os.path.join(SP, f"freeze_e6_test_{TAG}.txt")')
