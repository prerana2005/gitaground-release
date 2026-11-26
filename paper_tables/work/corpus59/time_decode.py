import time, os, platform, sys
import patch59
from evalkit import load_gold
lectures, _, _ = load_gold()
path = os.path.join(patch59.R, "results", "v2_teacher145.pkl")
patch59.predict_locked({v: lectures[v] for v in list(lectures)[:2]}, "a05", path=path)   # warm-up
t0 = time.perf_counter(); patch59.predict_locked({v: lectures[v] for v in lectures}, "a05", path=path); dt = time.perf_counter() - t0
import sklearn, numpy
print(f"GitaGround-Cue decoding, all {len(lectures)} lectures: {dt:.2f} s total, {dt / len(lectures) * 1000:.0f} ms per lecture")
print("python", sys.version.split()[0], "numpy", numpy.__version__, "sklearn", sklearn.__version__, platform.machine())
