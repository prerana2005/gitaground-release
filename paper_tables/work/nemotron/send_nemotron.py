"""Pilot: NVIDIA Nemotron 3 Ultra (open weights, nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16) via OpenRouter's free
endpoint labels the 3 English TUNING lectures with the SAME paste files as Sol / Gemini (sent unchanged, one call per
lecture). reasoning effort high (as Sol), max_tokens 65536 (endpoint limit), temperature/seed: API default.
An error with no answer (connection, 429, 5xx) is retried with backoff; an answer is never re-asked; a lecture whose
answer file exists is skipped. Answers + usage: review/llm_frontier/answers_openrouter_nemotron-3-ultra_high/tuning/
Usage: send_nemotron.py [lecture ...]   (default video_01 video_04 video_07)"""
import json, os, sys, time
import requests
from dotenv import load_dotenv
sys.path.insert(0, "/Users/maitreyi/Documents/Capstone/RAG-pipeline/RAG-pipeline-llama/scripts")
from llm_frontier_api import DIR, tuning_files
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
OUT = os.path.join(DIR, "answers_openrouter_nemotron-3-ultra_high", "tuning")
load_dotenv("/Users/maitreyi/Documents/Capstone/RAG-pipeline/.env", override=True)
KEY = os.environ["OPENROUTER_API_KEY"]
os.makedirs(OUT, exist_ok=True)
want = sys.argv[1:] or ["video_01", "video_04", "video_07"]
files = dict(tuning_files())
for v in want:
    if os.path.exists(os.path.join(OUT, f"{v}.txt")):
        print(f"  {v}: already answered, skipped"); continue
    prompt = open(files[v], encoding="utf-8").read()
    for i in range(8):
        t0 = time.time()
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=3600,
                              headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                              json=dict(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                        reasoning={"effort": "high"}, max_tokens=65536))
            d = r.json()
            if r.status_code != 200 or "choices" not in d:
                raise RuntimeError(f"HTTP {r.status_code}: {str(d.get('error', d))[:200]}")
            break
        except Exception as e:
            print(f"  {v}: attempt {i + 1} no answer ({e}); retrying", flush=True)
            if i == 7:
                sys.exit(f"  {v}: FAILED")
            time.sleep(min(30 * (i + 1), 180))
    c = d["choices"][0]
    content = c["message"].get("content") or ""
    open(os.path.join(OUT, f"{v}.txt"), "w", encoding="utf-8").write(content)
    json.dump(dict(model=d.get("model"), provider=d.get("provider"), finish_reason=c.get("finish_reason"),
                   native_finish_reason=c.get("native_finish_reason"), usage=d.get("usage"),
                   seconds=round(time.time() - t0, 1), reasoning_effort="high", max_tokens=65536),
              open(os.path.join(OUT, f"{v}_usage.json"), "w"), indent=1)
    u = d.get("usage") or {}
    print(f"  {v}: {d.get('model')} via {d.get('provider')}  finish {c.get('finish_reason')}  in {u.get('prompt_tokens')}  "
          f"out {u.get('completion_tokens')}  {round(time.time() - t0)}s  answer lines {content.count(chr(10)) + 1}", flush=True)
    time.sleep(10)
