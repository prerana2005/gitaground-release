# paper_tables/ — the 59-lecture numbers of the paper

Everything here reproduces the paper's tables and figures on the 59-lecture corpus (9 tuning, 50 test lectures).
`work/` mirrors the working folder the numbers were computed in, so the scripts find their inputs by the same relative
paths. `llm_frontier/` holds the raw whole-lecture LLM answers.

No transcript or passage text is included. Scripts that need text (BM25, the retrieval index, new embeddings) require
the passages rebuilt with `reconstruct/`.

## Running

The scripts call the repository code in `code/` and `code/scripts/` and were run with the versions in
`code/requirements.txt` (Python 3.13). Each script sets two paths at the top: `R`, the code folder with its `data/` and
`results/` (point it at `code/` with the reconstructed `data/`), and `W` / `SP` / `N`, this `work/` folder. No other
change is needed. Scripts marked *API* call a paid model and are included as the record of how labels were produced;
every other script reads stored labels only.

## Labels (59 lectures, 2,583 passages)

| File | Contents |
|---|---|
| `work/new4e2/gemini_low_59.csv`, `gemini_high_59.csv` | Gemini 3.1 Pro whole-lecture labels, thinking level low / high |
| `../frozen/sol_labels_59.csv` | GPT-6.1 Sol (high) whole-lecture labels |
| `work/new4e2/all59_gemini_low.csv`, `all59_gemini.csv` (high), `all59_sol.csv` | per passage: LLM label, GitaGround-Cue label, confidence-routing cascade (`old_3l`), GitaGround-3L (`improved_3l`) and the layer that decided |
| `work/new4e2/silver/v2_window_low_gpt-5-mini.jsonl` | discourse cue (gpt-5-mini, effort low): label and confidence for every passage |
| `work/corpus59/llm_e2e_59_gpt-5-mini.csv` | gpt-5-mini whole-lecture labels (E2 lectures) |
| `work/new4e2/step2_features_full.npz` | the eight evidence features for all 2,583 passages (first 1,884 identical to `embeddings_features/step2_features.npz`) |
| `work/new4e2/unit_embeddings_full.npz`, `work/corpus59/embed_scores_new400.npz` | BGE-M3 embeddings and similarity scores for the passages added in v1.1.0 |
| `work/corpus59/local145.pkl` | GitaGround-Local with 145 states (Table XVIII) |
| `llm_frontier/answers_*` | raw LLM answers, one file per lecture, with `*_usage.json`; `*_ids.csv` map passage numbers to unit ids |

## Tables and figures

| Paper | Script | Output |
|---|---|---|
| Table III (benchmark), recited share | `work/paper7/paper_numbers.py`, `recited_share.py` | `paper_numbers.txt` §0, `recited_share.txt`, `corpus_stats.txt` |
| Table V (E1 agreement) | `code/scripts/exp_e1_iaa.py` | `frozen/exp_e1_iaa.txt` |
| Fig. 3 (veto threshold on tuning), Table XIII (ablations) | `work/paper7/paper_numbers.py` | `paper_numbers.txt` §5 (the criterion at each threshold is the net gain of the cue-confidence level it admits) |
| Table VII, Fig. 4 (all 50 test lectures) | `work/paper7/paper_numbers.py`, `report_figdata.py` | `paper_numbers.txt` §1–3, `report_figdata.json` (intervals) |
| Fig. 5 (the 470 vetoed passages), Kannada fill | `work/paper7/paper_numbers.py` | `paper_numbers.txt` §3–4 |
| Table VIII (detection vs identification) | `work/paper7/paper_numbers.py` | `paper_numbers.txt` §1 |
| Table IX, Fig. 6 (oracle repairs) | `work/paper7/paper_numbers.py`, `v8_extra.py` | `paper_numbers.txt` §6, `v8_extra.txt` |
| Table X, Table XV (paired differences) | `work/paper7/paper_numbers.py`; `work/corpus59/grounding59.py` (Cue vs Plain Carry) | `paper_numbers.txt` §2, `work/corpus59/freeze_v2_test_59.txt` |
| Table XI (false and missed verses) | `work/paper7/paper_numbers.py`, `e3en_ratio.py` | `paper_numbers.txt` §3, `e3en_ratio.txt` |
| Fig. 7 (trade-off) | `work/paper7/paper_numbers.py` | `paper_numbers.txt` §9 |
| Fig. 8 (two vetoed passages) | — | passage ids in the figure; text from the reconstructed passages |
| Table XII (threshold sensitivity on test) | `work/paper7/veto_sensitivity.py` | `veto_sensitivity.txt` |
| Table XIV (all systems, E2) | `work/corpus59/grounding59.py`, `llm_ci.py`, `run_mini.py` (*API*) | `freeze_v2_test_59.txt`, `llm_ci.txt`, `llm_e2e_59_gpt-5-mini.txt` |
| Table XVI (E3 per speaker) | `work/paper7/paper_numbers.py` | `paper_numbers.txt` §8 |
| Table XVII (recall by mention type) | `work/paper7/paper_numbers.py` | `paper_numbers.txt` §7 |
| Table XVIII (E6, decoder variants), E5 | `work/corpus59/e6_59.py`, `cross_pk.py` | `freeze_e6_test_59.txt`, `cross_pk.txt` |
| E4 (unseen verses) | `code/scripts/freeze_e4.py` | `frozen/freeze_e4_test.txt` |
| Table XIX (E7, deployed index) | `work/e7s50/app59.py`, `work/corpus59/e7_bm25.py`, `r5_ceiling.py` | `app59_e7_questions_150.log`, `app59_e7_questions_scholar50.log`, `e7_bm25.txt`, `r5_ceiling.txt` |
| E7 abstention threshold (development queries) | `code/scripts/freeze_e7.py` | `frozen/freeze_e7_dev_dev.txt` |
| Table XX (E7 by label source) | `work/corpus59/labelsrc59.py` | `ls150_all.log`, `ls150_testonly.log` |
| Decoding time | `work/corpus59/time_decode.py` | `time_decode.txt` |
| Nemotron 3 Ultra pilot | `work/nemotron/send_nemotron.py` (*API*), `pilot_score.py` | `pilot_video_01.txt`, `pilot_video_01_answered55.txt` |
| Hybrid + cross-encoder reranker (BGE-reranker-v2-m3) | `work/corpus59/e7_rerank.py` | `e7_rerank.txt` |

How the labels were produced: `work/new4e2/send_llms.py`, `send_sol.py`, `send_gemini_low.py` (*API*, whole-lecture
labels), `teacher_new.py` (*API*, discourse cue), `feats_new.py` (features), `allsys.py` (cascade on all 59 lectures;
writes `all59_*.csv/.txt`).

## Not yet scripted here

These numbers in the paper were computed with short scripts that were not kept; they will be added:
cue calibration on unseen speakers (flag rates and AUC, Section VI-G), the 16 of 21 mid-lecture returns (Section IV-A),
LLM call times and costs (Section V), and the fusion variant's tuning F1 (Table XIII).
