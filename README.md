# GitaGround release

Companion release to *Where Is the Verse? A Benchmark and a Structured Decoder with LLM Discourse Cues for Grounding Oral Exegesis to Canonical Texts* (Maitreyi, Prerana M P, Meghana S, Meghana Gajendran, Badri Prasad; PES University).


Labels, passage grid, embeddings and code for *verse grounding in oral discourse* (Bhagavad Gita, Chapter 2):
51 lectures, 6 speakers, three traditions, 1,884 passages in Kannada and English.

**Route (a).** No lecture audio and no transcript or passage text is distributed. The release has URLs, offsets,
text hashes, code, labels and pre-computed embeddings. `reconstruct/` explains how to rebuild the text yourself.

DOI: [10.5281/zenodo.24037830](https://doi.org/10.5281/zenodo.24037830)

## Contents (Section 8)

| path | contents |
|---|---|
| `catalogue/` | lecture catalogue (speaker, tradition, source URL, split, fold, hash of normalized text), mention-type table |
| `themes/verse_themes.csv` | verse to theme map (72 verses, 9 themes) |
| `catalogue/verse_catalogue.csv` | the 72 verses (Devanagari mūla, Kannada script) with theme; no translations |
| `passage_grid/` | `units_v1`, `units_v2`, `units_heldout_v1`: ids, offsets, word counts, `text_sha256`; **no text** |
| `labels/gold/` | gold mention labels (`mention_spans.csv`), per-annotator files, shloka-level annotations |
| `labels/second_pass/` | second-pass (IAA) labels and assignments |
| `labels/adjudication/` | expert adjudication of disagreements |
| `labels/llm/` | all LLM labels: silver labels, GPT-4o-mini / GPT-5-mini / Opus / Gemini runs, pilot ids, RAG-QA runs (answers removed) |
| `embeddings_features/` | BGE-M3 passage embeddings (1024-d), passage x verse similarity scores, step-2 features, student scores |
| `frozen/` | frozen system files, predictions and result tables (`results_v3`, `results_v4`, `freeze_*.txt`, locked scorers) |
| `experiment_log/` | `experiment_log.md`, one row per run with commit and config |
| `eval/` | question sets used for retrieval evaluation |
| `reconstruct/` | `sources.csv` (URLs, ASR settings), `asr_segment_timings.csv`, ASR / segmentation / alignment scripts, `verify_reconstruction.py` |
| `code/` | all experiment, detection, retrieval and app code |
| `tools/check_release.py` | pre-release checks |

## What is not here

Audio, transcripts (`data/transcripts*`, `data/normalized`), passage text, per-lecture exports, chunk files.
Lecture rights belong to the speakers and publishers. Please fetch lectures from the source URLs.

## Licence

Code: MIT (`LICENSE`). Labels and data: CC BY-NC 4.0 (`LICENSE-DATA`).

## Update, 21 Nov 2026 (after the paper snapshot)

Everything above describes the 51-lecture paper release; those files are unchanged unless listed here.
- **8 more lectures** (59 in all, 2,583 passages): `video_11`..`video_14` (original English speaker, E2 test) and
  `en_chinmaya_day06`..`day09` (Swami Aparajitananda, held-out). Added to `catalogue/lecture_catalogue.csv`,
  `reconstruct/sources.csv`, `passage_grid/` (with fingerprints) and gold labels (`labels/gold/`). Earlier rows are unchanged.
- **New runs and results** (experiment log rows from 7 Nov on): GPT-6.1 Sol and Gemini 3.1 Pro (API) labels in `labels/llm/`;
  cascade / hybrid, headroom, recall-by-type and fine-tuning CV results in `frozen/`; code in `code/scripts/` and `code/gg_finetune/`.
  `frozen/README.md` is a guide to the result files.
- **59-lecture rebuilds kept beside the paper's frozen files:** `frozen/step5_index_all_59.csv`,
  `frozen/step5_segments_all_59.csv`, `embeddings_features/step2_features_59.npz`. The meaning-profile features
  of the earlier passages differ in the 59-lecture rebuild, so the paper's `step2_features.npz` and `step5_*` stay as they were.
  `unit_embeddings*.npz` now cover all 59 lectures (vectors of earlier passages identical).
- Sarvam ASR script: `reconstruct/asr/transcribe_new.py` (model `saaras:v3`).

## Notes for users

- Unit-level timestamps do not exist for most passages; `time_start`/`time_end` are empty in the grid.
  `reconstruct/asr_segment_timings.csv` has Whisper segment timings for `video_01`..`video_10` only.
- `frozen/*.pkl` are scikit-learn pickles. Load only from this repository, with the versions in `code/requirements.txt`.
- Annotators are the authors plus one domain expert, as in the paper. Data statement and datasheet: `DATASHEET.md`.
- English verse translations used for alpha > 0 profiles are under copyright and not included; see `code/verse_translations.NOTE.md`.
