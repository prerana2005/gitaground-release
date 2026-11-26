# GitaGround release

Companion release to *GitaGround: A Bilingual Benchmark and Structured Digression Veto for Canonical Verse Grounding
in Spoken Lectures* (Maitreyi V, Prerana M P, Meghana S, Meghana Gajendran, Badri Prasad V R; PES University).

Labels, passage grid, embeddings and code for *Canonical Verse Grounding* (Bhagavad Gītā, Chapter 2):
59 lectures (38 Kannada, 21 English) by six speakers in three traditions, 2,583 passages.

**Route (a).** No lecture audio and no transcript or passage text is distributed. The release has URLs, offsets,
text hashes, code, labels and pre-computed embeddings. `reconstruct/` explains how to rebuild the text yourself.

DOI (all versions, always the latest): [10.5281/zenodo.24037829](https://doi.org/10.5281/zenodo.24037829)
(each version also has its own DOI on Zenodo).

## Contents

| path | contents |
|---|---|
| `catalogue/` | lecture catalogue (speaker, tradition, source URL, split, fold, hash of normalized text), lecture metadata, mention-type table |
| `themes/verse_themes.csv` | verse to theme map (72 verses, 9 themes) |
| `catalogue/verse_catalogue.csv` | the 72 verses (Devanagari mūla, Kannada script) with theme; no translations |
| `passage_grid/` | `units_v1`, `units_v2`, `units_heldout_v1`: ids, character offsets, word counts, `text_sha256`; **no text** |
| `labels/gold/` | gold mention labels (`mention_spans.csv`), per-annotator files, shloka-level annotations |
| `labels/second_pass/` | blind second-pass (E1) labels and assignments |
| `labels/adjudication/` | adjudication of the 27 second-pass disagreements |
| `labels/llm/` | LLM labels from the earlier runs (42 and 51 lectures; see below) |
| `embeddings_features/` | BGE-M3 passage embeddings (1024-d), passage x verse similarity scores, step-2 features, student scores |
| `frozen/` | frozen system files, predictions and result tables of the earlier runs (`frozen/README.md`) |
| **`paper_tables/`** | **the 59-lecture labels, scripts and outputs behind every table and figure of the paper** (`paper_tables/README.md`) |
| `experiment_log/` | `experiment_log.md`, one row per run with commit and config |
| `eval/` | retrieval query sets: 150 team-written, 50 scholar-written, development queries |
| `reconstruct/` | `sources.csv` (URLs, ASR settings), `asr_segment_timings.csv`, ASR / segmentation / alignment scripts, `verify_reconstruction.py` |
| `code/` | all experiment, detection, retrieval and app code |
| `tools/check_release.py` | pre-release checks |

**Splits.** Code and catalogue call the 9 tuning lectures `dev`; `test` is the 34 same-speaker test lectures (E2) and
`heldout` the 16 lectures by unseen speakers (E3).

## What is not here

Audio, transcripts (`data/transcripts*`, `data/normalized`), passage text, per-lecture exports, chunk files.
Lecture rights belong to the speakers and publishers. Please fetch lectures from the source URLs.

## Licence

Code: MIT (`LICENSE`). Labels and data: CC BY-NC 4.0 (`LICENSE-DATA`).

## Versions

- **v1.0.0 (7 Nov 2026)**: 51 lectures, first release.
- **v1.1.0 (21–22 Nov 2026)**: 8 more lectures (`video_11`..`video_14`, original English speaker, E2 test;
  `en_chinmaya_day06`..`day09`, Swami Aparajitananda, E3), GPT-6.1 Sol and Gemini 3.1 Pro labels, cascade results,
  `catalogue/lecture_meta.csv`.
- **v1.2.0**: `paper_tables/` with the 59-lecture labels (Gemini 3.1 Pro low and high, GPT-6.1 Sol, GitaGround-3L,
  discourse cue, gpt-5-mini), features, scripts and outputs for every table of the paper; speaker names for the two
  original courses; ASR model for every lecture; updated datasheet and citation files.

## Notes for users

- The paper's 59-lecture numbers come from `paper_tables/`. `frozen/` and `labels/llm/` hold the earlier 42- and
  51-lecture runs; those files are unchanged.
- `embeddings_features/step2_features_59.npz` is a separate rebuild in which the meaning profiles include the new
  lectures; the paper uses `paper_tables/work/new4e2/step2_features_full.npz`, whose first 1,884 passages are identical
  to `embeddings_features/step2_features.npz`.
- `labels/llm/` also holds runs that are not reported in the paper (Claude Opus 5.5, Gemini 3.8 Flash, GPT-4o-mini
  routes); they are kept for completeness.
- All 59 lectures were transcribed with Sarvam AI speech recognition (`saaras:v3`); `reconstruct/asr/transcribe_new.py`
  is the call script.
- Unit-level timestamps do not exist for most passages; `time_start`/`time_end` are empty in the grid.
  `reconstruct/asr_segment_timings.csv` has Whisper segment timings for `video_01`..`video_10` only (an earlier draft
  transcription, not the one used in the paper).
- `*.pkl` files are scikit-learn pickles. Load only from this repository, with the versions in `code/requirements.txt`.
- Annotators: see `DATASHEET.md`.
- English verse translations used for alpha > 0 profiles are under copyright and not included; see `code/verse_translations.NOTE.md`.
