# Reconstructing the passage text (Route a)

This release does not contain audio or transcript text. To rebuild the passages you must fetch the
lectures yourself, from the original uploaders, and run the same pipeline.

1. `sources.csv` lists each lecture, its `source_url` and ASR settings. Fetch audio for lectures you are
   entitled to use: `python reconstruct/fetch_audio.py` (needs `yt-dlp`; writes to `reconstruct/audio/`).
2. **ASR** — `asr/backend/src/process_videos.py` (video to mp3) then `transcribe.py` (faster-whisper).
   The corpus used in the paper was transcribed with Sarvam AI ASR (all 51 lectures). The Whisper
   pipeline here produced the earlier drafts. The Sarvam call script is not yet in this repository (see below).
   `asr_segment_timings.csv` gives Whisper segment start/end seconds and a SHA-256 of each segment text
   for `video_01`..`video_10`, so a rerun can be checked segment by segment.
3. **Segmentation** — `segmentation/build_units.py` (and `build_units_heldout.py`) split transcripts into
   ~100-word passages and write the normalized text.
4. **Alignment / verification** — `python reconstruct/verify_reconstruction.py --units data/units_v2.csv --grid passage_grid/units_v2.csv`
   checks that every rebuilt passage has the same `unit_id`, character offsets and `text_sha256` as the
   released grid. `alignment/timestamp_grounder.py` maps passages to YouTube deep links.

Known limit: ASR is not bit-for-bit repeatable across model versions. If hashes differ, the offsets in
the grid will not line up and the labels cannot be applied safely.

## Status of the Sarvam ASR step
The paper's transcripts come from Sarvam AI. The script that called it is not in `asr/` yet; add it before
release, with the model name, mode and date of each run in `sources.csv` (`asr_model`, `asr_provider`,
`retrieved_at`).
