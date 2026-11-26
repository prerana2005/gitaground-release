# Reconstructing the passage text (Route a)

This release does not contain audio or transcript text. To rebuild the passages you must fetch the
lectures yourself, from the original uploaders, and run the same pipeline.

1. `sources.csv` lists each lecture, its `source_url` and ASR settings. Fetch audio for lectures you are
   entitled to use: `python reconstruct/fetch_audio.py` (needs `yt-dlp`; writes to `reconstruct/audio/`).
2. **ASR** — all 59 lectures were transcribed with Sarvam AI speech recognition, model `saaras:v3`
   (model per lecture in `sources.csv`). `asr/transcribe_new.py` is the batch call script, with mode `transcribe`.
   `asr/backend/src/process_videos.py` converts video to audio. `asr/backend/src/transcribe.py` (faster-whisper)
   produced earlier draft transcripts that are not used in the paper; `asr_segment_timings.csv` gives their
   segment start/end seconds and a SHA-256 of each segment text for `video_01`..`video_10`.
3. **Segmentation** — `segmentation/build_units.py` (and `build_units_heldout.py`) split transcripts into
   ~100-word passages and write the normalized text.
4. **Alignment / verification** — `python reconstruct/verify_reconstruction.py --units data/units_v2.csv --grid passage_grid/units_v2.csv`
   checks that every rebuilt passage has the same `unit_id`, character offsets and `text_sha256` as the
   released grid. `alignment/timestamp_grounder.py` maps passages to YouTube deep links.

Known limit: ASR is not bit-for-bit repeatable across model versions. If hashes differ, the offsets in
the grid will not line up and the labels cannot be applied safely.
