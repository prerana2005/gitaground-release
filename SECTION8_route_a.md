## 8 Reproducibility and copyright-safe release

Because recordings and transcripts are protected by speaker and institutional copyright, the transcript text cannot be
redistributed. We release: the verified verse catalogue and theme file; the passage grid with lecture identifiers and
offsets; gold verse labels and mention types; the blind second pass and its adjudication; the query files; the LLM cue
labels (label, confidence) for all 1,884 passages; the end-to-end gpt-5-mini, Claude Opus 5.5 and Gemini 3.1 Pro labels;
pre-computed BGE-M3 passage embeddings and evidence features, which contain no transcript text; the frozen system files;
and the experiment log with the pre-registration entries, from which every table can be recomputed. A data statement [3]
and datasheet [13] accompany the release.

Reconstruction: we provide source URLs and retrieval dates of the public recordings, with the ASR and segmentation scripts
that regenerate the passage grid. ASR output may differ slightly between runs, so each released passage carries a hash of
its text, and a verifier reports which regenerated passages match. Code is released under the MIT licence and labels under
CC BY-NC 4.0.
