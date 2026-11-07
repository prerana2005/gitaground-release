# Datasheet / data statement

**Motivation.** A benchmark for Canonical Verse Grounding: label every ~100-word passage of a spoken lecture with the
Bhagavad Gita Chapter 2 verse it explains, or with an off-verse state.

**Composition.** 1,884 passages from 51 lectures by 6 speakers (Dvaita, Advaita, Gaudiya Vaishnava; Kannada and English),
verses BG 2.1 to 2.72, 1,617 verse mentions, 499 off-verse passages (26.5%). Splits: 9 tuning, 30 same-speaker test,
12 unseen-speaker lectures. Each mention has a type T1 to T5. Nine themes cover the 72 verses.

**Collection.** Audio from public recordings, transcribed with Sarvam AI ASR and split at sentence boundaries into
passages of about 100 words. Labels by the authors; each passage has one annotator, except 209 passages from 7 lectures
that had a blind second pass (kappa 0.863 strict) with 27 disagreements adjudicated by a domain expert.
LLM labels (gpt-5-mini cue, gpt-5-mini, Claude Opus 5.5 and Gemini 3.1 Pro end to end) are included and marked as such.

**What is released.** Identifiers, offsets, word counts and text hashes; labels; embeddings and features; code; source URLs.
**What is not.** Audio and transcript text, which are copyrighted by the speakers and publishing institutions.

**Uses and limits.** One chapter of one text; one ASR system; 150 retrieval queries written by the team; three unseen speakers
have only 1 to 2 lectures. Not suitable for evaluating new text models without reconstructing the passages. Non-commercial use.

**Licences.** Code MIT; labels and data CC BY-NC 4.0. Speaker names are those of public lecturers.

**Maintenance.** Issues on the GitHub repository; versions are archived on Zenodo.
