# Datasheet / data statement

**Motivation.** A benchmark for Canonical Verse Grounding: label every ~100-word passage of a spoken lecture with the
Bhagavad Gītā Chapter 2 verse(s) it explains, or as a digression.

**Composition.** 2,583 passages from 59 lectures by 6 speakers (Dvaita, Advaita, Gauḍīya Vaiṣṇava; 38 Kannada and
21 English lectures), verses BG 2.1 to 2.72, 2,053 verse mentions, 869 digressions (33.6%). Splits: 9 tuning lectures
(6 Kannada, 3 English); 34 same-speaker test lectures, E2 (23 Kannada, 11 English); 16 lectures by four unseen speakers,
E3 (9 Kannada, 7 English). Each mention has a type T1 to T5. Nine themes cover the 72 verses.

| Speaker | Language, tradition | Lectures |
|---|---|---|
| Haridasa Bhat Acharya | Kannada, Dvaita | 29 (6 tuning, 23 E2) |
| Swami Sarvapriyananda | English, Advaita | 14 (3 tuning, 11 E2) |
| Sumith | Kannada, Dvaita | 8 (E3) |
| Vidwan Brahmanyacharya | Kannada, Dvaita | 1 (E3) |
| HG Gaurmandal Das (ISKCON) | English, Gauḍīya Vaiṣṇava | 1 (E3) |
| Swami Aparajitananda (Chinmaya Mission) | English, Advaita | 6 (E3) |

**Collection.** Audio from public recordings (42 h 40 min), transcribed with Sarvam AI speech recognition (`saaras:v3`)
and split at sentence boundaries into passages of about 100 words (274,992 words).

**Annotation.** Labels follow a written decision procedure (`docs/annotation_guideline_mention_types.md`). The corpus
was labeled by four Gītā experts. Before any test evaluation, 209 passages from 7 lectures were relabeled blind to the
first labels by two of these experts, each only on lectures they had not labeled (`labels/second_pass/`): strict
κ = 0.863. The 27 disagreements were adjudicated by a fifth Gītā expert who took part in neither pass and saw the two
labels as A and B in random order (`labels/adjudication/`). The authors entered the experts' labels into the files, so
the names in the annotator columns (`first_annotator`, `second_annotator`, `labels/gold/by_annotator/`,
`labels/second_pass/*.csv`) record who entered the labels, not who made them.

**LLM labels.** Included and marked as such, never merged into gold: the gpt-5-mini discourse cue (label, confidence),
whole-lecture labels from GPT-6.1 Sol (high) and Gemini 3.1 Pro (low, high), gpt-5-mini end to end, and the
GitaGround-3L cascade outputs (`paper_tables/`), plus earlier runs not reported in the paper (`labels/llm/`).

**Retrieval queries.** 150 team-written queries (50 questions, 40 pasted ślokas, 35 explanation excerpts written by the
team, 25 unanswerable) and 50 queries written by a Gītā scholar who had no access to system outputs (11 unanswerable);
expected verses were fixed before any system was run. Development queries: 69 answerable, 12 unanswerable.

**What is released.** Identifiers, offsets, word counts and text hashes; labels; embeddings and features; code; source URLs.
**What is not.** Audio and transcript text, which are copyrighted by the speakers and publishing institutions.

**Uses and limits.** One chapter of one text; one ASR system; the Kannada course rarely digresses; two unseen speakers
have one lecture each. Not suitable for evaluating new text models without reconstructing the passages. Non-commercial use.

**Licences.** Code MIT; labels and data CC BY-NC 4.0. Speaker names are those of public lecturers.

**Maintenance.** Issues on the GitHub repository; versions are archived on Zenodo.
