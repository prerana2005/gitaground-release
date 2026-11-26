## Data and code availability (as stated in the paper)

We release the verified verse catalog and theme file; the list of all 59 lectures with their public URLs and speaker
metadata; the passage grid with lecture identifiers, character offsets, and a hash of each passage's text; gold verse
labels and mention types; the blind second pass and its adjudication; both query files; the discourse-cue labels; the
frontier-LLM labels; pre-computed passage embeddings and evidence features, which contain no transcript text; the frozen
system files; and the scripts and experiment log from which every table can be recomputed. Code is released under the
MIT license and data under CC BY-NC 4.0.

Recordings and transcripts are protected by speaker and institutional copyright and are not redistributed; the released
scripts rebuild the passage grid from a transcript of each public recording by the same deterministic sentence-boundary
segmentation, a verifier checks each rebuilt passage against its hash, and the labels attach to it by lecture identifier
and offset (a fresh transcription may differ slightly from ours).
