# Pre-specified protocol bundle (hold-out evaluation)

This folder is meant to be uploaded unchanged to a public archive (Zenodo or OSF). The DOI of that deposit is cited in
Section 4.6 of the manuscript. The protocol was fixed in these files before any hold-out inference, but it was not
lodged with an external registry; the manuscript therefore calls it pre-specified rather than pre-registered. (The
file names keep the original `prereg` prefix so that the hashes in `MANIFEST.json` stay valid.)

## Contents
- `prereg.json` — frozen protocol: 30 method configurations, the confirmatory families F1 (plug-in) and F2
  (single-engine efficiency) with directional hypotheses, statistics (sequence-level paired bootstrap, Holm), and
  SHA-256 hashes of the detector weights and of the evaluation code at the time of freezing.
- `prereg_amendment1.json` — the single amendment, adding UAVDT (written before any UAVDT inference).
- `prereg.py` — the script that wrote both files; `holdout.py` — the script that ran the frozen protocol.
- `MANIFEST.json` — SHA-256 of every file above, their local creation times, and the creation time of the first
  hold-out output (first detection file written for each hold-out dataset).

## Timeline (local time, KST)
| Event | Time |
|---|---|
| `prereg.json` written | 2026-10-20 18:11:18 |
| `prereg_amendment1.json` written (UAVDT added) | 2026-10-20 18:27:25 |
| first VisDrone-VID hold-out output | 2026-10-20 18:28:43 |
| first UAVDT hold-out output | 2026-10-20 19:43:03 |

## Limitations (stated in the manuscript)
- The protocol is internal: the times above come from the local file system and were not lodged with a registry
  before the hold-out inference. Archiving this bundle makes the files public and immutable, but does not prove the
  original times.
- The protocol specified 1,000 bootstrap replicates. After the evaluation, the confirmatory pairs were re-run with
  10,000 replicates (same seed and resampling scheme) because 1,000 replicates resolve p only to about 0.001; both
  analyses are reported, with p = (k + 1)/(B + 1). The number of significant F1 comparisons after Holm correction
  is 39 of 44 in both.
- After the hold-out evaluation, the evaluation code was extended with switches that are off by default (alternative
  fusion rules for the overlap analysis, Soft-NMS, a scale factor for the TAF border distance, a GPU voting option,
  the Dynamic Tiling baseline and a Faster R-CNN wrapper) and with environment-variable dataset roots.
  The hold-out results were produced and stored before these changes, so the current files no longer match the code
  hashes listed in `prereg.json`; the raw hold-out detections (`dets.pkl`) are deposited together with the trained
  weights (see the repository README) so that every reported number can be recomputed.
