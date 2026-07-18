# Manual Utility Evaluation Plan — validating the YAMNet proxy

## Motivation
Utility metrics (mAP / TC@3 / TA@1 / preserve_score) are computed from **YAMNet**,
which is only a *proxy* classifier (521 AudioSet classes, not trained on
CitySpeechMix). This plan hand-validates a random sample against the dataset's
ground-truth AudioSet labels (`label1_audioset` = environmental,
`label2_audioset` = speech).

## Design
- **Sampling frame:** all classified chunks from a chosen run/config
  (`*_report.json` → `classification_top3`, `classification_top3_original`).
- **Stratification:** keep the natural speech / env ratio (≈50/50 in this
  dataset) so both regimes are represented.
- **Sample size:** start with **n = 60** (≈30 speech, ≈30 env). Enough for a
  ±13% Wilson 95% CI on an accuracy proportion; bump to 100–150 if a tighter
  interval is needed for the write-up.
- **Blinding:** the reviewer sees the audio + YAMNet top-3 + ground-truth label
  and marks correctness. (Optionally hide GT for a stricter blind pass.)

## Procedure
1. Generate the review sheet:
   ```bash
   python3 scripts/sample_manual_utility_eval.py \
     --reports-dir logs/s3/<RUN>/<config> \
     --metadata s3://<BUCKET>/cityspeechmix/metadata/metadata.csv \
     --n 60 --out manual_eval_sheet.csv
   ```
2. For each row, listen to the chunk and fill:
   - `human_top1_correct(Y/N)` — is YAMNet's top-1 the true dominant sound?
   - `human_gt_in_top3(Y/N)` — is the ground-truth class anywhere in top-3?
   - `human_true_label`, `notes`.
3. Score:
   ```bash
   python3 scripts/sample_manual_utility_eval.py --score manual_eval_sheet.csv
   ```
   → reports human-judged Top-1 accuracy, Top-3 hit rate, and per-regime
   (speech vs env) accuracy.

## What to report in the paper
- Human-judged Top-1 accuracy and Top-3 hit rate of the YAMNet proxy, with a
  Wilson 95% CI, split by speech vs environment.
- A short agreement statement: e.g. "the YAMNet proxy agreed with human ground
  truth on X% of top-1 predictions (95% CI …), supporting its use as a utility
  proxy; disagreements were concentrated in <cases>."
- 2–3 qualitative examples where proxy ≠ GT (e.g. GT `Siren` → YAMNet top-1
  `Emergency vehicle` but `Siren` present in top-3) to show near-misses vs true
  errors — motivates reporting **Top-3** consistency alongside Top-1.

## Caveats to acknowledge
- Vocabulary mismatch: CitySpeechMix uses AudioSet parent labels; YAMNet emits
  fine-grained AudioSet leaves → a "wrong" top-1 may be a correct hypernym
  (`Emergency vehicle` ⊃ `Siren`). Count hypernym/hyponym matches as correct and
  note this rule.
- This validates the *proxy*, not the privacy transform. Keep it a small,
  clearly-scoped sub-study.
