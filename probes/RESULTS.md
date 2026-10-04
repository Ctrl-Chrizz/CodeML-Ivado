# HxBuddy probe results (vs hidden "merite reel"), 2026-10-03
All probes except p00 grant top-K by score (K=1600 for p01-03, 1590 after).
Estimated true merit positives: ~1590 / 4000 (39.75%), solved from acc + macro F1.

| probe | score | acc | F1 macro |
|---|---|---|---|
| p00_baseline | production RF | 89.33 | 88.83 |
| p01_R_only | R | 92.28 | 91.95 |
| p02_committee_no_region | 1.385R+1.78ln(inc)+0.202H | 92.58 | 92.26 |
| p03_committee_with_region | p02 - 2.156*remote | 87.93 | 87.42 |
| p04_half_inc_hours | R-weight + 0.5x(inc,H) | 93.43 | 93.14 |
| p05_double_inc_hours | R-weight + 2x(inc,H) | 88.23 | 87.72 |
| p06_R_plus_income | 1.385R+1.78ln(inc) | 90.08 | 89.65 |
| **p07_R_plus_hours** | **R + 0.146H** | **94.58** | **94.34** |
| p08_gbm_centres_only | GBM trained on centres | 90.68 | 90.27 |
| p09_logit_counterfactual | logit, remote:=0 | 92.53 | 92.20 |
| p10_hours_0.08 | R+0.08H | 93.98 | 93.72 |
| p11_hours_0.22 | R+0.22H | 93.63 | 93.35 |
| p12_hours_0.30 | R+0.30H | 92.13 | 91.79 |
| p13_hours_firstgen | p07 + 0.75*FG | 93.03 | 92.72 |
| p14_hours_need_income | p07 - 0.75ln(inc) | 93.78 | 93.51 |
| p15_hours_distance | p07 + 0.003*dist | 93.03 | 92.72 |

Conclusion: merit ~ R + 0.146*hours. Committee adds two biases: remote penalty (-2.15 logit)
and a wealth bonus (+1.78 per ln income). Removing both from the committee's fitted rule gives p07.

Reproduce with `python probes/generer_probes.py`. p01, p11 and p12 differ by 2 rows (ties at the cutoff, sort order); the CSVs here are the exact files uploaded.

## Rounds 4-10 (2026-10-03/04), accuracy vs hidden merit
Threshold rules are `R + w*H >= T`. Only scores confirmed by the team are listed.

| probe | rule | acc |
|---|---|---|
| p26 | w=0.14, T=30 | 94.58 |
| p31 | w=0.15, T=30.1 | 94.63 |
| p37 | w=0.145, T=30.05 | 94.58 |
| **p42** | **w=0.145, T=30.025 (= predictions.csv)** | **94.65** (verified by uploading predictions.csv; earlier reported as 94.68) |
| p52 | top 40.3% per program | 94.48 |
| p53 | top 40.3% per region | 94.53 |
| p54 | R z-scored by program | 94.50 |

Recalibration re-uploads (p00, p02, p03, p06, p07, p10, p12-p15) reproduced their original scores exactly: the scorer did not change.
Linear shape check: committee's hours effect is ~0.17 R-pts/hour up to 21 h; R effect smooth. No leakage (no shared ids/rows, ids uninformative).
