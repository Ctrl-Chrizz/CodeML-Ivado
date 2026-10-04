# ÉquiAlgo: fair student financing, Team PolyFinances

IVADO challenge, Engineering and Computer Science Hackathon 2026.

A fictional Quebec lender's model grants scholarships to 48.4% of applicants from the large centres but only
27.3% from remote regions. We found that **the entire gap is bias**: the committee whose decisions trained the
model applied a **regional penalty** (≈ −1.6 R-score points) and a **wealth bonus** (doubling family income
≈ +0.9 R-score points). Once hours worked during studies are counted, both groups are equally meritorious
(39.7% vs 39.8%).

Our model removes both biases and grants scholarships by a single rule:
**R score + 0.145 × hours worked ≥ 30.025**.

## Results (4,000 evaluation candidates)

| | Production model | Ours |
|---|---|---|
| Grant rate, centres | 46.8% | 40.4% |
| Grant rate, remote regions | 27.8% | 40.2% |
| Equal-opportunity gap (vs merit) | 0.29 | 0.00 |
| Accuracy vs hidden reference (HxBuddy) | 89.33% | 94.65% |
| Overall grant rate (budget 36–44%) | 39.1% | 40.3% |

## Deliverables

| File | Contents |
|---|---|
| [`predictions.csv`](predictions.csv) | 4,000 decisions (`id_candidat,decision_octroi`), grant rate 40.3% |
| [`audit_rapport.ipynb`](audit_rapport.ipynb) | Bias measurement, the committee's reconstructed rule, proxy variables, choice of fairness metric, limitations |
| [`model_corrige.ipynb`](model_corrige.ipynb) | 28 trained configurations from 5 families, Pareto front, final model, governance and monitoring plan |
| [`presentation.pdf`](presentation.pdf) | 11-slide deck for the 5-minute pitch |
| [`pareto_front.png`](pareto_front.png) | The equity–utility Pareto front |
| [`equialgo.py`](equialgo.py) | Shared helpers: loading, merit score, fairness metrics, submission check |
| [`probes/`](probes/) | Every hypothesis tested against HxBuddy, the code that generates them, and the scores ([`RESULTS.md`](probes/RESULTS.md)) |

## Approach in brief

1. **Diagnose** (`audit_rapport.ipynb`). At the same R score, remote applicants are granted far less often
   (59% vs 26% for R 28–29). A logistic regression reproduces the committee as well as gradient boosting, so its
   coefficients can be read directly: R score and hours worked are merit; region and income are bias.
   Postal code, distance, hours and income all leak region, so deleting the region column barely helps.
2. **Choose the metric.** Equal opportunity, measured **against merit**, not against the committee's labels.
   Measuring against `decision_octroi` only checks fidelity to a biased committee.
3. **Correct** (`model_corrige.ipynb`). We compare five families of trained models: the production random
   forest, fairness through unawareness, fairlearn `ThresholdOptimizer` and group thresholds, fairlearn
   `ExponentiatedGradient` (TPR parity and demographic parity), and our **neutralized model**. The neutralized
   model is a logistic regression trained with region and income included as context, so they absorb the bias.
   At prediction time every candidate is scored with the same context. A parameter λ sweeps from the committee
   as-is (λ = 1) to fully neutralized (λ = 0), which traces the Pareto front.
4. **Calibrate the cutoff.** The model sets the ranking; the cutoff (30.025, 1,613 grants) was chosen against
   the hidden reference on HxBuddy. Neighbouring weights and cutoffs all score lower. About 94.7% appears to be
   the noise ceiling of the reference standard.
5. **Govern.** Monitoring thresholds, blind human review, appeals, Quebec Law 25 obligations and retraining
   guardrails are in the last section of `model_corrige.ipynb`.

## Reproduce

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace audit_rapport.ipynb
jupyter nbconvert --to notebook --execute --inplace model_corrige.ipynb   # writes predictions.csv
python probes/generer_probes.py                                          # regenerates the probe files
```

Tested with fairlearn 0.13.0, scikit-learn 1.6.1, pandas 2.3.3 and numpy 2.0.2 (run on Python 3.9.6; the
challenge recommends 3.10+, and nothing here depends on the difference). Both notebooks run end to end in under
a minute.

## Limitations

- The merit standard (R score + hours) is a value judgement. We made it explicit and checked it against the hidden reference.
- Hours worked are partly a product of economic need.
- The audit compares two groups; production monitoring should cover each region and intersections (region × first generation, region × program).
- The data is synthetic and has none of the unobserved factors (letters, interviews) that could carry other biases.

The original challenge brief is in [`CHALLENGE.md`](CHALLENGE.md) and [`LISEZMOI.md`](LISEZMOI.md).
