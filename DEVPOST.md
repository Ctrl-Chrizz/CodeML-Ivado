# Devpost submission text (copy into the form)

**Project name:** ÉquiAlgo: It was never the R score (PolyFinances)

**Prize / challenge:** IVADO – ÉquiAlgo (select exactly this one)

**Tagline (short description):**
We found that a scholarship committee penalized remote regions twice, and built one transparent rule that closes the equal-opportunity gap while raising accuracy from 89% to 95%.

## Inspiration
A model that is 88% accurate can still be unfair: it grants scholarships to 48% of applicants in Montréal and Québec City but only 27% in Bas-Saint-Laurent, Côte-Nord and Gaspésie. We wanted to find out how much of that gap is merit and how much is bias.

## What it does
- **Audits** the historical committee: at the same R score, remote students are granted far less often (59% vs 26% for R 28–29).
- **Reconstructs the committee's rule** and finds two terms unrelated to merit: a regional penalty (≈ −1.6 R-score points) and a wealth bonus (doubling family income ≈ +0.9 R-score points). Together they explain the entire gap.
- **Shows that both groups are equally meritorious** (39.7% vs 39.8%) once hours worked during studies count.
- **Replaces the model** with one transparent rule: R score + 0.145 × hours ≥ 30.025. Grant rates become 40.4% (centres) and 40.2% (remote), within the 36–44% budget.
- **Proposes a governance plan:** monitoring thresholds, blind human review, appeals, Quebec Law 25 compliance, and retraining guardrails.

## How we built it
- Python, pandas, scikit-learn and fairlearn.
- We compared 28 trained configurations from five families: the production random forest, fairness through unawareness, ThresholdOptimizer and group thresholds, ExponentiatedGradient (TPR parity and demographic parity), and our neutralized model.
- The neutralized model is a logistic regression trained *with* region and income as context, so they absorb the bias. At prediction time everyone is scored with the same context. Sweeping the share of bias kept traces the Pareto front.
- We tested each hypothesis about merit against the hidden reference on HxBuddy; all probes and scores are in the repo.

## Challenges we ran into
- Deleting the region column doesn't work: postal code identifies region perfectly and distance almost perfectly.
- Hours worked are both a regional proxy *and* real merit, so dropping every proxy also drops merit.
- fairlearn's equal-opportunity constraint plateaus at a 0.20 gap against merit, because the only labels it can constrain against are the committee's biased ones.

## Accomplishments that we're proud of
- Equal-opportunity gap: 0.29 → 0.00.
- Accuracy against the hidden reference: 89.33% → 94.65%. Fairness and accuracy improved together.
- The model learned the hours weight (0.146 R-score points per hour) from the committee itself, and the leaderboard confirmed it.

## What we learned
Measuring fairness against biased labels only measures fidelity to the bias. The question to ask of a variable is not "does it correlate with region?" but "does it measure merit?"

## What's next for ÉquiAlgo
Per-region and intersectional monitoring, a blind human-reviewed sample each round, and a stakeholder review (including students from the regions) of what counts as merit.

## Built with
python, pandas, scikit-learn, fairlearn, matplotlib, jupyter

## Links
- GitHub repository: https://github.com/Ctrl-Chrizz/CodeML-Ivado
- Presentation: `presentation.pdf` in the repository
