"""Shared helpers for the EquiAlgo audit and corrected model.

Both notebooks import from here so the data loading, group definition and
merit score are defined exactly once.
"""
import numpy as np
import pandas as pd

ELOIGNEES = ['Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine']
BUDGET_MIN, BUDGET_MAX = 0.36, 0.44
TAUX_OCTROI = 0.3975  # estimated share of truly meritorious applicants (see probes/RESULTS.md)

# Committee rule reconstructed by logistic regression on donnees_demandes.csv
# (audit_rapport.ipynb, section 4). Logit units.
COEF_R = 1.385
COEF_HEURES = 0.202
COEF_LOG_REVENU = 1.78   # wealth bonus: not merit
COEF_ELOIGNEE = -2.15    # regional penalty: not merit

# Merit weight of one hour worked, in R-score points: the committee's own trade-off.
POIDS_HEURES = COEF_HEURES / COEF_R  # ~0.146


def charger(chemin='data'):
    """Load historical applications and evaluation candidates, with derived columns."""
    demandes = pd.read_csv(f'{chemin}/donnees_demandes.csv')
    candidats = pd.read_csv(f'{chemin}/candidats_evaluation.csv')
    for df in (demandes, candidats):
        df['eloignee'] = df['region_administrative'].isin(ELOIGNEES).astype(int)
        df['groupe'] = np.where(df['eloignee'] == 1, 'Eloignee', 'Centre')
        df['log_revenu'] = np.log(df['revenu_familial_estime'])
    return demandes, candidats


def score_merite(df, poids_heures=POIDS_HEURES):
    """Region- and wealth-neutral merit score, in R-score points."""
    return df['cote_r_equivalent'] + poids_heures * df['heures_travail_semaine']


def octroyer_top(score, taux=TAUX_OCTROI):
    """Grant to the top `taux` share of applicants by score (fixed budget)."""
    score = np.asarray(score)
    k = int(round(taux * len(score)))
    decision = np.zeros(len(score), dtype=int)
    decision[np.argsort(-score, kind='stable')[:k]] = 1
    return decision


def ecart_egalite_chances(y_merite, y_pred, eloignee):
    """TPR(centre) - TPR(remote), with merit as the ground truth."""
    y_merite, y_pred, eloignee = map(np.asarray, (y_merite, y_pred, eloignee))
    tpr = lambda g: y_pred[(eloignee == g) & (y_merite == 1)].mean()
    return tpr(0) - tpr(1)


def ecart_parite(y_pred, eloignee):
    """Selection rate(centre) - selection rate(remote)."""
    y_pred, eloignee = np.asarray(y_pred), np.asarray(eloignee)
    return y_pred[eloignee == 0].mean() - y_pred[eloignee == 1].mean()


def verifier_soumission(df_soumission, candidats):
    """Raise if the submission breaks the format or the budget."""
    assert list(df_soumission.columns) == ['id_candidat', 'decision_octroi']
    assert len(df_soumission) == len(candidats) == 4000
    assert (df_soumission['id_candidat'].values == candidats['id_candidat'].values).all()
    assert df_soumission['decision_octroi'].isin([0, 1]).all()
    taux = df_soumission['decision_octroi'].mean()
    assert BUDGET_MIN <= taux <= BUDGET_MAX, f'budget broken: {taux:.3f}'
    return taux
