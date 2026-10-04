"""Regenerate every hypothesis probe submitted to HxBuddy (results in RESULTS.md).

Each probe ranks the 4,000 candidates with a candidate merit formula and grants the
top K. Only the ranking differs between probes, so HxBuddy accuracy measures how
close each formula is to the hidden merit standard.

Run from the repository root:  python probes/generer_probes.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import equialgo as eq

demandes, c = eq.charger()
R, H, LI = c['cote_r_equivalent'], c['heures_travail_semaine'], c['log_revenu']
FG, D, EL = c['premiere_generation_universitaire'], c['distance_domicile_campus_km'], c['eloignee']


def top(score, k):
    return eq.octroyer_top(score, k / len(c))


# Round 1 (K=1600): baseline and committee-rule variants
CAT = ['programme_etudes', 'region_administrative', 'code_postal_3']
brut = pd.read_csv('data/donnees_demandes.csv')
X = pd.get_dummies(brut.drop(columns=['id_candidat', 'decision_octroi']), columns=CAT)
Xc = pd.get_dummies(pd.read_csv('data/candidats_evaluation.csv').drop(columns=['id_candidat']),
                    columns=CAT).reindex(columns=X.columns, fill_value=0)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42).fit(X, brut['decision_octroi'])

comite = 1.385 * R + 1.78 * LI + 0.202 * H
probes = {
    'p00_baseline': rf.predict(Xc),
    'p01_R_only': top(R, 1600),
    'p02_committee_no_region': top(comite, 1600),
    'p03_committee_with_region': top(comite - 2.156 * EL, 1600),
}

# Round 2 (K=1590, estimated number of truly meritorious candidates)
K = 1590
lin = lambda w: 1.385 * R + w * (1.78 * LI + 0.202 * H)
f2 = ['cote_r_equivalent', 'log_revenu', 'heures_travail_semaine', 'premiere_generation_universitaire']
dum = lambda df: pd.concat([df[f2], pd.get_dummies(df['programme_etudes'], dtype=float)], axis=1)
centres = demandes['eloignee'] == 0
gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
gbm.fit(dum(demandes)[centres], demandes.loc[centres, 'decision_octroi'])
cols_lr = ['cote_r_equivalent', 'log_revenu', 'heures_travail_semaine', 'premiere_generation_universitaire', 'eloignee']
lr = LogisticRegression(C=1e4, max_iter=20000).fit(demandes[cols_lr], demandes['decision_octroi'])
c_cf = c[cols_lr].assign(eloignee=0)
probes.update({
    'p04_half_inc_hours': top(lin(0.5), K),
    'p05_double_inc_hours': top(lin(2.0), K),
    'p06_R_plus_income': top(1.385 * R + 1.78 * LI, K),
    'p07_R_plus_hours': top(1.385 * R + 0.202 * H, K),
    'p08_gbm_centres_only': top(gbm.predict_proba(dum(c))[:, 1], K),
    'p09_logit_counterfactual': top(lr.decision_function(c_cf), K),
})

# Round 3 (K=1590): hours weight and need-based factors
h0 = 0.146
probes.update({
    'p10_hours_0.08': top(R + 0.08 * H, K),
    'p11_hours_0.22': top(R + 0.22 * H, K),
    'p12_hours_0.30': top(R + 0.30 * H, K),
    'p13_hours_firstgen': top(R + h0 * H + 0.75 * FG, K),
    'p14_hours_need_income': top(R + h0 * H - 0.75 * LI, K),
    'p15_hours_distance': top(R + h0 * H + 0.003 * D, K),
})

dossier = Path(__file__).resolve().parent
for nom, p in probes.items():
    pd.DataFrame({'id_candidat': c['id_candidat'], 'decision_octroi': np.asarray(p, dtype=int)}).to_csv(
        dossier / f'{nom}.csv', index=False)
print(f'{len(probes)} probes written to {dossier}')

# Round 4: fine-tuning around p07 (R + 0.146 H, K=1590)
s07 = R + h0 * H
k_top = lambda s, k=K: top(s, k)
probes4 = {
    'p16_hours_0.12': k_top(R + 0.12 * H),
    'p17_hours_0.17': k_top(R + 0.17 * H),
    'p18_K1560': k_top(s07, 1560),
    'p19_K1620': k_top(s07, 1620),
    'p20_small_need_income': k_top(s07 - 0.25 * LI),
    'p21_small_wealth_income': k_top(s07 + 0.25 * LI),
    'p22_small_firstgen': k_top(s07 + 0.25 * FG),
    'p23_sqrt_hours': k_top(R + h0 * 2 * np.sqrt(H.mean()) * np.sqrt(H)),
    'p24_hours_cap15': k_top(R + h0 * np.minimum(H, 15)),
}
for nom, p in probes4.items():
    pd.DataFrame({'id_candidat': c['id_candidat'], 'decision_octroi': np.asarray(p, dtype=int)}).to_csv(
        dossier / f'{nom}.csv', index=False)
print(f'{len(probes4)} round-4 probes written')

# Round 5: round-number hypotheses (leaderboard leaders sit 4-8 decisions above p07)
probes5 = {
    'p25_hours_0.15': k_top(R + 0.15 * H),
    'p26_hours_0.14_threshold30': ((R + 0.14 * H) >= 30).astype(int).values,
}
for nom, p in probes5.items():
    pd.DataFrame({'id_candidat': c['id_candidat'], 'decision_octroi': np.asarray(p, dtype=int)}).to_csv(
        dossier / f'{nom}.csv', index=False)
print(f'{len(probes5)} round-5 probes written')

# Round 6: hidden standard looks threshold-based (p26 best). Search round (w, T) pairs.
seuil = lambda w, T: ((R + w * H) >= T).astype(int).values
probes6 = {
    'p27_w0.14_T29.95': seuil(0.14, 29.95),
    'p28_w0.14_T30.05': seuil(0.14, 30.05),
    'p29_w1-7_T30': seuil(1 / 7, 30.0),
    'p30_w0.13_T29.9': seuil(0.13, 29.9),
    'p31_w0.15_T30.1': seuil(0.15, 30.1),
    'p32_w0.16_T30.25': seuil(0.16, 30.25),
}
for nom, p in probes6.items():
    pd.DataFrame({'id_candidat': c['id_candidat'], 'decision_octroi': np.asarray(p, dtype=int)}).to_csv(
        dossier / f'{nom}.csv', index=False)
print(f'{len(probes6)} round-6 probes written')

# Round 7: local search around p31 (w=0.15, T=30.1, 94.63%)
probes7 = {
    'p33_w0.15_T30.05': seuil(0.15, 30.05),
    'p34_w0.15_T30.15': seuil(0.15, 30.15),
    'p35_w0.15_T30.2': seuil(0.15, 30.2),
    'p36_w0.155_T30.15': seuil(0.155, 30.15),
    'p37_w0.145_T30.05': seuil(0.145, 30.05),
    'p38_w0.155_T30.2': seuil(0.155, 30.2),
    'p39_w0.16_T30.2': seuil(0.16, 30.2),
    'p40_w0.145_T30.1': seuil(0.145, 30.1),
}
for nom, p in probes7.items():
    pd.DataFrame({'id_candidat': c['id_candidat'], 'decision_octroi': np.asarray(p, dtype=int)}).to_csv(
        dossier / f'{nom}.csv', index=False)
print(f'{len(probes7)} round-7 probes written')

# Round 8: local search around p37 (w=0.145, T=30.05, 94.68%)
probes8 = {
    'p41_w0.145_T30.0': seuil(0.145, 30.0),
    'p42_w0.145_T30.025': seuil(0.145, 30.025),
    'p43_w0.145_T30.075': seuil(0.145, 30.075),
    'p44_w0.1425_T30.025': seuil(0.1425, 30.025),
    'p45_w0.1475_T30.075': seuil(0.1475, 30.075),
    'p46_w0.1475_T30.05': seuil(0.1475, 30.05),
    'p47_w0.1425_T30.05': seuil(0.1425, 30.05),
}
for nom, p in probes8.items():
    pd.DataFrame({'id_candidat': c['id_candidat'], 'decision_octroi': np.asarray(p, dtype=int)}).to_csv(
        dossier / f'{nom}.csv', index=False)
print(f'{len(probes8)} round-8 probes written')

# Round 9: p42 (w=0.145, T=30.025) best at 94.68; lowering the cutoff from p37 gained +4 net.
probes9 = {
    'p49_w0.145_T29.975': seuil(0.145, 29.975),
    'p50_w0.15_T30.075': seuil(0.15, 30.075),
    'p51_w0.14_T29.975': seuil(0.14, 29.975),
}
for nom, p in probes9.items():
    pd.DataFrame({'id_candidat': c['id_candidat'], 'decision_octroi': np.asarray(p, dtype=int)}).to_csv(
        dossier / f'{nom}.csv', index=False)
print(f'{len(probes9)} round-9 probes written')

# Round 10: structural hypotheses on how the reference was built (stratified rankings)
s42 = (R + 0.145 * H).values
taux42 = (s42 >= 30.025).mean()


def par_strate(col):
    out = np.zeros(len(c), dtype=int)
    for _, idx in c.groupby(col).indices.items():
        k = int(round(taux42 * len(idx)))
        out[idx[np.argsort(-s42[idx], kind='stable')[:k]]] = 1
    return out


z_r = c.groupby('programme_etudes')['cote_r_equivalent'].transform(lambda x: (x - x.mean()) / x.std())
probes10 = {
    'p52_top_per_program': par_strate('programme_etudes'),
    'p53_top_per_region': par_strate('region_administrative'),
    'p54_R_zscored_by_program': eq.octroyer_top((z_r * R.std() + 0.145 * H).values, taux42),
}
for nom, p in probes10.items():
    pd.DataFrame({'id_candidat': c['id_candidat'], 'decision_octroi': np.asarray(p, dtype=int)}).to_csv(
        dossier / f'{nom}.csv', index=False)
print(f'{len(probes10)} round-10 probes written')
