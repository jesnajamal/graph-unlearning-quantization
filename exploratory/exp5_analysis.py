"""Applies the Experiment 5 pre-registered screening / confirmation rules."""
import sys
import numpy as np, pandas as pd
from scipy import stats
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
RES, STAGE = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ('results_exp5', 'screen')
d = pd.read_csv(f'{RES}/{STAGE}.csv')
g = d.groupby(['config', 'dataset', 'labels'])
s = g.agg(gap_mean=('gap', 'mean'), gap_sd=('gap', 'std'), gap_pos=('gap', lambda x: int((x > 0).sum())), n=('gap', 'size'),
          noise_mean=('retrain_noise', 'mean'), auc_orig=('auc_orig', 'mean'), auc_retA=('auc_retA', 'mean'),
          adv_orig=('adv_orig', 'mean'), adv_retA=('adv_retA', 'mean'), tpr_fpr_orig=('tpr_fpr_orig', 'mean'),
          tpr_fpr_retA=('tpr_fpr_retA', 'mean'), test_acc=('test_acc_orig', 'mean'), train_acc=('train_acc_orig', 'mean'),
          n_deleted=('n_deleted', 'mean')).reset_index()
if STAGE == 'screen':
    best = s.groupby(['dataset', 'labels']).test_acc.transform('max')
    s['c1_all_seeds_positive'] = s.gap_pos == s.n
    s['c2_gap_ge_3x_noise'] = s.gap_mean >= 3 * s.noise_mean
    s['c3_gap_ge_0.10'] = s.gap_mean >= 0.10
    s['stretch_gap_ge_0.15'] = s.gap_mean >= 0.15
    s['c4_utility'] = s.test_acc >= 0.9 * best
    s['passes'] = s.c1_all_seeds_positive & s.c2_gap_ge_3x_noise & s['c3_gap_ge_0.10'] & s.c4_utility
    print(s.round(3).to_string(index=False))
    sel = s[s.passes].sort_values(['gap_mean', 'test_acc'], ascending=False).groupby('dataset').head(1)
    print('\nSelected for confirmation (frozen):\n', sel[['dataset', 'config', 'gap_mean', 'noise_mean', 'test_acc']].round(3).to_string(index=False))
    s.to_csv(f'{RES}/screen_summary.csv', index=False); sel.to_csv(f'{RES}/selected_for_confirmation.csv', index=False)
else:
    out = []
    for cfg, x in d.groupby('config'):
        n = len(x); m, sd = x.gap.mean(), x.gap.std(ddof=1); t = stats.t.ppf(0.975, n - 1)
        lo, hi = m - t * sd / np.sqrt(n), m + t * sd / np.sqrt(n)
        out.append(dict(config=cfg, n=n, gap_mean=m, ci95_low=lo, ci95_high=hi, seeds_positive=f'{int((x.gap > 0).sum())}/{n}',
                        noise_mean=x.retrain_noise.mean(), adv_orig=x.adv_orig.mean(), adv_retA=x.adv_retA.mean(),
                        tpr_fpr_orig=x.tpr_fpr_orig.mean(), tpr_fpr_retA=x.tpr_fpr_retA.mean(), test_acc=x.test_acc_orig.mean(),
                        confirmed=bool(lo > 0 and (x.gap > 0).sum() >= 4 and m >= 3 * x.retrain_noise.mean())))
    o = pd.DataFrame(out); print(o.round(4).to_string(index=False)); o.to_csv(f'{RES}/confirmation_summary.csv', index=False)
