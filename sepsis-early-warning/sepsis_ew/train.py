"""Training: CV terkelompok, pemilihan bobot ensemble, simpan artefak.

Jalankan dari folder sepsis-early-warning:
    python -m sepsis_ew.train
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold
from xgboost import XGBClassifier

from sepsis_ew.models import (FEATURES, SoftVotingEnsemble, TorchMLPClassifier,
                              build_preprocessor)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "sepsis_emr_data.csv"
OUT = BASE / "artifacts"
SEED = 42
N_SPLITS = 5


def make_xgb():
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                         random_state=SEED, n_jobs=1)


def make_ann():
    return TorchMLPClassifier(hidden=(32, 16), seed=SEED)


def load():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    # Data hanya berisi ~100 profil unik yang diulang ~17x -> kelompokkan baris kembar
    # supaya satu profil tidak terbelah ke data latih DAN data validasi (data leakage).
    groups = df.groupby(FEATURES, dropna=False).ngroup().to_numpy()
    return df[FEATURES], df["sepsis_risk"].to_numpy(), groups


def oof_probs(X, y, splits, lactate_flag):
    """Probabilitas out-of-fold XGB dan ANN. Preprocessor di-fit hanya pada data latih fold."""
    p_xgb, p_ann = np.zeros(len(y)), np.zeros(len(y))
    for tr, va in splits:
        pre = build_preprocessor(lactate_flag).fit(X.iloc[tr])
        Ztr, Zva = pre.transform(X.iloc[tr]), pre.transform(X.iloc[va])
        p_xgb[va] = make_xgb().fit(Ztr, y[tr]).predict_proba(Zva)[:, 1]
        p_ann[va] = clone(make_ann()).fit(Ztr, y[tr]).predict_proba(Zva)[:, 1]
    return p_xgb, p_ann


def fold_metrics(y, p, splits, thr=0.5):
    rows = []
    for _, va in splits:
        pred = (p[va] >= thr).astype(int)
        rows.append({
            "auc": roc_auc_score(y[va], p[va]),
            "precision": precision_score(y[va], pred, zero_division=0),
            "recall": recall_score(y[va], pred),
            "f1": f1_score(y[va], pred),
        })
    m = pd.DataFrame(rows)
    return {k: {"mean": round(float(m[k].mean()), 4), "std": round(float(m[k].std()), 4)}
            for k in m}


def main():
    X, y, groups = load()
    print(f"Data: {len(y)} baris, {len(set(groups))} profil unik, sepsis={y.mean():.1%}")

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(cv.split(X, y, groups))

    # 1) Perbandingan: CV biasa (bocor) vs CV terkelompok, pakai XGB saja
    leaky = list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(X))
    px_leaky, _ = oof_probs(X, y, leaky, True)
    print(f"AUC XGB dengan KFold biasa (bocor): {roc_auc_score(y, px_leaky):.4f}")

    # 2) Pakai penanda 'laktat kosong' atau tidak? (CV terkelompok, rata-rata 50/50)
    results = {}
    for flag in (True, False):
        px, pa = oof_probs(X, y, splits, flag)
        results[flag] = (px, pa)
        print(f"lactate_flag={flag}: AUC xgb={roc_auc_score(y, px):.4f} "
              f"ann={roc_auc_score(y, pa):.4f} ens50={roc_auc_score(y, (px + pa) / 2):.4f}")

    use_flag = False  # lihat README: tidak bergantung pada kebiasaan pemeriksaan laktat
    px, pa = results[use_flag]

    # 3) Pilih bobot XGB dari AUC out-of-fold; kalau selisih < 0.001, pilih yang paling dekat 50/50
    grid = np.round(np.linspace(0, 1, 11), 1)
    aucs = {float(w): roc_auc_score(y, w * px + (1 - w) * pa) for w in grid}
    best = max(aucs.values())
    near = [w for w, a in aucs.items() if best - a < 1e-3]
    w_xgb = min(near, key=lambda w: abs(w - 0.5))
    print(f"Bobot XGB terpilih: {w_xgb} (AUC terbaik {best:.4f})")

    p_ens = w_xgb * px + (1 - w_xgb) * pa
    metrics = {
        "n_rows": int(len(y)), "n_unique_profiles": int(len(set(groups))),
        "cv": f"StratifiedGroupKFold k={N_SPLITS}, grup = profil fitur",
        "lactate_missing_flag": use_flag,
        "weight_xgb": w_xgb, "weight_ann": round(1 - w_xgb, 1),
        "weight_search_auc": {str(k): round(v, 4) for k, v in aucs.items()},
        "xgb_only": fold_metrics(y, px, splits),
        "ann_only": fold_metrics(y, pa, splits),
        "ensemble_threshold_0.5": fold_metrics(y, p_ens, splits, 0.5),
        "ensemble_threshold_0.3": fold_metrics(y, p_ens, splits, 0.3),
        "kfold_biasa_bocor_auc_xgb": round(float(roc_auc_score(y, px_leaky)), 4),
    }
    for name in ("xgb_only", "ann_only", "ensemble_threshold_0.5", "ensemble_threshold_0.3"):
        print(name, {k: v["mean"] for k, v in metrics[name].items()})

    # 4) Latih ulang pada seluruh data dan simpan artefak
    pre = build_preprocessor(use_flag).fit(X)
    Z = pre.transform(X)
    xgb = make_xgb().fit(Z, y)
    ann = make_ann().fit(Z, y)
    ens = SoftVotingEnsemble(pre, xgb, ann, weight_xgb=w_xgb, threshold=0.5)
    OUT.mkdir(exist_ok=True)
    joblib.dump(pre, OUT / "preprocessor.joblib")
    joblib.dump(xgb, OUT / "xgb_model.joblib")
    joblib.dump(ann, OUT / "ann_model.joblib")
    joblib.dump(ens, OUT / "ensemble.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Artefak tersimpan di", OUT)


if __name__ == "__main__":
    main()
