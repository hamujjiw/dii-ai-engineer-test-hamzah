"""Komponen model: preprocessing, ANN (PyTorch) dan ensemble soft-voting.

File ini dipakai bersama oleh skrip training dan API. Kelas-kelas di sini
di-pickle ke artefak, jadi path import-nya (sepsis_ew.models) harus sama
saat training dan saat API dijalankan.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

FEATURES = [
    "heart_rate",
    "respiratory_rate",
    "temperature",
    "wbc_count",
    "lactate_level",
    "age",
    "num_comorbidities",
]


def build_preprocessor(lactate_missing_flag: bool = True) -> Pipeline:
    """Isi nilai kosong dengan median, lalu samakan skala (mean 0, std 1).

    lactate_missing_flag=True menambah kolom 0/1 "lactate kosong?" supaya
    informasi bahwa laktat tidak diperiksa tidak hilang saat diisi median.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=lactate_missing_flag)),
            ("scale", StandardScaler()),
        ]
    )


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """ANN sederhana (MLP) dengan antarmuka scikit-learn: fit / predict_proba."""

    def __init__(self, hidden=(32, 16), lr=1e-3, epochs=150, batch_size=64,
                 weight_decay=1e-4, dropout=0.1, seed=42):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.seed = seed

    def _build(self, n_in):
        layers, prev = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))  # 1 neuron output -> logit
        return nn.Sequential(*layers)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        y = torch.tensor(np.asarray(y, dtype=np.float32)).unsqueeze(1)
        self.classes_ = np.array([0, 1])
        self.module_ = self._build(X.shape[1])
        opt = torch.optim.Adam(self.module_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        n = len(X)
        self.module_.train()
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for i in range(0, n, self.batch_size):
                idx = torch.tensor(order[i:i + self.batch_size])
                opt.zero_grad()
                loss_fn(self.module_(X[idx]), y[idx]).backward()
                opt.step()
        self.module_.eval()
        return self

    def predict_proba(self, X):
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            p = torch.sigmoid(self.module_(X)).numpy().ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SoftVotingEnsemble:
    """Gabungan preprocessor + XGBoost + ANN. Probabilitas dirata-rata berbobot."""

    def __init__(self, preprocessor, xgb_model, ann_model, weight_xgb=0.5, threshold=0.5):
        self.preprocessor = preprocessor
        self.xgb_model = xgb_model
        self.ann_model = ann_model
        self.weight_xgb = weight_xgb
        self.threshold = threshold

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilitas sepsis (kelas 1) untuk tiap baris df."""
        Z = self.preprocessor.transform(df[FEATURES])
        p_xgb = self.xgb_model.predict_proba(Z)[:, 1]
        p_ann = self.ann_model.predict_proba(Z)[:, 1]
        return self.weight_xgb * p_xgb + (1 - self.weight_xgb) * p_ann

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= self.threshold).astype(int)
