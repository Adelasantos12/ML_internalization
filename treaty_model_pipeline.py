#!/usr/bin/env python3
"""Train and evaluate treaty internalization capacity models for LATAM."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SEED = 42
random.seed(SEED)


def to_float(value: str) -> float | None:
    if value is None:
        return None
    v = value.strip()
    if v == "" or v.lower() in {"na", "nan", "null", "none"}:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_table(filename: str) -> list[dict[str, float | str]]:
    rows = []
    with (DATA_DIR / filename).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {}
            for k, v in row.items():
                fv = to_float(v)
                clean[k] = fv if fv is not None else (v.strip() if isinstance(v, str) else v)
            rows.append(clean)
    return rows


def key(row: dict) -> tuple[str, int]:
    return str(row["iso3c"]), int(float(row["year"]))


def merge_features() -> tuple[list[dict], list[str]]:
    spar = load_table("spar_clean_latam.csv")
    political = {key(r): r for r in load_table("political_panel_latam_2010_2020.csv")}
    vdem = {key(r): r for r in load_table("vdem_model1_subset_latam.csv")}
    wgi = {key(r): r for r in load_table("wgi_capa2_latam_2000_2024.csv")}
    gdp = {key(r): r for r in load_table("gdp_pc_wb_latam_2000_2024.csv")}
    health = {key(r): r for r in load_table("health_exp_gdp_wb_latam_2000_2024.csv")}
    uhc = {key(r): r for r in load_table("uhc_latam_2010_2023.csv")}

    by_country: dict[str, list[dict]] = {}
    for row in spar:
        by_country.setdefault(str(row["iso3c"]), []).append(row)

    outcome_rows = []
    for country, rows in by_country.items():
        rows = sorted(rows, key=lambda r: int(float(r["year"])))
        for i in range(1, len(rows)):
            prev, cur = rows[i - 1], rows[i]
            prev_spar = to_float(str(prev.get("spar_cap_law", "")))
            cur_spar = to_float(str(cur.get("spar_cap_law", "")))
            if prev_spar is None or cur_spar is None:
                continue
            k = key(cur)
            joined = {
                "iso3c": country,
                "year": int(float(cur["year"])),
                "spar_score": cur_spar,
                "spar_prev": prev_spar,
                "spar_delta": cur_spar - prev_spar,
            }
            for source in (political, vdem, wgi, gdp, health, uhc):
                if k in source:
                    for col, val in source[k].items():
                        if col in {"iso3c", "year"}:
                            continue
                        joined[col] = val
            outcome_rows.append(joined)

    feature_cols = [
        "pres_election_year",
        "leg_election_year",
        "yrsoffc",
        "exec_turnover",
        "gov_seat_share",
        "gov_majority",
        "oppmajh",
        "checks",
        "polariz",
        "years_since_pres_election",
        "years_since_leg_election",
        "v2xlg_legcon",
        "v2edideol_mean",
        "v2cacamps",
        "cc_est",
        "ge_est",
        "pv_est",
        "rq_est",
        "rl_est",
        "va_est",
        "state_capacity_wgi",
        "log_gdp_pc",
        "health_exp_gdp",
        "uhc",
    ]
    return outcome_rows, feature_cols


def impute_and_matrix(rows: list[dict], feature_cols: list[str]) -> tuple[list[list[float]], list[int], list[dict], dict[str, float]]:
    means: dict[str, float] = {}
    for col in feature_cols:
        vals = [float(r[col]) for r in rows if isinstance(r.get(col), (int, float))]
        means[col] = sum(vals) / len(vals) if vals else 0.0

    X, y, keep_rows = [], [], []
    for r in rows:
        target = 1 if r["spar_delta"] > 0 else 0
        vec = [float(r[c]) if isinstance(r.get(c), (int, float)) else means[c] for c in feature_cols]
        if any(math.isnan(v) or math.isinf(v) for v in vec):
            continue
        X.append(vec)
        y.append(target)
        keep_rows.append(r)
    return X, y, keep_rows, means


def accuracy(y, p):
    return sum(1 for a, b in zip(y, p) if a == b) / len(y)


def precision_recall_f1(y, p):
    tp = sum(1 for a, b in zip(y, p) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, p) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, p) if a == 1 and b == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def roc_auc(y_true, y_score):
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = sum(i + 1 for i, (_, y) in enumerate(pairs) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


@dataclass
class Node:
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    value: float = 0.0


class CARTClassifier:
    def __init__(self, max_depth=3, min_samples_split=8, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        self.root = self._grow(X, y, 0)

    def _gini(self, y):
        if not y:
            return 0.0
        p = sum(y) / len(y)
        return 1 - p * p - (1 - p) * (1 - p)

    def _split_score(self, left_y, right_y):
        n = len(left_y) + len(right_y)
        return (len(left_y) / n) * self._gini(left_y) + (len(right_y) / n) * self._gini(right_y)

    def _grow(self, X, y, depth):
        node = Node(value=sum(y) / len(y))
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
            return node

        n_features = len(X[0])
        feat_idx = list(range(n_features))
        if self.max_features:
            feat_idx = random.sample(feat_idx, min(self.max_features, n_features))

        best = (float("inf"), None, None, None)
        for f in feat_idx:
            col = [row[f] for row in X]
            uniq = sorted(set(col))
            if len(uniq) < 2:
                continue
            if len(uniq) > 12:
                step = len(uniq) // 12
                uniq = uniq[::step]
            thresholds = [(a + b) / 2 for a, b in zip(uniq[:-1], uniq[1:])]
            for t in thresholds:
                left = [i for i, v in enumerate(col) if v <= t]
                right = [i for i, v in enumerate(col) if v > t]
                if not left or not right:
                    continue
                ly, ry = [y[i] for i in left], [y[i] for i in right]
                score = self._split_score(ly, ry)
                if score < best[0]:
                    best = (score, f, t, (left, right))

        if best[1] is None:
            return node

        _, f, t, (left_idx, right_idx) = best
        node.feature, node.threshold = f, t
        node.left = self._grow([X[i] for i in left_idx], [y[i] for i in left_idx], depth + 1)
        node.right = self._grow([X[i] for i in right_idx], [y[i] for i in right_idx], depth + 1)
        return node

    def _predict_one(self, x, node):
        if node.feature is None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict_proba(self, X):
        return [self._predict_one(x, self.root) for x in X]


class CARTRegressor(CARTClassifier):
    def _mse(self, y):
        if not y:
            return 0.0
        mu = sum(y) / len(y)
        return sum((v - mu) ** 2 for v in y) / len(y)

    def _split_score(self, left_y, right_y):
        n = len(left_y) + len(right_y)
        return (len(left_y) / n) * self._mse(left_y) + (len(right_y) / n) * self._mse(right_y)


class RandomForestClassifier:
    def __init__(self, n_estimators=80, max_depth=4, min_samples_split=8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        m = max(1, int(math.sqrt(len(X[0]))))
        for _ in range(self.n_estimators):
            idx = [random.randrange(len(X)) for _ in range(len(X))]
            xb, yb = [X[i] for i in idx], [y[i] for i in idx]
            tree = CARTClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=m)
            tree.fit(xb, yb)
            self.trees.append(tree)

    def predict_proba(self, X):
        probs = [0.0] * len(X)
        for tree in self.trees:
            tp = tree.predict_proba(X)
            probs = [a + b for a, b in zip(probs, tp)]
        return [p / len(self.trees) for p in probs]


class GradientBoostingClassifier:
    def __init__(self, n_estimators=120, learning_rate=0.05, max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.base = 0.0
        self.trees = []

    @staticmethod
    def _sigmoid(z):
        z = max(min(z, 35), -35)
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        p0 = min(max(sum(y) / len(y), 1e-5), 1 - 1e-5)
        self.base = math.log(p0 / (1 - p0))
        F = [self.base] * len(y)
        self.trees = []
        for _ in range(self.n_estimators):
            p = [self._sigmoid(f) for f in F]
            residual = [yy - pp for yy, pp in zip(y, p)]
            tree = CARTRegressor(max_depth=self.max_depth, min_samples_split=10, max_features=None)
            tree.fit(X, residual)
            update = tree.predict_proba(X)
            F = [f + self.learning_rate * u for f, u in zip(F, update)]
            self.trees.append(tree)

    def predict_proba(self, X):
        F = [self.base] * len(X)
        for tree in self.trees:
            update = tree.predict_proba(X)
            F = [f + self.learning_rate * u for f, u in zip(F, update)]
        return [self._sigmoid(f) for f in F]


def kfold_cv_score(model_factory: Callable[[], object], X, y, k=5):
    idx = list(range(len(X)))
    random.shuffle(idx)
    fold_size = len(X) // k
    aucs = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else len(X)
        test_idx = idx[start:end]
        train_idx = idx[:start] + idx[end:]
        Xtr, ytr = [X[j] for j in train_idx], [y[j] for j in train_idx]
        Xte, yte = [X[j] for j in test_idx], [y[j] for j in test_idx]
        model = model_factory()
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)
        aucs.append(roc_auc(yte, p))
    return sum(aucs) / len(aucs)


def shapley_approx(model, X, feature_names, samples=25, mc=40):
    sample_rows = X[: min(samples, len(X))]
    importances = {f: 0.0 for f in feature_names}
    for x in sample_rows:
        for _ in range(mc):
            perm = list(range(len(feature_names)))
            random.shuffle(perm)
            z = random.choice(X)[:]
            prev = model.predict_proba([z])[0]
            for j in perm:
                z[j] = x[j]
                new = model.predict_proba([z])[0]
                importances[feature_names[j]] += abs(new - prev)
                prev = new
    scale = len(sample_rows) * mc
    return {k: v / scale for k, v in sorted(importances.items(), key=lambda kv: kv[1], reverse=True)}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows, features = merge_features()
    X, y, rows_used, means = impute_and_matrix(rows, features)

    idx = list(range(len(X)))
    random.shuffle(idx)
    split = int(0.7 * len(idx))
    tr_idx, te_idx = idx[:split], idx[split:]
    Xtr, ytr = [X[i] for i in tr_idx], [y[i] for i in tr_idx]
    Xte, yte = [X[i] for i in te_idx], [y[i] for i in te_idx]

    rf_grid = [(60, 3), (80, 4), (120, 4)]
    gbm_grid = [(80, 0.05), (120, 0.05), (120, 0.08)]

    best_rf = max(rf_grid, key=lambda p: kfold_cv_score(lambda: RandomForestClassifier(n_estimators=p[0], max_depth=p[1]), Xtr, ytr))
    best_gbm = max(gbm_grid, key=lambda p: kfold_cv_score(lambda: GradientBoostingClassifier(n_estimators=p[0], learning_rate=p[1]), Xtr, ytr))

    rf = RandomForestClassifier(n_estimators=best_rf[0], max_depth=best_rf[1])
    gbm = GradientBoostingClassifier(n_estimators=best_gbm[0], learning_rate=best_gbm[1])
    rf.fit(Xtr, ytr)
    gbm.fit(Xtr, ytr)

    evals = {}
    for name, model in [("random_forest", rf), ("gbm", gbm)]:
        proba = model.predict_proba(Xte)
        pred = [1 if p >= 0.5 else 0 for p in proba]
        prec, rec, f1 = precision_recall_f1(yte, pred)
        evals[name] = {
            "accuracy": round(accuracy(yte, pred), 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc(yte, proba), 4),
        }

    best_name = max(evals, key=lambda m: evals[m]["roc_auc"])
    best_model = gbm if best_name == "gbm" else rf
    shap_values = shapley_approx(best_model, Xte, features)

    tobacco = load_table("tobacco_law_enforce_index_latam_CORE_FULL.csv")
    tobacco_map = {key(r): r for r in tobacco}
    tobacco_vals = [to_float(str(r.get("tobacco_law_enforce_index_full", ""))) for r in tobacco]
    tobacco_vals = [v for v in tobacco_vals if v is not None]
    tobacco_median = sorted(tobacco_vals)[len(tobacco_vals) // 2] if tobacco_vals else None
    ext_y, ext_p = [], []
    for i in te_idx:
        r = rows_used[i]
        k = (r["iso3c"], r["year"])
        if k in tobacco_map and tobacco_median is not None:
            now = to_float(str(tobacco_map[k]["tobacco_law_enforce_index_full"]))
            if now is None:
                continue
            ext_y.append(1 if now >= tobacco_median else 0)
            ext_p.append(best_model.predict_proba([X[i]])[0])
    external_auc = roc_auc(ext_y, ext_p) if ext_y else None

    results = {
        "dataset": {
            "observations": len(X),
            "train_size": len(Xtr),
            "test_size": len(Xte),
            "positive_rate": round(sum(y) / len(y), 4),
        },
        "best_hyperparameters": {
            "random_forest": {"n_estimators": best_rf[0], "max_depth": best_rf[1]},
            "gbm": {"n_estimators": best_gbm[0], "learning_rate": best_gbm[1]},
        },
        "metrics": evals,
        "selected_model": best_name,
        "top_factors_shap": dict(list(shap_values.items())[:10]),
        "external_validation_tobacco_auc": round(external_auc, 4) if external_auc is not None else None,
        "imputation_means": means,
    }

    with (OUTPUT_DIR / "model_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with (OUTPUT_DIR / "model_results.md").open("w", encoding="utf-8") as f:
        f.write("# Treaty Internalization Capacity Model (LATAM)\n\n")
        f.write(f"- Observations: {len(X)} (train={len(Xtr)}, test={len(Xte)})\n")
        f.write(f"- Target: annual improvement in SPAR legal capacity (`spar_delta > 0`)\n")
        f.write(f"- Selected model: **{best_name}**\n\n")
        f.write("## Performance\n")
        for model_name, metric in evals.items():
            f.write(f"- {model_name}: {metric}\n")
        f.write("\n## Top SHAP-like factors (mean |contribution|)\n")
        for k, v in list(shap_values.items())[:10]:
            f.write(f"- {k}: {v:.5f}\n")
        f.write("\n## External validation\n")
        f.write(f"- Tobacco implementation transfer AUC: {results['external_validation_tobacco_auc']}\n")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
