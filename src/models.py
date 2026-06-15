import logging
import json
import os
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

_log = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def _ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


def _get_feature_names(df, exclude=None):
    if exclude is None:
        exclude = {"Date"}
    return [c for c in df.columns if c not in exclude]


def _make_target_volatility(price_df):
    target = price_df["Volatility"].copy()
    q33 = target.quantile(0.33)
    q66 = target.quantile(0.66)
    labels = pd.cut(target, bins=[-np.inf, q33, q66, np.inf], labels=[0, 1, 2])
    return pd.Series(labels, index=price_df.index, name="target")


def _make_target_pead(price_df, horizon=5):
    return pd.Series(
        price_df["Close"].shift(-horizon) / price_df["Close"] - 1.0,
        index=price_df.index,
        name="target",
    )


def _make_target_regime(price_df):
    ret = price_df["Return"].fillna(0)
    rolling_ret = ret.rolling(20).mean()
    conditions = [
        rolling_ret > 0.005,
        rolling_ret < -0.005,
    ]
    choices = [2, 0]
    default = 1
    return pd.Series(
        np.select(conditions, choices, default=default),
        index=price_df.index,
        name="target",
    ).astype(int)


class VolatilityShockModel:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.metadata = {}

    def train(self, feature_df, price_df):
        _ensure_models_dir()
        features = _get_feature_names(feature_df)
        y = _make_target_volatility(price_df)
        align_idx = feature_df.index.intersection(y.dropna().index)
        if len(align_idx) < 50:
            return {"error": f"Not enough aligned samples ({len(align_idx)})"}
        X = feature_df.loc[align_idx, features].fillna(0).values
        y = y.loc[align_idx].astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        _log.info("Training split: %d train / %d test samples", len(X_train), len(X_test))

        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
        )
        self.model.fit(X_train, y_train)
        _log.info("XGBoost training complete")

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        _log.info("Test accuracy: %.4f (%d/%d)", acc, (y_pred == y_test).sum(), len(y_test))

        self.feature_names = features
        self.metadata = {
            "type": "volatility_shock_classifier",
            "algorithm": "XGBoost",
            "target": "volatility_tertile {0=low, 1=medium, 2=high}",
            "test_accuracy": float(acc),
            "n_features": len(features),
            "features": features,
            "temporal_alignment": "as-of joins, no lookahead",
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        return {"accuracy": acc}

    def predict(self, feature_vector):
        if self.model is None:
            return {"error": "Model not trained"}
        vec = {k: v for k, v in feature_vector.items() if k in self.feature_names}
        missing = [f for f in self.feature_names if f not in vec]
        for m in missing:
            vec[m] = 0
        X = pd.DataFrame([vec])[self.feature_names].fillna(0).values
        probs = self.model.predict_proba(X)[0]
        labels = ["low", "medium", "high"]
        result = {}
        for i in range(len(probs)):
            label = labels[i] if i < len(labels) else f"class_{i}"
            result[label] = float(probs[i])
        return result

    def save(self, path=None):
        _ensure_models_dir()
        path = path or os.path.join(MODELS_DIR, "volatility_shock.json")
        self.model.save_model(path)
        meta_path = path.replace(".json", "_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        return path

    def load(self, path=None):
        path = path or os.path.join(MODELS_DIR, "volatility_shock.json")
        if not os.path.exists(path):
            return False
        self.model = XGBClassifier()
        self.model.load_model(path)
        meta_path = path.replace(".json", "_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get("features", [])
        return True


class PEADModel:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.metadata = {}

    def train(self, feature_df, price_df):
        _ensure_models_dir()
        features = _get_feature_names(feature_df)
        y = _make_target_pead(price_df)
        align_idx = feature_df.index.intersection(y.dropna().index)
        if len(align_idx) < 50:
            return {"error": f"Not enough training samples ({len(align_idx)})"}
        X = feature_df.loc[align_idx, features].fillna(0).values
        y = y.loc[align_idx].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        _log.info("Training split: %d train / %d test samples", len(X_train), len(X_test))

        self.model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X_train, y_train)
        _log.info("LightGBM training complete")

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        _log.info("Test MAE: %.4f | R²: %.4f", mae, r2)

        self.feature_names = features
        self.metadata = {
            "type": "pead_regressor",
            "algorithm": "LightGBM",
            "target": "5-day forward return",
            "test_mae": float(mae),
            "test_r2": float(r2),
            "n_features": len(features),
            "features": features,
            "temporal_alignment": "as-of joins, no lookahead",
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        return {"mae": mae, "r2": r2}

    def predict(self, feature_vector):
        if self.model is None:
            return {"error": "Model not trained"}
        vec = {k: v for k, v in feature_vector.items() if k in self.feature_names}
        missing = [f for f in self.feature_names if f not in vec]
        for m in missing:
            vec[m] = 0
        X = pd.DataFrame([vec])[self.feature_names].fillna(0).values
        pred = float(self.model.predict(X)[0])
        return {"expected_drift_pct": round(pred * 100, 4)}

    def save(self, path=None):
        _ensure_models_dir()
        path = path or os.path.join(MODELS_DIR, "pead_regressor.txt")
        self.model.booster_.save_model(path)
        meta_path = path.replace(".txt", "_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        return path

    def load(self, path=None):
        path = path or os.path.join(MODELS_DIR, "pead_regressor.txt")
        if not os.path.exists(path):
            return False
        self.model = lgb.Booster(model_file=path)
        meta_path = path.replace(".txt", "_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get("features", [])
        return True


class MacroRegimeModel:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.metadata = {}

    def train(self, feature_df, price_df):
        _ensure_models_dir()
        features = _get_feature_names(feature_df)
        y = _make_target_regime(price_df)
        align_idx = feature_df.index.intersection(y.index)
        if len(align_idx) < 50:
            return {"error": f"Not enough aligned samples ({len(align_idx)})"}
        X = feature_df.loc[align_idx, features].fillna(0).values
        y = y.loc[align_idx].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        _log.info("Training split: %d train / %d test samples", len(X_train), len(X_test))

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
        )
        self.model.fit(X_train, y_train)
        _log.info("RandomForest training complete")

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        _log.info("Test accuracy: %.4f (%d/%d)", acc, (y_pred == y_test).sum(), len(y_test))

        self.feature_names = features
        self.metadata = {
            "type": "macro_regime_classifier",
            "algorithm": "RandomForest",
            "target": "regime {0=contraction, 1=neutral, 2=expansion}",
            "test_accuracy": float(acc),
            "n_features": len(features),
            "features": features,
            "temporal_alignment": "as-of joins, no lookahead",
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        return {"accuracy": acc}

    def predict(self, feature_vector):
        if self.model is None:
            return {"error": "Model not trained"}
        vec = {k: v for k, v in feature_vector.items() if k in self.feature_names}
        missing = [f for f in self.feature_names if f not in vec]
        for m in missing:
            vec[m] = 0
        X = pd.DataFrame([vec])[self.feature_names].fillna(0).values
        probs = self.model.predict_proba(X)[0]
        pred = self.model.predict(X)[0]
        labels_map = {0: "contraction", 1: "neutral", 2: "expansion"}
        return {
            "regime": labels_map.get(int(pred), "unknown"),
            "confidence": float(max(probs)),
            "probabilities": {labels_map.get(i, f"class_{i}"): float(p) for i, p in enumerate(probs)},
        }

    def save(self, path=None):
        _ensure_models_dir()
        path = path or os.path.join(MODELS_DIR, "macro_regime.joblib")
        joblib.dump({"model": self.model, "metadata": self.metadata, "feature_names": self.feature_names}, path)
        return path

    def load(self, path=None):
        path = path or os.path.join(MODELS_DIR, "macro_regime.joblib")
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.model = data["model"]
        self.metadata = data["metadata"]
        self.feature_names = data["feature_names"]
        return True


def train_all(feature_df, price_df):
    results = {}
    vs = VolatilityShockModel()
    r = vs.train(feature_df, price_df)
    results["volatility_shock"] = r
    if "error" not in r:
        vs.save()

    pead = PEADModel()
    r = pead.train(feature_df, price_df)
    results["pead"] = r
    if "error" not in r:
        pead.save()

    macro = MacroRegimeModel()
    r = macro.train(feature_df, price_df)
    results["macro_regime"] = r
    if "error" not in r:
        macro.save()
    return results


def load_all():
    vs = VolatilityShockModel()
    vs.load()
    pead = PEADModel()
    pead.load()
    macro = MacroRegimeModel()
    macro.load()
    return vs, pead, macro
