"""
Forecasting Models Module
==========================
Implements solar generation and electricity demand forecasting.

Models:
  1. Persistence Baseline
  2. Linear Regression
  3. Random Forest Regressor
  4. XGBoost Regressor
  5. LSTM (optional, enabled via use_lstm=True)

Evaluation Metrics:
  - MAE, RMSE, R²
  - MAPE only reported when no zero values exist in actuals
  - NMAE (normalized) for solar (avoids MAPE pitfall)

All models use chronological train/val/test splits (no random shuffling).
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("forecasting")

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS = BASE_DIR / "outputs"
METRICS_DIR = OUTPUTS / "metrics"
FORECASTS_DIR = OUTPUTS / "forecasts"

for d in [MODELS_DIR, METRICS_DIR, FORECASTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── Evaluation utilities ───────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str = "") -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    rng = y_true.max() - y_true.min()
    nmae = mae / rng if rng > 0 else float("nan")

    # MAPE only when no zeros
    if (y_true == 0).any():
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    metrics = {"model": name, "mae": round(mae, 4), "rmse": round(rmse, 4),
               "r2": round(r2, 4), "nmae": round(nmae, 4), "mape_pct": round(mape, 2) if not np.isnan(mape) else None}
    log.info("[%s] MAE=%.3f RMSE=%.3f R²=%.4f NMAE=%.4f", name, mae, rmse, r2, nmae)
    return metrics


# ── Persistence Baseline ──────────────────────────────────────────────────────
class PersistenceModel:
    """Naive persistence: ŷ(t) = y(t-1)."""

    def __init__(self, lag: int = 1):
        self.lag = lag
        self.name = f"persistence_lag{lag}"

    def fit(self, X, y):  # noqa: no-op
        return self

    def predict(self, X: pd.DataFrame, y_full: pd.Series) -> np.ndarray:
        return y_full.shift(self.lag).bfill().values


# ── Linear Regression ─────────────────────────────────────────────────────────
class LinearModel:
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.name = "linear_ridge"

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs).clip(min=0)


# ── Random Forest ─────────────────────────────────────────────────────────────
class RFModel:
    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state,
            n_jobs=-1, min_samples_leaf=3,
        )
        self.name = "random_forest"

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).clip(min=0)

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": feature_names, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)


# ── XGBoost ──────────────────────────────────────────────────────────────────
class XGBModel:
    def __init__(self, **kwargs):
        try:
            from xgboost import XGBRegressor
            params = dict(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1, early_stopping_rounds=30,
            )
            params.update(kwargs)
            self.model = XGBRegressor(**params)
            self.name = "xgboost"
            self._available = True
        except ImportError:
            log.warning("XGBoost not available.")
            self._available = False
            self.name = "xgboost_unavailable"

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        if not self._available:
            return self
        eval_set = [(X_val, y_val)] if X_val is not None else None
        verbose = False
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._available:
            return np.zeros(len(X))
        return self.model.predict(X).clip(min=0)

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        if not self._available:
            return pd.DataFrame()
        scores = self.model.feature_importances_
        return pd.DataFrame(
            {"feature": feature_names, "importance": scores}
        ).sort_values("importance", ascending=False)


# ── LSTM ──────────────────────────────────────────────────────────────────────
class LSTMModel:
    """Sequence-to-one LSTM for time-series forecasting."""

    def __init__(self, lookback: int = 96, units: int = 64, epochs: int = 30, batch_size: int = 64):
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.name = "lstm"
        self._model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def _build_sequences(self, X: np.ndarray, y: np.ndarray | None = None):
        Xs, ys = [], []
        for i in range(self.lookback, len(X)):
            Xs.append(X[i - self.lookback:i])
            if y is not None:
                ys.append(y[i])
        return np.array(Xs), np.array(ys) if y is not None else None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            log.warning("TensorFlow not available. LSTM model skipped.")
            return self

        X_s = self.scaler_X.fit_transform(X_train)
        y_s = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        Xs, ys = self._build_sequences(X_s, y_s)

        self._model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=(self.lookback, X_train.shape[1])),
            Dropout(0.2),
            LSTM(self.units // 2),
            Dropout(0.2),
            Dense(1),
        ])
        self._model.compile(optimizer="adam", loss="mse")

        callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
        val_data = None
        if X_val is not None and y_val is not None:
            Xv = self.scaler_X.transform(X_val)
            yv = self.scaler_y.transform(y_val.reshape(-1, 1)).ravel()
            Xvs, yvs = self._build_sequences(Xv, yv)
            val_data = (Xvs, yvs)

        self._model.fit(Xs, ys, epochs=self.epochs, batch_size=self.batch_size,
                        validation_data=val_data, callbacks=callbacks, verbose=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.zeros(len(X))
        X_s = self.scaler_X.transform(X)
        Xs, _ = self._build_sequences(X_s)
        preds = self._model.predict(Xs, verbose=0)
        preds = self.scaler_y.inverse_transform(preds).ravel()
        # Prepend NaNs for the lookback period
        full = np.full(len(X), np.nan)
        full[self.lookback:] = preds
        return np.clip(full, 0, None)


# ── Training Pipeline ─────────────────────────────────────────────────────────
def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    ts_col: str,
    task_name: str = "solar",
    use_lstm: bool = False,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict:
    """
    Full train/val/test pipeline for all models.

    Returns dict with per-model metrics and predictions.
    """
    from src.data.feature_engineering import chronological_split

    # Deduplicate while preserving order
    seen = set()
    feature_cols_dedup = [c for c in feature_cols if not (c in seen or seen.add(c))]

    # Drop rows with NaN in features or target
    required = list(dict.fromkeys(feature_cols_dedup + [target_col, ts_col]))  # dedup
    avail = [c for c in required if c in df.columns]
    avail = list(dict.fromkeys(avail))  # ensure unique
    df_clean = df[avail].dropna(subset=[target_col])
    df_clean = df_clean.sort_values(ts_col).reset_index(drop=True)

    feat_avail = [c for c in feature_cols_dedup if c in df_clean.columns and c != target_col and c != ts_col]
    feat_avail = list(dict.fromkeys(feat_avail))  # final dedup
    log.info("[%s] Features used: %d/%d", task_name, len(feat_avail), len(feature_cols_dedup))

    train, val, test = chronological_split(df_clean, ts_col, train_frac, val_frac)

    # Fill NaN in features with median from train set
    medians = train[feat_avail].median()
    for split in [train, val, test]:
        split[feat_avail] = split[feat_avail].fillna(medians)

    X_train = train[feat_avail].values
    y_train = train[target_col].values
    X_val = val[feat_avail].values
    y_val = val[target_col].values
    X_test = test[feat_avail].values
    y_test = test[target_col].values

    all_metrics = []
    all_preds = {"timestamp": test[ts_col].values, "actual": y_test}

    # ── Persistence ──────────────────────────────────────────────────────────
    persist = PersistenceModel(lag=1)
    y_persist = df_clean[target_col].shift(1).bfill().values
    test_persist = y_persist[len(train) + len(val):]
    m = compute_metrics(y_test, test_persist, "persistence")
    all_metrics.append(m)
    all_preds["persistence"] = test_persist

    # ── Linear ──────────────────────────────────────────────────────────────
    lin = LinearModel()
    lin.fit(X_train, y_train)
    y_lin = lin.predict(X_test)
    m = compute_metrics(y_test, y_lin, "linear_ridge")
    all_metrics.append(m)
    all_preds["linear_ridge"] = y_lin
    joblib.dump(lin, MODELS_DIR / f"{task_name}_linear.pkl")

    # ── Random Forest ───────────────────────────────────────────────────────
    rf = RFModel()
    rf.fit(X_train, y_train)
    y_rf = rf.predict(X_test)
    m = compute_metrics(y_test, y_rf, "random_forest")
    all_metrics.append(m)
    all_preds["random_forest"] = y_rf
    joblib.dump(rf, MODELS_DIR / f"{task_name}_rf.pkl")
    fi = rf.feature_importance(feat_avail)
    fi.to_csv(METRICS_DIR / f"{task_name}_rf_feature_importance.csv", index=False)

    # ── XGBoost ─────────────────────────────────────────────────────────────
    xgb = XGBModel()
    xgb.fit(X_train, y_train, X_val, y_val)
    y_xgb = xgb.predict(X_test)
    m = compute_metrics(y_test, y_xgb, "xgboost")
    all_metrics.append(m)
    all_preds["xgboost"] = y_xgb
    if xgb._available:
        joblib.dump(xgb, MODELS_DIR / f"{task_name}_xgb.pkl")
        fi_xgb = xgb.feature_importance(feat_avail)
        fi_xgb.to_csv(METRICS_DIR / f"{task_name}_xgb_feature_importance.csv", index=False)

    # ── LSTM (optional) ──────────────────────────────────────────────────────
    if use_lstm:
        lstm = LSTMModel()
        lstm.fit(X_train, y_train, X_val, y_val)
        y_lstm = lstm.predict(X_test)
        valid_mask = ~np.isnan(y_lstm)
        if valid_mask.sum() > 0:
            m = compute_metrics(y_test[valid_mask], y_lstm[valid_mask], "lstm")
            all_metrics.append(m)
            all_preds["lstm"] = y_lstm

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = METRICS_DIR / f"{task_name}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info("Metrics saved → %s", metrics_path)

    # ── Save forecast results ─────────────────────────────────────────────────
    preds_df = pd.DataFrame(all_preds)
    preds_path = FORECASTS_DIR / f"{task_name}_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    log.info("Predictions saved → %s", preds_path)

    # ── Save feature list ─────────────────────────────────────────────────────
    with open(MODELS_DIR / f"{task_name}_features.json", "w") as f:
        json.dump({"features": feat_avail, "target": target_col, "ts_col": ts_col}, f, indent=2)

    return {"metrics": all_metrics, "predictions": all_preds, "features": feat_avail}


# ── Convenience entry points ──────────────────────────────────────────────────
def train_solar_forecasting(plant: int = 1, use_lstm: bool = False) -> dict:
    from src.data.feature_engineering import engineer_solar_dataset, SOLAR_FEATURE_COLS, get_available_features

    log.info("=== Solar Forecasting – Plant %d ===", plant)
    df = engineer_solar_dataset(plant)
    feats = get_available_features(df, SOLAR_FEATURE_COLS)

    return train_and_evaluate(
        df=df,
        feature_cols=feats,
        target_col="AC_POWER",
        ts_col="DATE_TIME",
        task_name=f"solar_plant{plant}",
        use_lstm=use_lstm,
    )


def train_demand_forecasting(country: str = "DE", use_lstm: bool = False) -> dict:
    from src.data.feature_engineering import engineer_demand_dataset, DEMAND_FEATURE_COLS_BASE, get_available_features

    log.info("=== Demand Forecasting – %s ===", country)
    df = engineer_demand_dataset(country)

    # Find actual load column
    load_cols = [c for c in df.columns if "load_actual" in c.lower() or "load" in c.lower()]
    if not load_cols:
        log.error("No load column found in demand data!")
        return {}
    target = load_cols[0]
    log.info("Target column: %s", target)

    # Build feature list - deduplicate
    extra = [c for c in df.columns if any(k in c.lower() for k in
             ["temperature", "radiation", "price", "lag_", "roll_", "hour", "month", "dow", "solar_generation"])]
    # Exclude target and timestamp from features
    combined = DEMAND_FEATURE_COLS_BASE + extra
    feats = list(dict.fromkeys([c for c in combined if c in df.columns and c != target and c != "timestamp"]))

    return train_and_evaluate(
        df=df,
        feature_cols=feats,
        target_col=target,
        ts_col="timestamp",
        task_name=f"demand_{country}",
        use_lstm=use_lstm,
    )


if __name__ == "__main__":
    train_solar_forecasting(plant=1)
    train_solar_forecasting(plant=2)
    train_demand_forecasting(country="DE")
