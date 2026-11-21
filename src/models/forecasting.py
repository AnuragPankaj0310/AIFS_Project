"""
Forecasting Models Module
==========================
Implements solar generation and electricity demand forecasting.

Models:
  1. Persistence Baseline
  2. Linear Regression (Ridge)
  3. Random Forest Regressor
  4. XGBoost Regressor
  5. SVR with Gaussian kernel + Similar-Day Averaging
     (Houchati, Beitelmal & Khraisheh, ASME 2022 method)
  6. LSTM (optional, enabled via use_lstm=True)

SVR Method (paper replication):
  - K-means classifies days into sky-condition clusters (default 4).
  - For each target day, N=3 nearest historical days (Euclidean distance
    on weather features) are found within the same sky cluster.
  - The mean of those N days' hourly target values becomes an extra feature
    added to the SVR input alongside standard weather/time features.
  - SVR uses an RBF (Gaussian) kernel; C, epsilon, gamma are grid-searched.

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
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
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


# ── SVR with Similar-Day Averaging (Houchati et al., ASME 2022) ───────────────
def _build_daily_weather_matrix(
    df: pd.DataFrame,
    ts_col: str,
    weather_cols: list[str],
) -> pd.DataFrame:
    """
    Aggregate per-day weather statistics for similar-day matching.

    For each calendar date, compute the mean of each weather column.
    Returns a DataFrame indexed by date with one row per day.
    """
    tmp = df.copy()
    tmp["_date"] = pd.to_datetime(tmp[ts_col]).dt.date
    avail = [c for c in weather_cols if c in tmp.columns]
    daily = tmp.groupby("_date")[avail].mean()
    return daily  # index = date, columns = weather means


def add_similar_day_feature(
    df: pd.DataFrame,
    ts_col: str,
    target_col: str,
    weather_cols: list[str],
    n_similar: int = 3,
    n_sky_clusters: int = 4,
    train_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Paper method (Houchati et al.):
      1. K-means cluster days by weather → sky-condition classes.
      2. For each day, find N nearest historical days in the same sky class
         (Euclidean distance on daily-mean weather).
      3. Average the target-variable hourly profile of those N days.
      4. Add this per-hour average as a new feature column.

    Parameters
    ----------
    df          : feature-engineered DataFrame (must contain ts_col, target_col
                  and at least one weather column).
    ts_col      : timestamp column name.
    target_col  : what we are forecasting (e.g. 'AC_POWER').
    weather_cols: columns used for day similarity (e.g. temperature, irradiation).
    n_similar   : N nearest days to average (paper uses N=3).
    n_sky_clusters: K for K-means (paper uses 4 classes).
    train_mask  : boolean mask of training rows; similar-day lookup is restricted
                  to training rows only to prevent data leakage.

    Returns df with a new column  `{target_col}_similar_day_avg`.
    """
    new_col = f"{target_col}_similar_day_avg"
    df = df.copy()
    df[new_col] = np.nan

    df["_date"] = pd.to_datetime(df[ts_col]).dt.date

    avail_weather = [c for c in weather_cols if c in df.columns]
    if not avail_weather:
        log.warning("No weather cols found for similar-day feature; skipping.")
        df = df.drop(columns=["_date"])
        return df

    # Build daily weather summary (train rows only → no leakage)
    if train_mask is not None:
        ref_df = df.loc[train_mask]
    else:
        ref_df = df

    daily_weather = ref_df.groupby("_date")[avail_weather].mean()
    all_dates = daily_weather.index.tolist()

    if len(all_dates) < n_sky_clusters + 1:
        log.warning("Too few days for K-means similar-day feature; skipping.")
        df = df.drop(columns=["_date"])
        return df

    # K-means sky classification
    scaler_km = StandardScaler()
    W = scaler_km.fit_transform(daily_weather.values)
    km = KMeans(n_clusters=n_sky_clusters, random_state=42, n_init=10)
    day_cluster = pd.Series(km.fit_predict(W), index=all_dates, name="sky_cluster")

    # Build hourly lookup for reference days: date → array of hourly target vals
    ref_hourly = {}
    for date, grp in ref_df.groupby("_date"):
        ref_hourly[date] = grp[target_col].values  # may vary in length

    # For every row in df, compute the similar-day average
    # (group by date for speed)
    target_dates = df["_date"].unique()

    for tdate in target_dates:
        # Cluster of this day (may not be in training set)
        if tdate in day_cluster.index:
            tcluster = day_cluster[tdate]
            # Same-cluster reference days (excluding current day to avoid leakage)
            same_cluster = [d for d in all_dates if day_cluster.get(d) == tcluster and d != tdate]
        else:
            # Day not in training set → use all training dates
            same_cluster = all_dates

        if not same_cluster:
            same_cluster = all_dates  # fallback

        # Daily weather of this target day
        if tdate in daily_weather.index:
            td_w = daily_weather.loc[tdate].values
        else:
            # Use df row means for this date
            td_w = df.loc[df["_date"] == tdate, avail_weather].mean().values

        # Euclidean distances to same-cluster days
        ref_w = daily_weather.loc[[d for d in same_cluster if d in daily_weather.index]]
        if len(ref_w) == 0:
            continue
        dists = np.linalg.norm(
            scaler_km.transform(ref_w.values) - scaler_km.transform(td_w.reshape(1, -1)),
            axis=1,
        )
        nearest_n = ref_w.index[np.argsort(dists)[:n_similar]]

        # Collect hourly profiles of the N nearest days and align by step index
        profiles = []
        for nd in nearest_n:
            if nd in ref_hourly:
                profiles.append(ref_hourly[nd])

        if not profiles:
            continue

        # Align profiles: pad/truncate to the length of the target day
        tday_rows = df[df["_date"] == tdate]
        tlen = len(tday_rows)
        aligned = []
        for p in profiles:
            if len(p) >= tlen:
                aligned.append(p[:tlen])
            else:
                aligned.append(np.pad(p, (0, tlen - len(p)), mode="edge"))
        avg_profile = np.mean(aligned, axis=0)

        df.loc[tday_rows.index, new_col] = avg_profile

    df = df.drop(columns=["_date"])
    log.info("Similar-day feature '%s' added (%d NaN rows).", new_col, df[new_col].isna().sum())
    return df


class SVRModel:
    """
    SVR with Gaussian (RBF) kernel — replicates Houchati et al. (ASME 2022).

    The model:
      - Uses StandardScaler on inputs (SVR is distance-based).
      - Grid-searches C, epsilon, gamma over a small grid on the validation set.
      - Clips predictions to ≥ 0.

    Usage:
      model = SVRModel()
      model.fit(X_train, y_train, X_val, y_val)
      y_pred = model.predict(X_test)
    """

    # Grid from paper guidance: C controls accuracy-vs-complexity trade-off;
    # gamma (σ) is the Gaussian kernel width; epsilon is the ε-insensitive tube.
    # Kept small for fast training; widen for a full hyperparameter sweep.
    PARAM_GRID = {
        "C":       [1.0, 10.0],
        "epsilon": [0.1, 0.5],
        "gamma":   ["scale", 0.1],
    }

    def __init__(self):
        self.scaler = StandardScaler()
        self.model: SVR | None = None
        self.name = "svr_rbf"
        self.best_params_: dict = {}

    # SVR training is O(n²) — subsample large datasets to keep it fast.
    MAX_TRAIN_ROWS = 5000

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ):
        # Subsample if dataset is too large (SVR is O(n²) in training size)
        if len(X_train) > self.MAX_TRAIN_ROWS:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_train), self.MAX_TRAIN_ROWS, replace=False)
            idx.sort()  # preserve rough chronological order
            X_train = X_train[idx]
            y_train = y_train[idx]
            log.info("[SVR] Subsampled training set to %d rows (from %d) for speed.",
                     self.MAX_TRAIN_ROWS, len(X_train) + (len(X_train) - len(idx)))

        Xs_train = self.scaler.fit_transform(X_train)

        # Combine train+val for grid-search cross-validation if val available
        if X_val is not None and y_val is not None:
            # Also subsample val for grid-search
            if len(X_val) > 2000:
                rng2 = np.random.default_rng(99)
                vidx = rng2.choice(len(X_val), 2000, replace=False)
                X_val = X_val[vidx]; y_val = y_val[vidx]
            Xs_val = self.scaler.transform(X_val)
            Xs_all = np.vstack([Xs_train, Xs_val])
            y_all = np.concatenate([y_train, y_val])
            cv = 3
        else:
            Xs_all = Xs_train
            y_all = y_train
            cv = 3

        log.info("[SVR] Grid-searching hyperparameters (this may take a moment)...")
        gs = GridSearchCV(
            SVR(kernel="rbf"),
            self.PARAM_GRID,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            refit=True,
        )
        gs.fit(Xs_all, y_all)
        self.model = gs.best_estimator_
        self.best_params_ = gs.best_params_
        log.info("[SVR] Best params: %s", self.best_params_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs).clip(min=0)


# ── LSTM ──────────────────────────────────────────────────────────────────────
class LSTMModel:
    """Sequence-to-one LSTM for time-series forecasting."""

    def __init__(self, lookback: int = 96, units: int = 64, epochs: int = 10, batch_size: int = 64):
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
    use_svr: bool = True,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    similar_day_weather_cols: list[str] | None = None,
) -> dict:
    """
    Full train/val/test pipeline for all models.

    Parameters
    ----------
    use_svr : If True, trains SVR with similar-day averaging (paper method).
    similar_day_weather_cols : Weather columns used for similar-day Euclidean
        matching. Defaults to ['IRRADIATION','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']
        for solar; ['temperature','radiation_direct_horizontal'] for demand.

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

    # ── Similar-Day Feature (Houchati et al.) ─────────────────────────────────
    # Build before the train/test split so we can pass train_mask for leakage prevention
    sim_day_col = f"{target_col}_similar_day_avg"
    if use_svr:
        if similar_day_weather_cols is None:
            # Auto-detect weather columns
            similar_day_weather_cols = [
                c for c in ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
                             "temperature", "radiation_direct_horizontal",
                             "radiation_diffuse_horizontal"]
                if c in df_clean.columns
            ]
        n_rows = len(df_clean)
        train_end = int(n_rows * train_frac)
        train_mask = np.zeros(n_rows, dtype=bool)
        train_mask[:train_end] = True

        df_clean = add_similar_day_feature(
            df_clean,
            ts_col=ts_col,
            target_col=target_col,
            weather_cols=similar_day_weather_cols,
            n_similar=3,
            n_sky_clusters=4,
            train_mask=train_mask,
        )
        # Include the new feature in the SVR feature set
        svr_feature_cols = feature_cols_dedup + ([sim_day_col] if sim_day_col in df_clean.columns else [])
    else:
        svr_feature_cols = feature_cols_dedup

    feat_avail = [c for c in feature_cols_dedup if c in df_clean.columns and c != target_col and c != ts_col]
    feat_avail = list(dict.fromkeys(feat_avail))  # final dedup
    log.info("[%s] Features used: %d/%d", task_name, len(feat_avail), len(feature_cols_dedup))

    svr_feat_avail = [c for c in svr_feature_cols if c in df_clean.columns and c != target_col and c != ts_col]
    svr_feat_avail = list(dict.fromkeys(svr_feat_avail))

    train, val, test = chronological_split(df_clean, ts_col, train_frac, val_frac)

    # Fill NaN in features with median from train set
    all_feat_cols = list(dict.fromkeys(feat_avail + svr_feat_avail))
    medians = train[all_feat_cols].median()
    for split in [train, val, test]:
        split[all_feat_cols] = split[all_feat_cols].fillna(medians)

    X_train = train[feat_avail].values
    y_train = train[target_col].values
    X_val = val[feat_avail].values
    y_val = val[target_col].values
    X_test = test[feat_avail].values
    y_test = test[target_col].values

    # SVR uses augmented features (includes similar-day avg)
    X_train_svr = train[svr_feat_avail].values
    X_val_svr   = val[svr_feat_avail].values
    X_test_svr  = test[svr_feat_avail].values

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

    # ── SVR + Similar-Day (Houchati et al., ASME 2022) ───────────────────────
    if use_svr:
        log.info("[%s] Training SVR with similar-day feature (paper method)...", task_name)
        svr_model = SVRModel()
        svr_model.fit(X_train_svr, y_train, X_val_svr, y_val)
        y_svr = svr_model.predict(X_test_svr)
        m = compute_metrics(y_test, y_svr, "svr_rbf")
        m["svr_best_params"] = str(svr_model.best_params_)
        all_metrics.append(m)
        all_preds["svr_rbf"] = y_svr
        joblib.dump(svr_model, MODELS_DIR / f"{task_name}_svr.pkl")
        # Save SVR feature list (includes similar-day column)
        with open(MODELS_DIR / f"{task_name}_svr_features.json", "w") as f:
            json.dump({"features": svr_feat_avail, "best_params": svr_model.best_params_}, f, indent=2)
        log.info("[SVR] Best params saved.")

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
