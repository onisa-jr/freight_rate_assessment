from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    EARTH_RADIUS_MILES,
    TARGET_ENCODING_FOLDS,
    TARGET_ENCODING_SMOOTHING,
    USE_QUOTE_SIGNAL,
)
from data import calculate_bearing, haversine_distance


class FeaturePipeline:
    def __init__(
        self,
        use_quote_signal: bool = USE_QUOTE_SIGNAL,
        smoothing: float = TARGET_ENCODING_SMOOTHING,
        n_splits: int = TARGET_ENCODING_FOLDS,
        seed: int = 42,
    ):
        self.use_quote_signal = use_quote_signal
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.seed = seed

        self.medians_: dict[str, float] = {}
        self.global_target_mean_: float = 0.0
        self.lane_target_mapping_: dict[str, float] = {}
        self.categories_: dict[str, list] = {}
        self.ridge_scaler_: StandardScaler | None = None
        self.equipment_ohe_: OneHotEncoder | None = None
        self.ridge_numeric_cols_: list[str] = []
        self.lgb_feature_names_: list[str] = []

    def _impute_numeric(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        df_out = df.copy()
        numeric_cols = ["weight", "market_index", "distance"]
        if self.use_quote_signal and "quote_signal" in df.columns:
            numeric_cols.append("quote_signal")

        if is_train:
            for col in numeric_cols:
                if col in df.columns:
                    median_val = float(df[col].median(skipna=True))
                    self.medians_[col] = median_val

        for col in numeric_cols:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(self.medians_.get(col, 0.0))

        return df_out

    def _extract_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)

        feat["distance"] = df["distance"].values
        feat["haversine_dist"] = haversine_distance(
            df["pickup_lat"].values,
            df["pickup_lon"].values,
            df["delivery_lat"].values,
            df["delivery_lon"].values,
            radius_miles=EARTH_RADIUS_MILES,
        )
        feat["bearing"] = calculate_bearing(
            df["pickup_lat"].values,
            df["pickup_lon"].values,
            df["delivery_lat"].values,
            df["delivery_lon"].values,
        )

        safe_distance = np.maximum(df["distance"].values, 1.0)
        feat["weight"] = df["weight"].values
        feat["market_index"] = df["market_index"].values
        feat["weight_per_distance"] = df["weight"].values / safe_distance
        feat["dist_x_market"] = df["distance"].values * df["market_index"].values

        if self.use_quote_signal and "quote_signal" in df.columns:
            feat["quote_signal"] = df["quote_signal"].values
            feat["quote_x_dist"] = df["quote_signal"].values * df["distance"].values

        dates = pd.to_datetime(df["date"])
        feat["dayofweek"] = dates.dt.dayofweek.values
        feat["month"] = dates.dt.month.values
        feat["weekofyear"] = dates.dt.isocalendar().week.astype(int).values
        feat["dayofmonth"] = dates.dt.day.values

        feat["equipment"] = df["equipment"].astype(str).values
        feat["pickup"] = df["pickup"].astype(str).values
        feat["delivery"] = df["delivery"].astype(str).values
        feat["lane"] = (df["pickup"].astype(str) + "->" + df["delivery"].astype(str)).values

        return feat

    def fit_transform_train(
        self,
        train_df: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
        y_series = pd.Series(y_train).reset_index(drop=True)
        train_imputed = self._impute_numeric(train_df, is_train=True)
        base_feat = self._extract_base_features(train_imputed)

        self.global_target_mean_ = float(y_series.mean())
        lane_df = pd.DataFrame({"lane": base_feat["lane"], "target": y_series.values})
        lane_stats = lane_df.groupby("lane")["target"].agg(["count", "sum"])
        smoothed_target = (lane_stats["sum"] + self.smoothing * self.global_target_mean_) / (
            lane_stats["count"] + self.smoothing
        )
        self.lane_target_mapping_ = smoothed_target.to_dict()

        oof_target_enc = np.full(len(train_df), self.global_target_mean_, dtype=float)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        for train_idx, val_idx in kf.split(train_df):
            fold_y = y_series.iloc[train_idx]
            fold_lanes = base_feat["lane"].iloc[train_idx]
            fold_global_mean = float(fold_y.mean())

            fold_df = pd.DataFrame({"lane": fold_lanes, "target": fold_y.values})
            fold_stats = fold_df.groupby("lane")["target"].agg(["count", "sum"])
            fold_enc = (fold_stats["sum"] + self.smoothing * fold_global_mean) / (
                fold_stats["count"] + self.smoothing
            )
            val_lanes = base_feat["lane"].iloc[val_idx]
            oof_target_enc[val_idx] = val_lanes.map(fold_enc).fillna(fold_global_mean).values

        base_feat["lane_target_enc"] = oof_target_enc

        cat_cols = ["equipment", "pickup", "delivery", "lane"]
        for col in cat_cols:
            unique_levels = sorted(base_feat[col].unique().tolist())
            self.categories_[col] = unique_levels
            base_feat[col] = pd.Categorical(base_feat[col], categories=unique_levels)

        self.ridge_numeric_cols_ = ["distance", "weight", "market_index"]
        if self.use_quote_signal:
            self.ridge_numeric_cols_.append("quote_signal")

        self.equipment_ohe_ = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        equipment_arr = base_feat[["equipment"]].astype(str)
        equip_dummies = self.equipment_ohe_.fit_transform(equipment_arr)

        self.ridge_scaler_ = StandardScaler()
        num_arr = base_feat[self.ridge_numeric_cols_].values
        num_scaled = self.ridge_scaler_.fit_transform(num_arr)
        X_train_ridge = np.hstack([num_scaled, equip_dummies])

        self.lgb_feature_names_ = list(base_feat.columns)
        return base_feat, X_train_ridge, self.lgb_feature_names_

    def transform(self, val_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        val_imputed = self._impute_numeric(val_df, is_train=False)
        base_feat = self._extract_base_features(val_imputed)

        base_feat["lane_target_enc"] = (
            base_feat["lane"].map(self.lane_target_mapping_).fillna(self.global_target_mean_).values
        )

        for col, levels in self.categories_.items():
            s = base_feat[col].astype(str)
            s_clean = s.where(s.isin(levels), other=np.nan)
            base_feat[col] = s_clean.astype(pd.CategoricalDtype(categories=levels))

        X_val_lgb = base_feat[self.lgb_feature_names_].copy()

        equipment_arr = base_feat[["equipment"]].astype(str)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            equip_dummies = self.equipment_ohe_.transform(equipment_arr)
        num_arr = base_feat[self.ridge_numeric_cols_].values
        num_scaled = self.ridge_scaler_.transform(num_arr)
        X_val_ridge = np.hstack([num_scaled, equip_dummies])

        return X_val_lgb, X_val_ridge


def build_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    use_quote_signal: bool = USE_QUOTE_SIGNAL,
    smoothing: float = TARGET_ENCODING_SMOOTHING,
    n_splits: int = TARGET_ENCODING_FOLDS,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], FeaturePipeline]:
    pipeline = FeaturePipeline(
        use_quote_signal=use_quote_signal,
        smoothing=smoothing,
        n_splits=n_splits,
        seed=seed,
    )
    X_train_lgb, X_train_ridge, feature_names = pipeline.fit_transform_train(train_df, y_train)
    X_val_lgb, X_val_ridge = pipeline.transform(val_df)

    return X_train_lgb, X_val_lgb, X_train_ridge, X_val_ridge, feature_names, pipeline
