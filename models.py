from __future__ import annotations

import copy
import math
import random
import warnings
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    EARLY_STOPPING_ROUNDS,
    EARTH_RADIUS_MILES,
    LGBM_PARAMS,
    MLP_BATCH_SIZE,
    MLP_LR,
    MLP_LR_FACTOR,
    MLP_LR_PATIENCE,
    MLP_MAX_EPOCHS,
    MLP_NORM,
    MLP_PATIENCE,
    MLP_WEIGHT_DECAY,
    USE_QUOTE_SIGNAL,
)
from data import calculate_bearing, haversine_distance


def seed_everything(seed: int) -> torch.Generator:
    """Seed python random, numpy, and torch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def transform_target(
    y: pd.Series | np.ndarray,
    distance: pd.Series | np.ndarray,
    target_mode: str = "raw",
) -> np.ndarray:
    """Transform target based on configured target mode."""
    y_arr = np.asarray(y, dtype=float)
    dist_arr = np.maximum(np.asarray(distance, dtype=float), 1.0)

    if target_mode == "raw":
        return y_arr
    elif target_mode == "rate_per_mile":
        return y_arr / dist_arr
    elif target_mode == "log":
        return np.log(np.maximum(y_arr, 1e-4))
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}. Expected 'raw', 'rate_per_mile', or 'log'.")


def inverse_transform_target(
    pred: np.ndarray,
    distance: pd.Series | np.ndarray,
    target_mode: str = "raw",
) -> np.ndarray:
    """Invert target transformation back to dollar units for evaluation."""
    pred_arr = np.asarray(pred, dtype=float)
    dist_arr = np.maximum(np.asarray(distance, dtype=float), 1.0)

    if target_mode == "raw":
        return np.maximum(pred_arr, 0.0)
    elif target_mode == "rate_per_mile":
        return np.maximum(pred_arr, 0.0) * dist_arr
    elif target_mode == "log":
        return np.exp(pred_arr)
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}. Expected 'raw', 'rate_per_mile', or 'log'.")


class RidgeBaseline:
    """Baseline linear model on scaled numeric features and equipment dummies."""

    def __init__(self, alpha: float = 1.0, random_state: int = 42):
        self.alpha = alpha
        self.random_state = random_state
        self.model = Ridge(alpha=alpha, random_state=random_state)
        self.best_iteration_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> RidgeBaseline:
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class LightGBMModel:
    """Main LightGBM model with fixed sensible defaults and early stopping."""

    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = random_state
        params = dict(LGBM_PARAMS)
        params["random_state"] = random_state
        params.update(kwargs)
        self.model = LGBMRegressor(**params)
        self.feature_names_: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    ) -> LightGBMModel:
        self.feature_names_ = list(X_train.columns)
        callbacks = [early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    @property
    def best_iteration_(self) -> int | None:
        return getattr(self.model, "best_iteration_", None)

    def get_feature_importances(self, importance_type: str = "gain") -> pd.Series:
        importances = self.model.booster_.feature_importance(importance_type=importance_type)
        return pd.Series(importances, index=self.feature_names_, name="gain_importance")


class MLPNetwork(nn.Module):
    """
    Fixed PyTorch architecture for tabular regression:
    Concatenate numeric + embeddings -> Linear(128) -> ReLU -> Dropout(0.1)
    -> Linear(64) -> ReLU -> Dropout(0.1) -> Linear(1)
    Optional BatchNorm or LayerNorm after first layer.
    """

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: list[int],
        cat_embedding_dims: list[int],
        norm_type: str | None = None,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=cardinality, embedding_dim=emb_dim)
            for cardinality, emb_dim in zip(cat_cardinalities, cat_embedding_dims)
        ])
        total_emb_dim = sum(cat_embedding_dims)
        in_features = n_numeric + total_emb_dim

        self.fc1 = nn.Linear(in_features, 128)
        if norm_type == "batchnorm":
            self.norm1 = nn.BatchNorm1d(128)
        elif norm_type == "layernorm":
            self.norm1 = nn.LayerNorm(128)
        else:
            self.norm1 = nn.Identity()

        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + emb_outs, dim=1)
        x = self.drop1(self.relu(self.norm1(self.fc1(x))))
        x = self.drop2(self.relu(self.fc2(x)))
        out = self.fc3(x)
        return out.squeeze(-1)


class MLPRegressorModel:
    """
    Neural Network (MLP) for Freight Rate Estimation.
    - Standardizes numeric features with StandardScaler fit on train only (log1p applied to weight and distance).
    - Learned embeddings for categoricals (equipment, pickup, delivery, lane, dayofweek, month).
    - Unseen categories in validation map to reserved index.
    - Trains with AdamW, Huber (SmoothL1) loss, ReduceLROnPlateau, and early stopping.
    - All predictions unstandardized and converted back to dollar units.
    """

    def __init__(
        self,
        random_state: int = 42,
        max_epochs: int = MLP_MAX_EPOCHS,
        patience: int = MLP_PATIENCE,
        batch_size: int = MLP_BATCH_SIZE,
        lr: float = MLP_LR,
        weight_decay: float = MLP_WEIGHT_DECAY,
        norm_type: str | None = MLP_NORM,
        target_mode: str = "raw",
        use_quote_signal: bool = USE_QUOTE_SIGNAL,
    ):
        self.random_state = random_state
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.norm_type = norm_type
        self.target_mode = target_mode
        self.use_quote_signal = use_quote_signal

        # Learned preprocessing artifacts (train only)
        self.scaler_: StandardScaler | None = None
        self.medians_: dict[str, float] = {}
        self.cat_mappings_: dict[str, dict[Any, int]] = {}
        self.cat_unseen_indices_: dict[str, int] = {}
        self.cat_embedding_dims_: list[int] = []
        self.cat_cardinalities_: list[int] = []
        self.target_mean_: float = 0.0
        self.target_std_: float = 1.0

        # Trained model and artifacts
        self.model: MLPNetwork | None = None
        self.best_iteration_: int | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def _extract_features(
        self,
        df: pd.DataFrame,
        is_train: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract and format standardized numeric and indexed categorical matrices."""
        df_work = df.copy()

        # 1. Impute numerics using train medians
        numeric_impute_cols = ["weight", "market_index", "distance"]
        if self.use_quote_signal and "quote_signal" in df_work.columns:
            numeric_impute_cols.append("quote_signal")

        if is_train:
            for col in numeric_impute_cols:
                if col in df_work.columns:
                    self.medians_[col] = float(df_work[col].median(skipna=True))

        for col in numeric_impute_cols:
            if col in df_work.columns:
                df_work[col] = df_work[col].fillna(self.medians_.get(col, 0.0))

        # 2. Categoricals: equipment, pickup, delivery, lane, dayofweek, month
        dates = pd.to_datetime(df_work["date"])
        lane_series = df_work["pickup"].astype(str) + "->" + df_work["delivery"].astype(str)

        cat_dict = {
            "equipment": df_work["equipment"].astype(str),
            "pickup": df_work["pickup"].astype(str),
            "delivery": df_work["delivery"].astype(str),
            "lane": lane_series,
            "dayofweek": dates.dt.dayofweek.astype(str),
            "month": dates.dt.month.astype(str),
        }

        cat_cols = ["equipment", "pickup", "delivery", "lane", "dayofweek", "month"]
        cat_arrays = []

        if is_train:
            self.cat_mappings_ = {}
            self.cat_unseen_indices_ = {}
            self.cat_cardinalities_ = []
            self.cat_embedding_dims_ = []

            for col in cat_cols:
                unique_vals = sorted(cat_dict[col].unique().tolist())
                n_cats = len(unique_vals)
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                unseen_idx = n_cats  # Reserved index for unseen categories
                # Embedding dimension = min(16, ceil(n_categories / 2))
                emb_dim = min(16, max(2, int(math.ceil(n_cats / 2))))

                self.cat_mappings_[col] = mapping
                self.cat_unseen_indices_[col] = unseen_idx
                self.cat_cardinalities_.append(n_cats + 1)
                self.cat_embedding_dims_.append(emb_dim)

                mapped = cat_dict[col].map(mapping).values.astype(np.int64)
                cat_arrays.append(mapped)
        else:
            for col in cat_cols:
                mapping = self.cat_mappings_[col]
                unseen_idx = self.cat_unseen_indices_[col]
                mapped = cat_dict[col].map(mapping).fillna(unseen_idx).values.astype(np.int64)
                cat_arrays.append(mapped)

        x_cat = np.column_stack(cat_arrays)

        # 3. Numeric features (excluding target-encoded lane):
        # Apply log1p to weight and distance before scaling
        dist = df_work["distance"].values.astype(float)
        weight = df_work["weight"].values.astype(float)
        dist_log1p = np.log1p(np.maximum(dist, 0.0))
        weight_log1p = np.log1p(np.maximum(weight, 0.0))

        hav_dist = haversine_distance(
            df_work["pickup_lat"].values,
            df_work["pickup_lon"].values,
            df_work["delivery_lat"].values,
            df_work["delivery_lon"].values,
            radius_miles=EARTH_RADIUS_MILES,
        )
        bearing = calculate_bearing(
            df_work["pickup_lat"].values,
            df_work["pickup_lon"].values,
            df_work["delivery_lat"].values,
            df_work["delivery_lon"].values,
        )

        safe_dist = np.maximum(dist, 1.0)
        weight_per_dist = weight / safe_dist
        market_idx = df_work["market_index"].values.astype(float)
        dist_x_market = dist * market_idx
        weekofyear = dates.dt.isocalendar().week.astype(float).values
        dayofmonth = dates.dt.day.astype(float).values

        num_list = [
            dist_log1p,
            hav_dist,
            bearing,
            weight_log1p,
            market_idx,
            weight_per_dist,
            dist_x_market,
            weekofyear,
            dayofmonth,
        ]

        if self.use_quote_signal and "quote_signal" in df_work.columns:
            qs = df_work["quote_signal"].values.astype(float)
            num_list.append(qs)
            num_list.append(qs * dist)

        x_num_raw = np.column_stack(num_list)

        if is_train:
            self.scaler_ = StandardScaler()
            x_num_scaled = self.scaler_.fit_transform(x_num_raw)
        else:
            x_num_scaled = self.scaler_.transform(x_num_raw)

        return x_num_scaled.astype(np.float32), x_cat.astype(np.int64)

    def fit(
        self,
        train_df: pd.DataFrame,
        y_train_raw: pd.Series | np.ndarray,
        val_df: pd.DataFrame,
        y_val_raw: pd.Series | np.ndarray,
    ) -> MLPRegressorModel:
        """
        Fit MLP on train_df and validate on val_df:
        - Seed random, numpy, and PyTorch.
        - Target standardized using train mean and std.
        - AdamW optimizer, Huber (SmoothL1) loss, ReduceLROnPlateau.
        - Early stopping with patience restoring best weights.
        """
        seed_generator = seed_everything(self.random_state)

        # 1. Transform target to target_mode and compute train standardization stats
        dist_train = train_df["distance"].values
        dist_val = val_df["distance"].values

        y_tr_trans = transform_target(y_train_raw, dist_train, target_mode=self.target_mode)
        y_va_trans = transform_target(y_val_raw, dist_val, target_mode=self.target_mode)

        self.target_mean_ = float(np.mean(y_tr_trans))
        self.target_std_ = float(np.std(y_tr_trans))
        if self.target_std_ < 1e-6:
            self.target_std_ = 1.0

        y_tr_std = (y_tr_trans - self.target_mean_) / self.target_std_
        y_va_std = (y_va_trans - self.target_mean_) / self.target_std_

        # 2. Extract features
        x_tr_num, x_tr_cat = self._extract_features(train_df, is_train=True)
        x_va_num, x_va_cat = self._extract_features(val_df, is_train=False)

        # 3. PyTorch Datasets & DataLoaders
        train_dataset = TensorDataset(
            torch.from_numpy(x_tr_num),
            torch.from_numpy(x_tr_cat),
            torch.from_numpy(y_tr_std.astype(np.float32)),
        )
        val_num_tensor = torch.from_numpy(x_va_num)
        val_cat_tensor = torch.from_numpy(x_va_cat)
        val_y_tensor = torch.from_numpy(y_va_std.astype(np.float32))

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=seed_generator,
            drop_last=False,
        )

        # 4. Initialize Network, Loss, Optimizer, Scheduler
        self.model = MLPNetwork(
            n_numeric=x_tr_num.shape[1],
            cat_cardinalities=self.cat_cardinalities_,
            cat_embedding_dims=self.cat_embedding_dims_,
            norm_type=self.norm_type,
        )

        criterion = nn.SmoothL1Loss(beta=1.0)  # Huber loss
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=MLP_LR_FACTOR, patience=MLP_LR_PATIENCE
        )

        best_val_loss = float("inf")
        best_epoch = 0
        best_weights = copy.deepcopy(self.model.state_dict())
        patience_counter = 0

        self.history_ = {"train_loss": [], "val_loss": []}

        # 5. Training loop
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            batch_losses = []
            for b_num, b_cat, b_y in train_loader:
                optimizer.zero_grad()
                pred = self.model(b_num, b_cat)
                loss = criterion(pred, b_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            train_loss = float(np.mean(batch_losses))
            self.history_["train_loss"].append(train_loss)

            # Validation loss
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(val_num_tensor, val_cat_tensor)
                val_loss = float(criterion(val_pred, val_y_tensor).item())
            self.history_["val_loss"].append(val_loss)

            scheduler.step(val_loss)

            # Check early stopping
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_epoch = epoch
                best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Restore best weights
        self.model.load_state_dict(best_weights)
        self.best_iteration_ = best_epoch
        return self

    def predict(self, df: pd.DataFrame, distance: pd.Series | np.ndarray | None = None) -> np.ndarray:
        """Predict posted_rate in dollar units."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        if distance is None:
            distance = df["distance"].values

        x_num, x_cat = self._extract_features(df, is_train=False)

        self.model.eval()
        with torch.no_grad():
            pred_std = self.model(torch.from_numpy(x_num), torch.from_numpy(x_cat)).numpy()

        # Unstandardize prediction
        pred_trans = pred_std * self.target_std_ + self.target_mean_

        # Invert target transform to return dollars
        pred_dollars = inverse_transform_target(pred_trans, distance, target_mode=self.target_mode)
        return pred_dollars
