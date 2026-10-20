from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    DATA_PATH,
    DECEMBER_INPUTS_PATH,
    SEEDS,
    TARGET_MODE,
    USE_QUOTE_SIGNAL,
    VALIDATION_PATH,
)
from features import FeaturePipeline
from models import LightGBMModel, MLPRegressorModel, inverse_transform_target, transform_target


def generate_submission(
    train_path: Path = DATA_PATH,
    val_path: Path = VALIDATION_PATH,
    december_path: Path = DECEMBER_INPUTS_PATH,
    output_dir: Path = Path("."),
    target_mode: str = TARGET_MODE,
    use_quote_signal: bool = USE_QUOTE_SIGNAL,
    seeds: list[int] = SEEDS,
    model_choice: str = "ensemble",
) -> None:
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    dec_df = pd.read_csv(december_path)

    city_lat_map = train_df.groupby("pickup")["pickup_lat"].first().to_dict()
    city_lon_map = train_df.groupby("pickup")["pickup_lon"].first().to_dict()

    dec_enriched = dec_df.copy()
    dec_enriched["pickup_lat"] = dec_enriched["pickup"].map(city_lat_map)
    dec_enriched["pickup_lon"] = dec_enriched["pickup"].map(city_lon_map)
    dec_enriched["delivery_lat"] = dec_enriched["delivery"].map(city_lat_map)
    dec_enriched["delivery_lon"] = dec_enriched["delivery"].map(city_lon_map)

    if "market_index" not in dec_enriched.columns:
        dec_enriched["market_index"] = float(train_df["market_index"].dropna().iloc[-100:].mean())
    if "quote_signal" not in dec_enriched.columns:
        dec_enriched["quote_signal"] = float(train_df["quote_signal"].dropna().iloc[-100:].mean())

    y_train_raw = train_df["posted_rate"].values
    dist_train = train_df["distance"].values
    dist_val = val_df["distance"].values
    dist_dec = dec_enriched["distance"].values

    y_train_trans = transform_target(y_train_raw, dist_train, target_mode=target_mode)

    val_preds_all = []
    dec_preds_all = []

    print(f"Generating predictions using model_choice='{model_choice}' across seeds {seeds}...")

    for seed in seeds:
        print(f"  Training seed {seed}...")
        if model_choice in ["mlp", "ensemble"]:
            mlp = MLPRegressorModel(
                random_state=seed,
                max_epochs=40,
                target_mode=target_mode,
                use_quote_signal=use_quote_signal,
            )
            split_idx = int(0.9 * len(train_df))
            tr_sub = train_df.iloc[:split_idx]
            val_sub = train_df.iloc[split_idx:]
            mlp.fit(tr_sub, tr_sub["posted_rate"].values, val_sub, val_sub["posted_rate"].values)

            val_preds_mlp = mlp.predict(val_df, distance=dist_val)
            dec_preds_mlp = mlp.predict(dec_enriched, distance=dist_dec)

            if model_choice == "mlp":
                val_preds_all.append(val_preds_mlp)
                dec_preds_all.append(dec_preds_mlp)

        if model_choice in ["lightgbm", "ensemble"]:
            pipeline = FeaturePipeline(
                use_quote_signal=use_quote_signal,
                smoothing=10.0,
                n_splits=5,
                seed=seed,
            )
            X_train_lgb, _, _ = pipeline.fit_transform_train(train_df, y_train_trans)
            X_val_lgb, _ = pipeline.transform(val_df)
            X_dec_lgb, _ = pipeline.transform(dec_enriched)

            model = LightGBMModel(random_state=seed, n_estimators=150)
            model.model.fit(X_train_lgb, y_train_trans)

            val_preds_lgb = inverse_transform_target(model.predict(X_val_lgb), dist_val, target_mode=target_mode)
            dec_preds_lgb = inverse_transform_target(model.predict(X_dec_lgb), dist_dec, target_mode=target_mode)

            if model_choice == "lightgbm":
                val_preds_all.append(val_preds_lgb)
                dec_preds_all.append(dec_preds_lgb)
            elif model_choice == "ensemble":
                val_preds_all.append(0.5 * val_preds_mlp + 0.5 * val_preds_lgb)
                dec_preds_all.append(0.5 * dec_preds_mlp + 0.5 * dec_preds_lgb)

    final_val_pred = np.mean(val_preds_all, axis=0)
    final_dec_pred = np.mean(dec_preds_all, axis=0)

    val_out = pd.DataFrame({
        "load_id": val_df["load_id"],
        "predicted_rate": np.round(final_val_pred, 2),
    })
    val_out_path = output_dir / "validation_predictions.csv"
    val_out.to_csv(val_out_path, index=False)
    print(f"[Saved] {val_out_path} ({len(val_out)} rows)")

    dec_out = dec_df.copy()
    dec_out["predicted_rate"] = np.round(final_dec_pred, 2)
    dec_out_path = output_dir / "december_predictions.csv"
    dec_out.to_csv(dec_out_path, index=False)
    print(f"[Saved] {dec_out_path} ({len(dec_out)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["ensemble", "mlp", "lightgbm"], default="ensemble")
    args = parser.parse_args()
    generate_submission(model_choice=args.model)
