import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import EARTH_RADIUS_MILES, USE_QUOTE_SIGNAL


def haversine_distance(
    lat1: np.ndarray | pd.Series | float,
    lon1: np.ndarray | pd.Series | float,
    lat2: np.ndarray | pd.Series | float,
    lon2: np.ndarray | pd.Series | float,
    radius_miles: float = EARTH_RADIUS_MILES,
) -> np.ndarray:
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return radius_miles * c


def calculate_bearing(
    lat1: np.ndarray | pd.Series | float,
    lon1: np.ndarray | pd.Series | float,
    lat2: np.ndarray | pd.Series | float,
    lon2: np.ndarray | pd.Series | float,
) -> np.ndarray:
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlon = lon2_rad - lon1_rad

    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    bearing_rad = np.arctan2(x, y)
    bearing_deg = (np.degrees(bearing_rad) + 360.0) % 360.0
    return bearing_deg


def load_data(
    path: str,
    parse_dates: bool = True,
    sample_rows: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if parse_dates and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    if sample_rows is not None and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=random_state).reset_index(drop=True)

    return df


def run_data_checks(df: pd.DataFrame, use_quote_signal: bool = USE_QUOTE_SIGNAL) -> dict:
    report = {}

    n_rows = len(df)
    min_date = df["date"].min() if "date" in df.columns else None
    max_date = df["date"].max() if "date" in df.columns else None
    null_counts = df.isna().sum().to_dict()
    report["n_rows"] = n_rows
    report["date_range"] = (str(min_date), str(max_date))
    report["null_counts"] = null_counts

    df_calc = df.copy()
    df_calc["haversine_dist"] = haversine_distance(
        df_calc["pickup_lat"], df_calc["pickup_lon"], df_calc["delivery_lat"], df_calc["delivery_lon"]
    )
    dist_diff = (df_calc["haversine_dist"] - df_calc["distance"]).abs()
    rel_diff = dist_diff / np.maximum(df_calc["distance"], 1e-6)

    corr_dist = float(df_calc["haversine_dist"].corr(df_calc["distance"]))
    median_abs_diff = float(dist_diff.median())
    frac_diff_gt_20 = float((rel_diff > 0.20).mean())

    suspicious_mask = rel_diff > 0.20
    n_suspicious = int(suspicious_mask.sum())
    suspicious_sample = (
        df_calc[suspicious_mask][["pickup", "delivery", "distance", "haversine_dist"]]
        .head(5)
        .to_dict(orient="records")
    )

    report["haversine_check"] = {
        "correlation": corr_dist,
        "median_abs_diff_miles": median_abs_diff,
        "frac_diff_gt_20_pct": frac_diff_gt_20 * 100.0,
        "n_suspicious_rows": n_suspicious,
        "suspicious_sample": suspicious_sample,
    }

    pickup_unique_coords = df.groupby("pickup")[["pickup_lat", "pickup_lon"]].nunique().max().to_dict()
    delivery_unique_coords = df.groupby("delivery")[["delivery_lat", "delivery_lon"]].nunique().max().to_dict()

    p_first = df.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    d_first = df.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
    common_cities = p_first.index.intersection(d_first.index)
    city_mismatches = 0
    for city in common_cities:
        plat, plon = p_first.loc[city]
        dlat, dlon = d_first.loc[city]
        if abs(plat - dlat) > 1e-4 or abs(plon - dlon) > 1e-4:
            city_mismatches += 1

    report["city_coord_check"] = {
        "max_unique_coords_per_pickup": pickup_unique_coords,
        "max_unique_coords_per_delivery": delivery_unique_coords,
        "pickup_delivery_mismatches": city_mismatches,
    }

    pr = df["posted_rate"]
    q1 = pr.quantile(0.25)
    q3 = pr.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((pr < lower_bound) | (pr > upper_bound)).sum())

    report["posted_rate_distribution"] = {
        "mean": float(pr.mean()),
        "std": float(pr.std()),
        "min": float(pr.min()),
        "median": float(pr.median()),
        "max": float(pr.max()),
        "skewness": float(pr.skew()),
        "zero_count": int((pr == 0).sum()),
        "negative_count": int((pr < 0).sum()),
        "iqr_outlier_count": outlier_count,
        "iqr_outlier_pct": float(outlier_count / n_rows * 100.0),
    }

    rpm = df["posted_rate"] / np.maximum(df["distance"], 1e-6)
    corr_qs_rate = float(df["quote_signal"].corr(df["posted_rate"])) if "quote_signal" in df else np.nan
    corr_qs_rpm = float(df["quote_signal"].corr(rpm)) if "quote_signal" in df else np.nan
    corr_mi_rate = float(df["market_index"].corr(df["posted_rate"])) if "market_index" in df else np.nan
    corr_mi_rpm = float(df["market_index"].corr(rpm)) if "market_index" in df else np.nan
    corr_dist_rate = float(df["distance"].corr(df["posted_rate"]))
    corr_qs_dist_rate = (
        float((df["quote_signal"] * df["distance"]).corr(df["posted_rate"])) if "quote_signal" in df else np.nan
    )

    report["leakage_check"] = {
        "USE_QUOTE_SIGNAL_FLAG": use_quote_signal,
        "corr_quote_signal_vs_posted_rate": corr_qs_rate,
        "corr_quote_signal_vs_rate_per_mile": corr_qs_rpm,
        "corr_market_index_vs_posted_rate": corr_mi_rate,
        "corr_market_index_vs_rate_per_mile": corr_mi_rpm,
        "corr_distance_vs_posted_rate": corr_dist_rate,
        "corr_quote_x_dist_vs_posted_rate": corr_qs_dist_rate,
    }

    print("=" * 70)
    print("DATA CHECKS REPORT (Step 1)")
    print("=" * 70)
    print(f"Row count: {n_rows:,}")
    print(f"Date range: {min_date} to {max_date}")
    print(f"Null values: { {k: v for k, v in null_counts.items() if v > 0} }")
    print("\n[Haversine vs Reported Distance]")
    print(f"  Correlation: {corr_dist:.4f}")
    print(f"  Median absolute difference: {median_abs_diff:.2f} miles")
    print(f"  Rows differing by > 20%: {frac_diff_gt_20 * 100:.2f}% ({n_suspicious:,} rows)")
    print(f"  Sample suspicious rows (flagged, not dropped): {suspicious_sample[:2]}")
    print("\n[City Coordinate Consistency]")
    print(f"  Each pickup city maps to exactly 1 coordinate pair: {pickup_unique_coords}")
    print(f"  Each delivery city maps to exactly 1 coordinate pair: {delivery_unique_coords}")
    print(f"  Mismatches between pickup and delivery for same city: {city_mismatches}")
    print("\n[Target (posted_rate) Distribution]")
    print(f"  Min: ${pr.min():.2f}, Median: ${pr.median():.2f}, Mean: ${pr.mean():.2f}, Max: ${pr.max():.2f}")
    print(f"  Skewness: {pr.skew():.2f} (moderate positive right-skew)")
    print(f"  Zeros: {(pr == 0).sum()}, Negatives: {(pr < 0).sum()}")
    print(f"  IQR Outliers: {outlier_count:,} ({outlier_count / n_rows * 100:.2f}%)")
    print("\n[Leakage & Correlation Check]")
    print(f"  USE_QUOTE_SIGNAL config flag: {use_quote_signal}")
    print(f"  quote_signal vs posted_rate: {corr_qs_rate:.4f} | vs rate_per_mile: {corr_qs_rpm:.4f}")
    print(f"  (quote_signal * distance) vs posted_rate: {corr_qs_dist_rate:.4f}")
    print(f"  market_index vs posted_rate: {corr_mi_rate:.4f} | vs rate_per_mile: {corr_mi_rpm:.4f}")
    print(f"  distance vs posted_rate: {corr_dist_rate:.4f}")
    print("=" * 70)

    return report


def split_protocol_a(
    df: pd.DataFrame,
    seed: int,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    return train_df.copy().reset_index(drop=True), val_df.copy().reset_index(drop=True)


def split_protocol_b(
    df: pd.DataFrame,
    train_date_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        dates = pd.to_datetime(df["date"])
    else:
        dates = df["date"]

    unique_dates = np.sort(dates.unique())
    n_unique = len(unique_dates)
    split_idx = int(np.floor(train_date_fraction * n_unique))
    cutoff_date = unique_dates[split_idx]

    train_mask = dates < cutoff_date
    val_mask = dates >= cutoff_date

    train_df = df[train_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)
    return train_df, val_df
