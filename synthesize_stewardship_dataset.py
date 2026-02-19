import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _coerce_timestamp(value: str) -> pd.Timestamp:
  return pd.to_datetime(value, utc=True, errors="coerce")


def _extract_numeric(stats_obj: dict, key: str) -> float:
  value = stats_obj.get(key)
  if value is None:
    return np.nan
  try:
    return float(value)
  except (TypeError, ValueError):
    return np.nan


def _pick_first_stats(outputs: dict) -> dict:
  for output_obj in outputs.values():
    bands = output_obj.get("bands", {})
    for band_obj in bands.values():
      stats = band_obj.get("stats")
      if isinstance(stats, dict):
        return stats
  return {}


def load_sentinel_features(folder: Path) -> pd.DataFrame:
  rows = []
  json_paths = sorted(folder.glob("*.json"))
  if not json_paths:
    raise ValueError(f"No JSON files found in folder: {folder}")

  for json_path in json_paths:
    location = json_path.stem
    with json_path.open("r", encoding="utf-8-sig") as f:
      payload = json.load(f)

    data_points = payload.get("data", [])
    for item in data_points:
      interval = item.get("interval", {})
      outputs = item.get("outputs", {})
      stats = _pick_first_stats(outputs)
      ts = _coerce_timestamp(interval.get("to"))
      if pd.isna(ts):
        continue

      sample_count = _extract_numeric(stats, "sampleCount")
      no_data = _extract_numeric(stats, "noDataCount")
      no_data_fraction = np.nan
      if pd.notna(sample_count) and sample_count > 0 and pd.notna(no_data):
        no_data_fraction = no_data / sample_count

      rows.append(
        {
          "Timestamp": ts,
          "Location": location,
          "NDVI": _extract_numeric(stats, "mean"),
          "NDVI_Std": _extract_numeric(stats, "stDev"),
          "NoData_Fraction": no_data_fraction,
        }
      )

  df = pd.DataFrame(rows).dropna(subset=["Timestamp"]).sort_values(["Location", "Timestamp"])
  if df.empty:
    raise ValueError("No valid Sentinel feature rows could be parsed.")
  return df


def load_intrusions(labels_csv: Path) -> pd.DataFrame:
  labels = pd.read_csv(labels_csv)
  if "Date" not in labels.columns or "Location" not in labels.columns:
    raise ValueError("Labels CSV must include columns: Date, Location")

  labels = labels.copy()
  labels["Date"] = pd.to_datetime(labels["Date"], utc=True, errors="coerce")
  labels = labels.dropna(subset=["Date", "Location"])
  labels["Location"] = labels["Location"].astype(str).str.strip()
  labels = labels.sort_values(["Location", "Date"])
  labels["Intrusion_Occurred"] = 1
  return labels[["Date", "Location", "Intrusion_Occurred"]]


def engineer_features(features: pd.DataFrame, lag_days: list[int]) -> pd.DataFrame:
  df = features.copy().sort_values(["Location", "Timestamp"])

  grouped = df.groupby("Location", group_keys=False)
  location_mean = grouped["NDVI"].transform("mean")
  location_std = grouped["NDVI"].transform("std").replace(0.0, np.nan)
  df["VSD"] = (df["NDVI"] - location_mean) / location_std

  # Local temporal gradient: positive means vegetation improving, negative means stress worsening.
  df["VSG"] = grouped["NDVI"].diff()

  rolling_vsd = grouped.rolling("30D", on="Timestamp")["VSD"].mean().reset_index(level=0, drop=True)
  df["Coupling_Index"] = rolling_vsd * df["VSG"]

  for lag in lag_days:
    df[f"NDVI_Lag_{lag}"] = grouped["NDVI"].shift(lag)

  # Proxy signal: high cloud/no-data and worsening gradient can indicate rain periods.
  df["Rainfall_Proxy"] = df["NoData_Fraction"].fillna(0.0) + np.maximum(0.0, -df["VSG"].fillna(0.0))
  return df


def merge_labels(
  features_df: pd.DataFrame,
  labels_df: pd.DataFrame,
  tolerance_days: int,
) -> pd.DataFrame:
  left = features_df.sort_values(["Location", "Timestamp"]).copy()
  right = labels_df.sort_values(["Location", "Date"]).copy()
  left["Location"] = left["Location"].astype(str).str.strip()
  right["Location"] = right["Location"].astype(str).str.strip()

  merged = pd.merge_asof(
    left,
    right,
    left_on="Timestamp",
    right_on="Date",
    by="Location",
    direction="nearest",
    tolerance=pd.Timedelta(days=tolerance_days),
  )

  merged["Intrusion_Occurred"] = merged["Intrusion_Occurred"].fillna(0).astype(int)
  merged["Matched_Intrusion_Date"] = merged["Date"]
  merged = merged.drop(columns=["Date"])
  return merged


def apply_lead_label(df: pd.DataFrame, lead_days: int) -> pd.DataFrame:
  out = df.copy()
  out["Target_Risk_Label"] = out["Intrusion_Occurred"]

  for location, group in out.groupby("Location"):
    intrusion_times = group.loc[group["Intrusion_Occurred"] == 1, "Timestamp"].tolist()
    if not intrusion_times:
      continue
    for event_ts in intrusion_times:
      lead_start = event_ts - pd.Timedelta(days=lead_days)
      lead_end = event_ts - pd.Timedelta(days=1)
      mask = (
        (out["Location"] == location)
        & (out["Timestamp"] >= lead_start)
        & (out["Timestamp"] <= lead_end)
      )
      out.loc[mask, "Target_Risk_Label"] = 1

  out["Target_Risk_Label"] = out["Target_Risk_Label"].astype(int)
  return out


def report_imbalance(df: pd.DataFrame) -> None:
  counts = df["Target_Risk_Label"].value_counts(dropna=False).to_dict()
  positives = counts.get(1, 0)
  negatives = counts.get(0, 0)
  total = positives + negatives
  pos_rate = (positives / total) if total else 0.0
  print(f"Class distribution: {counts} | positive_rate={pos_rate:.4f}")
  if pos_rate < 0.2:
    print(
      "Imbalance detected. Placeholder: apply SMOTE (after train split only) or "
      "undersample majority class in training folds."
    )


def make_time_series_splits(df: pd.DataFrame, n_splits: int = 5) -> None:
  try:
    from sklearn.model_selection import TimeSeriesSplit
  except Exception:
    print("scikit-learn unavailable; skipping TimeSeriesSplit preview.")
    return

  ordered = df.sort_values("Timestamp").reset_index(drop=True)
  if len(ordered) <= n_splits:
    print("Not enough rows for TimeSeriesSplit preview.")
    return

  splitter = TimeSeriesSplit(n_splits=n_splits)
  print("TimeSeriesSplit preview (train_end -> test_end row indices):")
  for fold, (train_idx, test_idx) in enumerate(splitter.split(ordered), start=1):
    print(f"  Fold {fold}: {train_idx[-1]} -> {test_idx[-1]}")


def build_dataset(
  features_dir: Path,
  labels_csv: Path,
  output_csv: Path,
  tolerance_days: int,
  lag_days: list[int],
  lead_days: int,
  preview_splits: bool,
) -> None:
  features = load_sentinel_features(features_dir)
  labels = load_intrusions(labels_csv)

  engineered = engineer_features(features, lag_days=lag_days)
  merged = merge_labels(engineered, labels, tolerance_days=tolerance_days)
  labeled = apply_lead_label(merged, lead_days=lead_days)
  report_imbalance(labeled)

  if preview_splits:
    make_time_series_splits(labeled)

  required_columns = [
    "Timestamp",
    "VSD",
    "VSG",
    "Coupling_Index",
    "Rainfall_Proxy",
    "Target_Risk_Label",
  ]
  extra_columns = ["Location", "Intrusion_Occurred"] + [f"NDVI_Lag_{lag}" for lag in lag_days]
  existing_extra = [c for c in extra_columns if c in labeled.columns]
  final_columns = required_columns + existing_extra

  final_df = labeled[final_columns].sort_values(["Location", "Timestamp"], na_position="last")
  output_csv.parent.mkdir(parents=True, exist_ok=True)
  final_df.to_csv(output_csv, index=False)
  print(f"Saved merged training dataset to: {output_csv}")


def parse_lag_days(raw: str) -> list[int]:
  values = []
  for token in raw.split(","):
    token = token.strip()
    if not token:
      continue
    lag = int(token)
    if lag <= 0:
      raise ValueError("Lag days must be positive integers.")
    values.append(lag)
  if not values:
    raise ValueError("At least one lag day must be provided.")
  return sorted(set(values))


def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Synthesize Sentinel Hub statistical JSON features + wildlife intrusion labels "
      "into stewardship_training_data.csv for supervised ML."
    )
  )
  parser.add_argument("--features-dir", required=True, type=Path, help="Folder with Statistical API JSON files.")
  parser.add_argument("--labels-csv", required=True, type=Path, help="CSV with Date and Location columns.")
  parser.add_argument(
    "--output-csv",
    default=Path("stewardship_training_data.csv"),
    type=Path,
    help="Output CSV path.",
  )
  parser.add_argument(
    "--label-tolerance-days",
    default=2,
    type=int,
    help="Nearest-merge tolerance window (days) for matching labels to feature timestamps.",
  )
  parser.add_argument(
    "--lag-days",
    default="5,10",
    type=str,
    help="Comma-separated lag day values for NDVI lag features, e.g. 5,10.",
  )
  parser.add_argument(
    "--lead-days",
    default=3,
    type=int,
    help="Number of days before a confirmed intrusion to label as high risk.",
  )
  parser.add_argument(
    "--preview-timeseries-split",
    action="store_true",
    help="Print a TimeSeriesSplit preview to verify leakage-safe temporal splitting.",
  )
  args = parser.parse_args()

  lag_days = parse_lag_days(args.lag_days)
  build_dataset(
    features_dir=args.features_dir,
    labels_csv=args.labels_csv,
    output_csv=args.output_csv,
    tolerance_days=args.label_tolerance_days,
    lag_days=lag_days,
    lead_days=args.lead_days,
    preview_splits=args.preview_timeseries_split,
  )


if __name__ == "__main__":
  main()
