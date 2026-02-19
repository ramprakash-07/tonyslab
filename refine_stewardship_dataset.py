import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("stewardship_refinery")


def configure_logging(level: str) -> None:
  numeric_level = getattr(logging, level.upper(), logging.INFO)
  logging.basicConfig(
    level=numeric_level,
    format="%(asctime)s | %(levelname)s | %(message)s",
  )


def parse_lag_days(raw: str) -> list[int]:
  lags = []
  for token in raw.split(","):
    token = token.strip()
    if not token:
      continue
    lag = int(token)
    if lag <= 0:
      raise ValueError("Lag days must be positive integers.")
    lags.append(lag)
  lags = sorted(set(lags))
  if not lags:
    raise ValueError("At least one lag day must be provided.")
  return lags


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
  for col in candidates:
    if col in df.columns:
      return col
  return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
  out = df.copy()
  rename_map = {}

  ndvi_col = _first_existing(out, ["NDVI", "ndvi", "NDVI_Mean", "mean_ndvi"])
  ndvi_std_col = _first_existing(out, ["NDVI_Std", "ndvi_std", "NDVI_stDev", "stDev"])
  vsd_col = _first_existing(out, ["VSD", "vsd"])
  vsg_col = _first_existing(out, ["VSG", "vsg"])
  coupling_col = _first_existing(out, ["Coupling_Index", "Ecological_Coupling_Score", "Coupling"])
  target_col = _first_existing(out, ["Target_Risk_Label", "target_risk_label", "Target"])
  intrusion_col = _first_existing(out, ["Intrusion_Occurred", "intrusion_occurred", "Intrusion"])

  if ndvi_col and ndvi_col != "NDVI":
    rename_map[ndvi_col] = "NDVI"
  if ndvi_std_col and ndvi_std_col != "NDVI_Std":
    rename_map[ndvi_std_col] = "NDVI_Std"
  if vsd_col and vsd_col != "VSD":
    rename_map[vsd_col] = "VSD"
  if vsg_col and vsg_col != "VSG":
    rename_map[vsg_col] = "VSG"
  if coupling_col and coupling_col != "Coupling_Index":
    rename_map[coupling_col] = "Coupling_Index"
  if target_col and target_col != "Target_Risk_Label":
    rename_map[target_col] = "Target_Risk_Label"
  if intrusion_col and intrusion_col != "Intrusion_Occurred":
    rename_map[intrusion_col] = "Intrusion_Occurred"

  out = out.rename(columns=rename_map)
  return out


def _ensure_timestamps(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
  out = df.copy()
  for col in ["interval_from", "interval_to", "Timestamp"]:
    if col in out.columns:
      out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")

  if "interval_to" in out.columns:
    time_col = "interval_to"
  elif "Timestamp" in out.columns:
    time_col = "Timestamp"
  else:
    raise ValueError("Input must include at least one time column: interval_to or Timestamp")

  if "interval_from" not in out.columns:
    out["interval_from"] = out[time_col]
  if "interval_to" not in out.columns:
    out["interval_to"] = out[time_col]
  if "Timestamp" not in out.columns:
    out["Timestamp"] = out[time_col]

  out = out.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
  return out, time_col


def _long_missing_mask(series: pd.Series, max_consecutive_missing: int) -> pd.Series:
  missing = series.isna()
  group_ids = missing.ne(missing.shift(fill_value=False)).cumsum()
  run_lengths = missing.groupby(group_ids).transform("sum")
  return missing & (run_lengths > max_consecutive_missing)


def _time_interpolate_group(group: pd.DataFrame, cols: list[str], time_col: str) -> pd.DataFrame:
  ordered = group.sort_values(time_col).copy()
  ordered = ordered.set_index(time_col)
  ordered[cols] = ordered[cols].interpolate(method="time", limit_direction="both")
  return ordered.reset_index()


def _apply_optional_imputation(df: pd.DataFrame, cols: list[str], mode: str) -> pd.DataFrame:
  out = df.copy()
  if mode == "time":
    return out

  usable_cols = [c for c in cols if c in out.columns and out[c].notna().any()]
  if not usable_cols:
    return out
  matrix = out[usable_cols]
  if mode == "knn":
    try:
      from sklearn.impute import KNNImputer

      imputer = KNNImputer(n_neighbors=3)
      out[usable_cols] = imputer.fit_transform(matrix)
      LOGGER.info("Applied KNN imputation for %s", usable_cols)
    except Exception as exc:
      LOGGER.warning("KNN imputation unavailable; keeping time interpolation only (%s)", exc)
  elif mode == "iterative":
    try:
      from sklearn.experimental import enable_iterative_imputer  # noqa: F401
      from sklearn.impute import IterativeImputer

      imputer = IterativeImputer(random_state=42, max_iter=25)
      out[usable_cols] = imputer.fit_transform(matrix)
      LOGGER.info("Applied Iterative imputation for %s", usable_cols)
    except Exception as exc:
      LOGGER.warning("Iterative imputation unavailable; keeping time interpolation only (%s)", exc)
  return out


def clean_and_engineer(
  df: pd.DataFrame,
  time_col: str,
  max_consecutive_missing: int,
  lag_days: list[int],
  imputation_mode: str,
  strict_missing_vsd_vsg: bool,
) -> pd.DataFrame:
  out = df.copy()
  if "Location" not in out.columns:
    out["Location"] = "single_site"

  numeric_cols = ["NDVI", "NDVI_Std", "VSD", "VSG", "Coupling_Index", "Intrusion_Occurred", "Target_Risk_Label"]
  for col in numeric_cols:
    if col in out.columns:
      out[col] = pd.to_numeric(out[col], errors="coerce")

  for required in ["NDVI", "VSD", "VSG"]:
    if required not in out.columns:
      out[required] = np.nan

  drop_mask = pd.Series(False, index=out.index)
  gap_checked_cols = [c for c in ["NDVI", "VSD", "VSG"] if out[c].notna().any()]
  for col in gap_checked_cols:
    col_drop = out.groupby("Location", group_keys=False)[col].transform(
      lambda s: _long_missing_mask(s, max_consecutive_missing)
    )
    drop_mask = drop_mask | col_drop

  dropped_rows = int(drop_mask.sum())
  if dropped_rows:
    LOGGER.info("Dropping %s rows with >%s consecutive missing values", dropped_rows, max_consecutive_missing)
  out = out.loc[~drop_mask].copy()

  out = (
    out.groupby("Location", group_keys=False)
    .apply(lambda g: _time_interpolate_group(g, ["NDVI", "VSD", "VSG"], time_col))
    .reset_index(drop=True)
  )
  if "Location" not in out.columns and "Location" in df.columns:
    out["Location"] = df["Location"].iloc[0]

  out = _apply_optional_imputation(out, ["NDVI", "VSD", "VSG"], mode=imputation_mode)

  if out["VSD"].isna().all() or out["VSG"].isna().all():
    msg = "VSD or VSG remains entirely missing after cleaning; cannot build reliable Coupling_Index."
    if strict_missing_vsd_vsg:
      raise ValueError(msg)
    LOGGER.warning(msg)

  if "Coupling_Index" not in out.columns:
    out["Coupling_Index"] = np.nan
  coupling_missing = out["Coupling_Index"].isna()
  if coupling_missing.any():
    out.loc[coupling_missing, "Coupling_Index"] = (
      (0.7 * out.loc[coupling_missing, "VSD"] * out.loc[coupling_missing, "VSG"])
      + (0.3 * ((out.loc[coupling_missing, "VSD"] + out.loc[coupling_missing, "VSG"]) / 2.0))
    )
    LOGGER.info("Filled Coupling_Index for %s rows", int(coupling_missing.sum()))

  for lag in lag_days:
    lag_col = f"NDVI_Lag_{lag}"
    safe_lag = out.groupby("Location")["NDVI"].shift(lag)
    if lag_col in out.columns:
      mismatches = (out[lag_col].fillna(0) - safe_lag.fillna(0)).abs() > 1e-12
      if mismatches.any():
        LOGGER.info("Recomputed %s: corrected %s rows to prevent leakage", lag_col, int(mismatches.sum()))
    out[lag_col] = safe_lag

  if "Target_Risk_Label" not in out.columns:
    out["Target_Risk_Label"] = 0
  if "Intrusion_Occurred" in out.columns:
    before = out["Target_Risk_Label"].fillna(0).astype(int)
    out["Target_Risk_Label"] = np.maximum(before, out["Intrusion_Occurred"].fillna(0).astype(int))
    corrected = int((out["Target_Risk_Label"] != before).sum())
    if corrected:
      LOGGER.info("Aligned Target_Risk_Label with Intrusion_Occurred for %s rows", corrected)

  out["Target_Risk_Label"] = out["Target_Risk_Label"].fillna(0).astype(int)

  class_counts = out["Target_Risk_Label"].value_counts().to_dict()
  LOGGER.info("Class distribution: %s", class_counts)
  if len(class_counts) == 2:
    majority = max(class_counts.values())
    minority = min(class_counts.values())
    if majority > 0 and (minority / majority) < 0.2:
      LOGGER.warning("High imbalance detected: consider SMOTE/undersampling in training only")

  if out["Location"].nunique(dropna=True) <= 1:
    out = out.drop(columns=["Location"])

  return out


def save_correlation_heatmap(df: pd.DataFrame, output_path: Path, lag_days: list[int]) -> None:
  cols = ["NDVI", "NDVI_Std", "VSD", "VSG", "Coupling_Index"]
  cols.extend([f"NDVI_Lag_{lag}" for lag in lag_days])
  cols.append("Target_Risk_Label")

  existing = [c for c in cols if c in df.columns]
  if df.empty or not existing:
    LOGGER.warning("Skipped heatmap generation (empty dataset or missing numeric columns)")
    return

  corr = df[existing].corr(numeric_only=True)
  if corr.empty:
    LOGGER.warning("Skipped heatmap generation (empty correlation matrix)")
    return

  try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    LOGGER.info("Saved correlation heatmap to: %s", output_path)
  except Exception as exc:
    LOGGER.warning("Skipped heatmap generation (%s)", exc)


def finalize_columns(df: pd.DataFrame, lag_days: list[int]) -> pd.DataFrame:
  preferred = [
    "interval_from",
    "interval_to",
    "Timestamp",
    "NDVI",
    "NDVI_Std",
    "VSD",
    "VSG",
    "Coupling_Index",
  ]
  preferred.extend([f"NDVI_Lag_{lag}" for lag in lag_days])
  preferred.extend(["Intrusion_Occurred", "Target_Risk_Label"])
  cols = [c for c in preferred if c in df.columns]
  return df[cols].copy()


def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Refine ecological time-series data into model-ready stewardship CSV. "
      "Expected input includes time columns (interval_to or Timestamp), NDVI/VSD/VSG, and target fields."
    )
  )
  parser.add_argument("--input-csv", default=Path("stewardship_training_data.csv"), type=Path)
  parser.add_argument("--output-csv", default=Path("refined_stewardship_model_ready.csv"), type=Path)
  parser.add_argument(
    "--heatmap-path",
    default=Path("stewardship_feature_correlation_heatmap.png"),
    type=Path,
  )
  parser.add_argument(
    "--lag-days",
    default="5,10",
    type=str,
    help="Comma-separated lag days (e.g., 5,10,15).",
  )
  parser.add_argument(
    "--max-consecutive-missing",
    default=3,
    type=int,
    help="Drop rows that belong to >N consecutive missing runs.",
  )
  parser.add_argument(
    "--imputation-mode",
    choices=["time", "knn", "iterative"],
    default="time",
    help="Imputation strategy after time interpolation.",
  )
  parser.add_argument(
    "--strict-missing-vsd-vsg",
    action="store_true",
    help="Fail if VSD or VSG remain entirely missing after preprocessing.",
  )
  parser.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    help="Logging verbosity.",
  )
  args = parser.parse_args()

  configure_logging(args.log_level)
  lag_days = parse_lag_days(args.lag_days)
  if args.max_consecutive_missing < 1:
    raise ValueError("--max-consecutive-missing must be >= 1")

  LOGGER.info("Loading input CSV: %s", args.input_csv)
  df = pd.read_csv(args.input_csv)
  df = _normalize_columns(df)
  df, time_col = _ensure_timestamps(df)
  refined = clean_and_engineer(
    df,
    time_col=time_col,
    max_consecutive_missing=args.max_consecutive_missing,
    lag_days=lag_days,
    imputation_mode=args.imputation_mode,
    strict_missing_vsd_vsg=args.strict_missing_vsd_vsg,
  )
  final_df = finalize_columns(refined, lag_days=lag_days)

  args.output_csv.parent.mkdir(parents=True, exist_ok=True)
  final_df.to_csv(args.output_csv, index=False)
  LOGGER.info("Saved refined dataset to: %s", args.output_csv)

  save_correlation_heatmap(final_df, args.heatmap_path, lag_days=lag_days)


if __name__ == "__main__":
  main()
