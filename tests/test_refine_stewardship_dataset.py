import pandas as pd

from refine_stewardship_dataset import clean_and_engineer


def test_lag_features_use_past_only() -> None:
  df = pd.DataFrame(
    {
      "interval_to": pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC"),
      "Location": ["A"] * 12,
      "NDVI": list(range(12)),
      "VSD": [0.1] * 12,
      "VSG": [0.2] * 12,
      "Target_Risk_Label": [0] * 12,
      "Intrusion_Occurred": [0] * 12,
    }
  )
  result = clean_and_engineer(
    df,
    time_col="interval_to",
    max_consecutive_missing=3,
    lag_days=[5, 10],
    imputation_mode="time",
    strict_missing_vsd_vsg=True,
  )
  assert result["NDVI_Lag_5"].iloc[5] == 0
  assert result["NDVI_Lag_10"].iloc[10] == 0


def test_target_alignment_with_intrusion() -> None:
  df = pd.DataFrame(
    {
      "interval_to": pd.date_range("2024-02-01", periods=4, freq="D", tz="UTC"),
      "Location": ["A"] * 4,
      "NDVI": [0.2, 0.3, 0.4, 0.5],
      "VSD": [0.1, 0.1, 0.2, 0.2],
      "VSG": [0.0, 0.1, 0.1, 0.1],
      "Target_Risk_Label": [0, 0, 0, 0],
      "Intrusion_Occurred": [0, 0, 1, 0],
    }
  )
  result = clean_and_engineer(
    df,
    time_col="interval_to",
    max_consecutive_missing=3,
    lag_days=[5, 10],
    imputation_mode="time",
    strict_missing_vsd_vsg=True,
  )
  assert result["Target_Risk_Label"].iloc[2] == 1


def test_drop_long_missing_sequences() -> None:
  df = pd.DataFrame(
    {
      "interval_to": pd.date_range("2024-03-01", periods=8, freq="D", tz="UTC"),
      "Location": ["A"] * 8,
      "NDVI": [0.1, None, None, None, None, 0.2, 0.3, 0.4],
      "VSD": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
      "VSG": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
      "Target_Risk_Label": [0] * 8,
      "Intrusion_Occurred": [0] * 8,
    }
  )
  result = clean_and_engineer(
    df,
    time_col="interval_to",
    max_consecutive_missing=3,
    lag_days=[5, 10],
    imputation_mode="time",
    strict_missing_vsd_vsg=True,
  )
  assert len(result) < len(df)
