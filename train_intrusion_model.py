import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
  ConfusionMatrixDisplay,
  PrecisionRecallDisplay,
  classification_report,
  confusion_matrix,
  fbeta_score,
  make_scorer,
  precision_score,
  recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline


class EcologicalFeatureBuilder(BaseEstimator, TransformerMixin):
  def __init__(self, ndvi_col: str = "NDVI", vsd_col: str = "VSD"):
    self.ndvi_col = ndvi_col
    self.vsd_col = vsd_col

  def fit(self, X: pd.DataFrame, y: Any = None) -> "EcologicalFeatureBuilder":
    return self

  def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    if "Timestamp" not in out.columns:
      raise ValueError("Timestamp column is required for temporal feature engineering.")

    ts = pd.to_datetime(out["Timestamp"], utc=True, errors="coerce")
    out["month_sin"] = np.sin((2 * np.pi * ts.dt.month.fillna(1)) / 12.0)
    out["month_cos"] = np.cos((2 * np.pi * ts.dt.month.fillna(1)) / 12.0)

    for base_col in [self.ndvi_col, self.vsd_col]:
      if base_col in out.columns:
        out[f"{base_col}_roll_mean_7"] = out[base_col].rolling(7, min_periods=1).mean()
        out[f"{base_col}_roll_std_7"] = out[base_col].rolling(7, min_periods=2).std()
        out[f"{base_col}_roll_mean_14"] = out[base_col].rolling(14, min_periods=1).mean()
        out[f"{base_col}_roll_std_14"] = out[base_col].rolling(14, min_periods=2).std()

    out = out.drop(columns=[c for c in ["Timestamp", "interval_from", "interval_to"] if c in out.columns], errors="ignore")
    for col in out.columns:
      out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


class DropAllNaNColumns(BaseEstimator, TransformerMixin):
  def __init__(self):
    self.columns_: list[str] = []

  def fit(self, X: pd.DataFrame, y: Any = None) -> "DropAllNaNColumns":
    frame = pd.DataFrame(X)
    self.columns_ = [c for c in frame.columns if frame[c].notna().any()]
    return self

  def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(X)
    cols = [c for c in self.columns_ if c in frame.columns]
    return frame[cols]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Advanced ecological ML pipeline with nested time-series CV, calibration, SHAP, and drift checks."
  )
  parser.add_argument("--input-csv", default=Path("refined_stewardship_model_ready.csv"), type=Path)
  parser.add_argument("--current-csv", default=None, type=Path, help="Optional latest feature CSV for drift check.")
  parser.add_argument("--target-col", default="Target_Risk_Label")
  parser.add_argument("--n-iter", default=15, type=int)
  parser.add_argument("--random-state", default=42, type=int)
  parser.add_argument("--outer-splits", default=5, type=int)
  parser.add_argument("--inner-splits", default=3, type=int)
  parser.add_argument("--backtest-window-days", default=60, type=int)
  parser.add_argument("--high-risk-threshold", default=0.5, type=float)
  parser.add_argument("--model-output", default=Path("stewardship_intrusion_model.joblib"), type=Path)
  parser.add_argument("--metrics-output", default=Path("stewardship_model_metrics.json"), type=Path)
  parser.add_argument("--feature-importance-plot", default=Path("stewardship_feature_importance.png"), type=Path)
  parser.add_argument("--confusion-matrix-plot", default=Path("stewardship_confusion_matrix.png"), type=Path)
  parser.add_argument("--pr-curve-plot", default=Path("stewardship_precision_recall_curve.png"), type=Path)
  parser.add_argument("--drift-output", default=Path("stewardship_drift_report.json"), type=Path)
  parser.add_argument("--backtest-output", default=Path("stewardship_backtest_report.json"), type=Path)
  parser.add_argument("--prediction-log-output", default=Path("stewardship_prediction_log.csv"), type=Path)
  parser.add_argument("--shap-output-dir", default=Path("shap_force_plots"), type=Path)
  return parser.parse_args()


def _find_time_column(df: pd.DataFrame) -> str:
  for col in ["Timestamp", "interval_to", "interval_from"]:
    if col in df.columns:
      return col
  raise ValueError("Input must include one of Timestamp, interval_to, interval_from.")


def prepare_dataframe(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
  if target_col not in df.columns:
    raise ValueError(f"Missing target column: {target_col}")

  time_col = _find_time_column(df)
  data = df.copy()
  data["Timestamp"] = pd.to_datetime(data[time_col], utc=True, errors="coerce")
  data = data.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
  y = data[target_col].fillna(0).astype(int)

  feature_cols = [c for c in data.columns if c not in {target_col, "Intrusion_Occurred"}]
  X = data[feature_cols].copy()
  for col in X.columns:
    if col != "Timestamp":
      X[col] = pd.to_numeric(X[col], errors="coerce")
  return X, y, data["Timestamp"]


def split_feature_frame(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
  expanded = EcologicalFeatureBuilder().fit_transform(X)
  numeric = expanded.apply(pd.to_numeric, errors="coerce")
  keep_cols = [c for c in numeric.columns if numeric[c].notna().any()]
  return numeric[keep_cols], keep_cols


def _smotetomek_available() -> bool:
  try:
    from imblearn.combine import SMOTETomek  # noqa: F401
    from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: F401

    return True
  except Exception:
    return False


def _build_boosting_estimators(random_state: int, scale_pos_weight: float) -> list[tuple[str, Any, dict]]:
  estimators = []
  try:
    from lightgbm import LGBMClassifier

    estimators.append(
      (
        "LightGBM",
        LGBMClassifier(
          random_state=random_state,
          n_jobs=-1,
          class_weight="balanced",
          scale_pos_weight=scale_pos_weight,
          verbose=-1,
        ),
        {
          "model__n_estimators": [200, 400, 700],
          "model__max_depth": [-1, 4, 6, 8],
          "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
          "model__num_leaves": [15, 31, 63],
          "model__subsample": [0.75, 0.9, 1.0],
        },
      )
    )
  except Exception:
    pass

  try:
    from catboost import CatBoostClassifier

    estimators.append(
      (
        "CatBoost",
        CatBoostClassifier(
          random_state=random_state,
          verbose=False,
          auto_class_weights="Balanced",
        ),
        {
          "model__iterations": [200, 400, 700],
          "model__depth": [4, 6, 8],
          "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
          "model__l2_leaf_reg": [1, 3, 5, 7],
        },
      )
    )
  except Exception:
    pass
  return estimators


def _build_hybrid_estimator(random_state: int) -> tuple[str, Any, dict] | None:
  try:
    from imblearn.combine import SMOTETomek
    from imblearn.ensemble import BalancedRandomForestClassifier
    from imblearn.pipeline import Pipeline as ImbPipeline
  except Exception:
    return None

  estimator = ImbPipeline(
    steps=[
      ("features", EcologicalFeatureBuilder()),
      ("dropna_cols", DropAllNaNColumns()),
      ("imputer", IterativeImputer(random_state=random_state, max_iter=20)),
      ("resample", SMOTETomek(random_state=random_state)),
      (
        "model",
        BalancedRandomForestClassifier(
          random_state=random_state,
          n_estimators=300,
          n_jobs=-1,
        ),
      ),
    ]
  )
  params = {
    "model__n_estimators": [200, 400, 700],
    "model__max_depth": [None, 5, 8, 12],
    "model__min_samples_leaf": [1, 2, 4],
  }
  return "SMOTETomek+BalancedRF", estimator, params


def _build_standard_pipeline(base_model: Any, random_state: int) -> Pipeline:
  return Pipeline(
    steps=[
      ("features", EcologicalFeatureBuilder()),
      ("dropna_cols", DropAllNaNColumns()),
      ("imputer", IterativeImputer(random_state=random_state, max_iter=20)),
      ("model", base_model),
    ]
  )


def _nested_cv_model_selection(
  X: pd.DataFrame,
  y: pd.Series,
  candidate_estimators: list[tuple[str, Any, dict]],
  outer_splits: int,
  inner_splits: int,
  n_iter: int,
  random_state: int,
) -> tuple[str, Any, dict, list[dict], tuple[np.ndarray, np.ndarray]]:
  scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
  outer_cv = TimeSeriesSplit(n_splits=outer_splits)
  last_split = None
  all_results = []
  avg_scores = {}

  for name, estimator, params in candidate_estimators:
    fold_scores = []
    failed = False
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
      last_split = (train_idx, test_idx)
      X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
      y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

      inner_cv = TimeSeriesSplit(n_splits=inner_splits)
      search = RandomizedSearchCV(
        estimator=clone(estimator),
        param_distributions=params,
        n_iter=min(n_iter, 10 if len(X_train) < 300 else n_iter),
        scoring=scorer,
        cv=inner_cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
      )
      try:
        search.fit(X_train, y_train)
      except Exception as exc:
        print(f"{name} skipped on fold {fold_idx}: {exc}")
        failed = True
        break
      y_pred = search.best_estimator_.predict(X_test)
      fold_f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
      fold_scores.append(float(fold_f2))
      all_results.append(
        {
          "model": name,
          "fold": fold_idx,
          "f2_score": float(fold_f2),
          "best_params": search.best_params_,
        }
      )
      print(f"{name} outer fold {fold_idx} F2={fold_f2:.4f}")

    if failed or not fold_scores:
      avg_scores[name] = -1.0
      print(f"{name} excluded from selection (training failed under current class distribution).")
      continue

    avg_scores[name] = float(np.mean(fold_scores))
    print(f"{name} nested CV average F2={avg_scores[name]:.4f}")

  valid_scores = {k: v for k, v in avg_scores.items() if v >= 0}
  if not valid_scores:
    raise RuntimeError("No candidate estimator could be trained successfully.")
  best_name = max(valid_scores, key=valid_scores.get)
  best_tpl = next(item for item in candidate_estimators if item[0] == best_name)
  return best_name, best_tpl[1], best_tpl[2], all_results, last_split


def fit_best_model(
  X_train: pd.DataFrame,
  y_train: pd.Series,
  model_name: str,
  estimator: Any,
  params: dict,
  inner_splits: int,
  n_iter: int,
  random_state: int,
) -> Any:
  scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
  inner_cv = TimeSeriesSplit(n_splits=inner_splits)
  search = RandomizedSearchCV(
    estimator=clone(estimator),
    param_distributions=params,
    n_iter=n_iter,
    scoring=scorer,
    cv=inner_cv,
    random_state=random_state,
    n_jobs=-1,
    refit=True,
  )
  search.fit(X_train, y_train)
  print(f"{model_name} tuned best CV F2={search.best_score_:.4f}")
  print(f"{model_name} tuned params={search.best_params_}")
  return search


def calibrate_model(best_estimator: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
  class_counts = y_train.value_counts()
  min_class = int(class_counts.min()) if not class_counts.empty else 0
  if min_class < 2:
    print("Calibration skipped: not enough positive/negative samples per fold.")
    return best_estimator

  cv_splits = min(3, min_class)
  calibrator = CalibratedClassifierCV(
    estimator=best_estimator,
    method="isotonic",
    cv=TimeSeriesSplit(n_splits=cv_splits),
  )
  try:
    calibrator.fit(X_train, y_train)
    print("Applied isotonic calibration.")
    return calibrator
  except Exception as exc:
    print(f"Calibration fallback (isotonic failed): {exc}")
    fallback = CalibratedClassifierCV(
      estimator=best_estimator,
      method="sigmoid",
      cv=TimeSeriesSplit(n_splits=cv_splits),
    )
    fallback.fit(X_train, y_train)
    return fallback


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict:
  y_prob = model.predict_proba(X_test)[:, 1]
  y_pred = (y_prob >= threshold).astype(int)
  return {
    "f2_score": float(fbeta_score(y_test, y_pred, beta=2, zero_division=0)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
    "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    "y_prob": y_prob,
    "y_pred": y_pred,
  }


def save_eval_plots(y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray, cm_path: Path, pr_path: Path) -> None:
  disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=[0, 1], cmap="Blues")
  disp.ax_.set_title("Confusion Matrix (Calibrated Holdout)")
  plt.tight_layout()
  plt.savefig(cm_path, dpi=150)
  plt.close()

  pr = PrecisionRecallDisplay.from_predictions(y_test, y_prob)
  pr.ax_.set_title("Precision-Recall Curve (Calibrated Holdout)")
  plt.tight_layout()
  plt.savefig(pr_path, dpi=150)
  plt.close()


def save_feature_importance_plot(model: Any, X: pd.DataFrame, y: pd.Series, feature_path: Path) -> None:
  fitted = clone(model)
  fitted.fit(X, y)

  if hasattr(fitted, "named_steps"):
    base = fitted.named_steps.get("model", fitted)
    selected_features = fitted.named_steps.get("dropna_cols", None)
    if selected_features is not None:
      feature_names = selected_features.columns_
    else:
      feature_names = []
  else:
    base = fitted
    feature_names = []

  if not hasattr(base, "feature_importances_"):
    print("Skipping feature importance plot (no feature_importances_ attribute).")
    return

  importances = np.array(base.feature_importances_)
  if len(importances) != len(feature_names):
    print("Skipping feature importance plot (feature length mismatch).")
    return

  top_idx = np.argsort(importances)[::-1][:20]
  plt.figure(figsize=(10, 8))
  plt.barh(np.array(feature_names)[top_idx][::-1], importances[top_idx][::-1], color="#366b45")
  plt.title("Ecological Feature Importance (Top 20)")
  plt.xlabel("Importance")
  plt.tight_layout()
  plt.savefig(feature_path, dpi=150)
  plt.close()


def _season_name(ts: pd.Timestamp) -> str:
  month = int(ts.month)
  if month in {6, 7, 8, 9}:
    return "Monsoon"
  if month in {3, 4, 5}:
    return "Summer"
  return "Other"


def backtest_windows(
  base_estimator: Any,
  X: pd.DataFrame,
  y: pd.Series,
  timestamps: pd.Series,
  window_days: int,
  threshold: float,
) -> list[dict]:
  start = timestamps.min()
  end = timestamps.max()
  window = pd.Timedelta(days=window_days)
  reports = []
  current_start = start

  while current_start < end:
    current_end = current_start + window
    test_mask = (timestamps >= current_start) & (timestamps < current_end)
    train_mask = timestamps < current_start
    if train_mask.sum() < 10 or test_mask.sum() == 0:
      current_start = current_end
      continue

    model = clone(base_estimator)
    model.fit(X.loc[train_mask], y.loc[train_mask])
    y_prob = model.predict_proba(X.loc[test_mask])[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y.loc[test_mask]
    reports.append(
      {
        "window_start": str(current_start),
        "window_end": str(current_end),
        "season": _season_name(current_start),
        "n_samples": int(test_mask.sum()),
        "f2_score": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
      }
    )
    current_start = current_end
  return reports


def run_drift_check(
  train_df: pd.DataFrame,
  current_df: pd.DataFrame,
  output_path: Path,
  alpha: float = 0.05,
) -> dict:
  report = {}
  for col in ["VSD", "VSG"]:
    if col not in train_df.columns or col not in current_df.columns:
      report[col] = {"status": "missing_column"}
      continue
    t = pd.to_numeric(train_df[col], errors="coerce").dropna()
    c = pd.to_numeric(current_df[col], errors="coerce").dropna()
    if t.empty or c.empty:
      report[col] = {"status": "insufficient_data"}
      continue
    ks_stat, p_val = ks_2samp(t.values, c.values)
    report[col] = {
      "ks_stat": float(ks_stat),
      "p_value": float(p_val),
      "drift_detected": bool(p_val < alpha),
    }

  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
  return report


def log_prediction_confidence_with_shap(
  model: Any,
  X_frame: pd.DataFrame,
  timestamps: pd.Series,
  threshold: float,
  output_csv: Path,
  shap_output_dir: Path,
) -> None:
  y_prob = model.predict_proba(X_frame)[:, 1]
  y_pred = (y_prob >= threshold).astype(int)

  shap_values = None
  top_shap_features = ["shap_unavailable"] * len(X_frame)
  try:
    import shap

    shap_output_dir.mkdir(parents=True, exist_ok=True)
    explainer = shap.Explainer(model, X_frame)
    shap_values = explainer(X_frame)

    high_risk_idx = np.where(y_pred == 1)[0]
    for row_idx in high_risk_idx:
      row_shap = shap_values[row_idx]
      abs_vals = np.abs(row_shap.values)
      order = np.argsort(abs_vals)[::-1][:3]
      top_feats = [f"{row_shap.feature_names[i]}:{row_shap.values[i]:.4f}" for i in order]
      top_shap_features[row_idx] = "; ".join(top_feats)

      try:
        html = shap.plots.force(
          row_shap.base_values,
          row_shap.values,
          features=X_frame.iloc[row_idx],
          feature_names=X_frame.columns.tolist(),
          matplotlib=False,
          show=False,
        )
        shap.save_html(str(shap_output_dir / f"force_plot_idx_{row_idx}.html"), html)
      except Exception:
        pass
  except Exception as exc:
    print(f"SHAP unavailable or failed: {exc}")

  output = pd.DataFrame(
    {
      "Timestamp": timestamps.astype(str),
      "Prediction_Probability": y_prob,
      "Predicted_Label": y_pred,
      "Confidence": np.where(y_pred == 1, y_prob, 1 - y_prob),
      "Top_SHAP_Contributors": top_shap_features,
    }
  )
  output_csv.parent.mkdir(parents=True, exist_ok=True)
  output.to_csv(output_csv, index=False)


def main() -> None:
  args = parse_args()
  raw_df = pd.read_csv(args.input_csv)
  X_raw, y, timestamps = prepare_dataframe(raw_df, target_col=args.target_col)

  positives = int(y.sum())
  negatives = int((y == 0).sum())
  if positives == 0:
    raise ValueError("No positive class found in target.")
  scale_pos_weight = negatives / positives
  print(f"Class distribution: negatives={negatives}, positives={positives}, scale_pos_weight={scale_pos_weight:.2f}")

  candidate_estimators = []
  for name, base_model, params in _build_boosting_estimators(args.random_state, scale_pos_weight):
    candidate_estimators.append((name, _build_standard_pipeline(base_model, args.random_state), params))

  hybrid = _build_hybrid_estimator(args.random_state)
  if hybrid is not None:
    candidate_estimators.append(hybrid)

  if not candidate_estimators:
    raise RuntimeError(
      "No required model backends available. Install one of: lightgbm, catboost, imbalanced-learn."
    )

  best_name, best_estimator, best_params, nested_results, last_split = _nested_cv_model_selection(
    X_raw,
    y,
    candidate_estimators,
    outer_splits=args.outer_splits,
    inner_splits=args.inner_splits,
    n_iter=args.n_iter,
    random_state=args.random_state,
  )
  print(f"Selected model from nested CV: {best_name}")

  train_idx, test_idx = last_split
  X_train, X_test = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
  y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
  ts_test = timestamps.iloc[test_idx]

  tuned = fit_best_model(
    X_train,
    y_train,
    model_name=best_name,
    estimator=best_estimator,
    params=best_params,
    inner_splits=args.inner_splits,
    n_iter=args.n_iter,
    random_state=args.random_state,
  )
  calibrated_model = calibrate_model(tuned.best_estimator_, X_train, y_train)
  holdout_eval = evaluate_model(calibrated_model, X_test, y_test, threshold=args.high_risk_threshold)
  print(f"Calibrated holdout F2={holdout_eval['f2_score']:.4f}")

  args.confusion_matrix_plot.parent.mkdir(parents=True, exist_ok=True)
  save_eval_plots(
    y_test=y_test,
    y_pred=holdout_eval["y_pred"],
    y_prob=holdout_eval["y_prob"],
    cm_path=args.confusion_matrix_plot,
    pr_path=args.pr_curve_plot,
  )

  save_feature_importance_plot(
    model=tuned.best_estimator_,
    X=X_train,
    y=y_train,
    feature_path=args.feature_importance_plot,
  )

  # Backtesting on rolling 60-day windows for seasonal stability.
  backtest_report = backtest_windows(
    base_estimator=calibrated_model,
    X=X_raw,
    y=y,
    timestamps=timestamps,
    window_days=args.backtest_window_days,
    threshold=args.high_risk_threshold,
  )
  args.backtest_output.parent.mkdir(parents=True, exist_ok=True)
  with args.backtest_output.open("w", encoding="utf-8") as f:
    json.dump(backtest_report, f, indent=2)

  current_df = pd.read_csv(args.current_csv) if args.current_csv else raw_df.copy()
  drift_report = run_drift_check(
    train_df=raw_df.iloc[train_idx],
    current_df=current_df,
    output_path=args.drift_output,
  )

  # Refit calibrated model on all available history before export.
  final_search = fit_best_model(
    X_raw,
    y,
    model_name=best_name,
    estimator=best_estimator,
    params=best_params,
    inner_splits=args.inner_splits,
    n_iter=args.n_iter,
    random_state=args.random_state,
  )
  final_calibrated = calibrate_model(final_search.best_estimator_, X_raw, y)

  processed_all, processed_cols = split_feature_frame(X_raw)
  log_prediction_confidence_with_shap(
    model=final_calibrated,
    X_frame=X_raw,
    timestamps=timestamps,
    threshold=args.high_risk_threshold,
    output_csv=args.prediction_log_output,
    shap_output_dir=args.shap_output_dir,
  )

  export_bundle = {
    "model": final_calibrated,
    "base_estimator_name": best_name,
    "best_params": final_search.best_params_,
    "feature_columns_after_engineering": processed_cols,
    "training_time_range": [str(timestamps.min()), str(timestamps.max())],
    "drift_baseline_features": ["VSD", "VSG"],
  }
  args.model_output.parent.mkdir(parents=True, exist_ok=True)
  joblib.dump(export_bundle, args.model_output)
  print(f"Saved model bundle to: {args.model_output}")

  metrics = {
    "selected_model": best_name,
    "nested_cv_results": nested_results,
    "holdout_metrics": {
      "f2_score": holdout_eval["f2_score"],
      "precision": holdout_eval["precision"],
      "recall": holdout_eval["recall"],
      "confusion_matrix": holdout_eval["confusion_matrix"],
      "classification_report": holdout_eval["report"],
    },
    "backtest_summary": backtest_report,
    "drift_report": drift_report,
  }
  args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
  with args.metrics_output.open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
  print(f"Saved metrics to: {args.metrics_output}")

  print(f"Saved backtest report to: {args.backtest_output}")
  print(f"Saved drift report to: {args.drift_output}")
  print(f"Saved confidence + SHAP log to: {args.prediction_log_output}")


if __name__ == "__main__":
  main()
