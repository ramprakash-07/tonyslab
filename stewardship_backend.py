import __main__
import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import joblib
import numpy as np
import pandas as pd
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sentinelhub import SHConfig
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


LOGGER = logging.getLogger("stewardship_backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MODEL_PATH = os.getenv("MODEL_PATH", "stewardship_intrusion_model_tn.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "stewardship_scaler.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "stewardship-v1")
RISK_THRESHOLD_LOW = float(os.getenv("RISK_THRESHOLD_LOW", "0.33"))
RISK_THRESHOLD_HIGH = float(os.getenv("RISK_THRESHOLD_HIGH", "0.66"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))
CACHE_MAXSIZE = int(os.getenv("CACHE_MAXSIZE", "2000"))

EVALSCRIPT_NDVI = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "SCL", "dataMask"] }],
    output: [
      { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function evaluatePixel(sample) {
  const cloudClasses = [3, 8, 9, 10];
  if (cloudClasses.includes(sample.SCL)) {
    return { ndvi: [0], dataMask: [0] };
  }
  const denominator = sample.B08 + sample.B04;
  const ndvi = denominator === 0 ? 0 : (sample.B08 - sample.B04) / denominator;
  return { ndvi: [ndvi], dataMask: [sample.dataMask] };
}
""".strip()


class PredictRequest(BaseModel):
  location_id: str = Field(default="unknown-site", min_length=1, max_length=128)
  latitude: float = Field(..., ge=-90.0, le=90.0)
  longitude: float = Field(..., ge=-180.0, le=180.0)


class PredictResponse(BaseModel):
  request_id: str
  model_version: str
  location_id: str
  timestamp_utc: str
  raw_prediction: float
  risk_score: float
  risk_level: str
  stewardship_message: str
  primary_ecological_driver: str
  top_3_drivers: list[str]
  diagnostics: dict[str, float | str | None]


def _register_pickle_compat() -> None:
  try:
    from train_intrusion_model import DropAllNaNColumns, EcologicalFeatureBuilder

    setattr(__main__, "EcologicalFeatureBuilder", EcologicalFeatureBuilder)
    setattr(__main__, "DropAllNaNColumns", DropAllNaNColumns)
  except Exception as exc:
    LOGGER.warning("Pickle compatibility registration skipped: %s", exc)


def _extract_model(loaded: Any) -> Any:
  if isinstance(loaded, dict) and "model" in loaded:
    return loaded["model"]
  return loaded


def _load_shap_explainer(model: Any) -> Any | None:
  try:
    import shap
  except Exception:
    LOGGER.warning("SHAP not installed. Driver explainability will use fallback heuristics.")
    return None

  base_model = model
  if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
    try:
      base_model = model.calibrated_classifiers_[0].estimator
    except Exception:
      base_model = model
  try:
    return shap.Explainer(base_model)
  except Exception as exc:
    LOGGER.warning("SHAP explainer initialization failed: %s", exc)
    return None


def _build_sh_config() -> SHConfig:
  config = SHConfig()
  if not config.sh_client_id or not config.sh_client_secret:
    raise RuntimeError("Missing SH_CLIENT_ID/SH_CLIENT_SECRET.")
  return config


@asynccontextmanager
async def lifespan(app: FastAPI):
  _register_pickle_compat()
  if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

  loaded_model = await asyncio.to_thread(joblib.load, MODEL_PATH)
  app.state.model = _extract_model(loaded_model)
  app.state.scaler = await asyncio.to_thread(joblib.load, SCALER_PATH) if os.path.exists(SCALER_PATH) else None
  app.state.sh_config = _build_sh_config()
  app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
  app.state.ndvi_cache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS)
  app.state.shap_cache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL_SECONDS)
  app.state.shap_explainer = await asyncio.to_thread(_load_shap_explainer, app.state.model)
  app.state.oauth_token = {"token": None, "expires_at": 0.0}
  LOGGER.info("Startup complete: model warm, Sentinel session ready, cache initialized.")
  try:
    yield
  finally:
    await app.state.http_client.aclose()


app = FastAPI(title="Stewardship Ecological Backend", version="2.0.0", lifespan=lifespan)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def request_metrics_middleware(request: Request, call_next):
  request_id = str(uuid.uuid4())
  request.state.request_id = request_id
  start = time.perf_counter()
  response = await call_next(request)
  elapsed_ms = (time.perf_counter() - start) * 1000
  LOGGER.info(
    "request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
    request_id,
    request.method,
    request.url.path,
    response.status_code,
    elapsed_ms,
  )
  response.headers["X-Request-ID"] = request_id
  return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
  return JSONResponse(
    status_code=exc.status_code,
    content={
      "request_id": getattr(request.state, "request_id", "unknown"),
      "model_version": MODEL_VERSION,
      "detail": exc.detail,
    },
  )


@retry(
  retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException, httpx.HTTPStatusError)),
  wait=wait_exponential(multiplier=1, min=1, max=8),
  stop=stop_after_attempt(4),
  reraise=True,
)
async def _post_with_retry(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
  response = await client.post(url, **kwargs)
  response.raise_for_status()
  return response


async def _get_access_token(request: Request) -> str:
  token_data = request.app.state.oauth_token
  now_ts = time.time()
  if token_data["token"] and token_data["expires_at"] > now_ts:
    return token_data["token"]

  cfg: SHConfig = request.app.state.sh_config
  url = f"{cfg.sh_base_url.rstrip('/')}/oauth/token"
  payload = {
    "grant_type": "client_credentials",
    "client_id": cfg.sh_client_id,
    "client_secret": cfg.sh_client_secret,
  }

  try:
    response = await _post_with_retry(request.app.state.http_client, url, data=payload)
    body = response.json()
  except Exception as exc:
    raise HTTPException(status_code=503, detail=f"Sentinel auth failed: {exc}") from exc

  access_token = body.get("access_token")
  expires_in = int(body.get("expires_in", 3600))
  if not access_token:
    raise HTTPException(status_code=503, detail="Sentinel auth token missing in response.")

  token_data["token"] = access_token
  token_data["expires_at"] = now_ts + max(60, expires_in - 60)
  return access_token


def _point_buffer_geometry(lat: float, lon: float, delta: float = 0.0015) -> dict[str, Any]:
  return {
    "type": "Polygon",
    "coordinates": [[
      [lon - delta, lat - delta],
      [lon + delta, lat - delta],
      [lon + delta, lat + delta],
      [lon - delta, lat + delta],
      [lon - delta, lat - delta],
    ]],
  }


async def _fetch_ndvi_last_15_days(request: Request, lat: float, lon: float) -> pd.DataFrame:
  key = f"{round(lat, 5)}:{round(lon, 5)}"
  cached = request.app.state.ndvi_cache.get(key)
  if cached is not None:
    return cached.copy()

  end_dt = datetime.now(timezone.utc)
  start_dt = end_dt - timedelta(days=15)
  token = await _get_access_token(request)
  cfg: SHConfig = request.app.state.sh_config
  url = f"{cfg.sh_base_url.rstrip('/')}/api/v1/statistics"

  payload = {
    "input": {
      "bounds": {"geometry": _point_buffer_geometry(lat, lon)},
      "data": [{"type": "sentinel-2-l2a"}],
    },
    "aggregation": {
      "timeRange": {
        "from": start_dt.strftime("%Y-%m-%dT00:00:00Z"),
        "to": end_dt.strftime("%Y-%m-%dT00:00:00Z"),
      },
      "aggregationInterval": {"of": "P1D"},
      "evalscript": EVALSCRIPT_NDVI,
      "resx": 20,
      "resy": 20,
    },
    "calculations": {"ndvi": {"statistics": {"default": {}}}},
  }
  headers = {"Authorization": f"Bearer {token}"}

  try:
    response = await _post_with_retry(request.app.state.http_client, url, json=payload, headers=headers)
    result = response.json()
  except Exception as exc:
    raise HTTPException(status_code=503, detail=f"Sentinel fetch failed: {exc}") from exc

  rows = []
  for item in result.get("data", []):
    stats = item.get("outputs", {}).get("ndvi", {}).get("bands", {}).get("B0", {}).get("stats", {})
    mean_ndvi = stats.get("mean")
    if mean_ndvi is None:
      continue
    sample_count = stats.get("sampleCount")
    no_data_count = stats.get("noDataCount")
    no_data_ratio = np.nan
    if sample_count and no_data_count is not None:
      no_data_ratio = float(no_data_count) / float(sample_count)
    rows.append(
      {
        "Timestamp": pd.to_datetime(item.get("interval", {}).get("to"), utc=True, errors="coerce"),
        "NDVI": float(mean_ndvi),
        "NoData_Ratio": float(no_data_ratio) if pd.notna(no_data_ratio) else np.nan,
      }
    )

  ndvi_df = pd.DataFrame(rows).dropna(subset=["Timestamp"]).sort_values("Timestamp")
  if ndvi_df.empty:
    raise HTTPException(
      status_code=503,
      detail="No usable NDVI data returned. Likely cloud/no-data coverage.",
    )

  # Reindex to full daily series to keep lag features stable during cloudy/no-data gaps.
  full_index = pd.date_range(
    start=start_dt.replace(hour=0, minute=0, second=0, microsecond=0),
    end=end_dt.replace(hour=0, minute=0, second=0, microsecond=0),
    freq="D",
    tz="UTC",
  )
  ndvi_df = ndvi_df.set_index("Timestamp").reindex(full_index)
  ndvi_df.index.name = "Timestamp"
  ndvi_df["NDVI"] = ndvi_df["NDVI"].interpolate(method="linear", limit_direction="both")
  ndvi_df["NoData_Ratio"] = ndvi_df["NoData_Ratio"].fillna(1.0)
  ndvi_df = ndvi_df.reset_index()

  request.app.state.ndvi_cache[key] = ndvi_df.copy()
  return ndvi_df


def _compute_features(ndvi_df: pd.DataFrame) -> dict[str, Any]:
  series = ndvi_df["NDVI"].astype(float).reset_index(drop=True)
  if len(series) < 11:
    raise HTTPException(status_code=503, detail="Insufficient NDVI history for lag features.")

  latest_ndvi = float(series.iloc[-1])
  ndvi_mean = float(series.mean())
  ndvi_std = float(series.std()) if float(series.std()) > 0 else 1e-6
  vsd = float((latest_ndvi - ndvi_mean) / ndvi_std)
  vsg = float(series.iloc[-1] - series.iloc[-2])
  lag_5 = float(series.iloc[-6])
  lag_10 = float(series.iloc[-11])

  return {
    "Timestamp": pd.to_datetime(ndvi_df["Timestamp"].iloc[-1], utc=True).isoformat(),
    "NDVI": latest_ndvi,
    "NDVI_Std": ndvi_std,
    "VSD": vsd,
    "VSG": vsg,
    "Coupling_Index": float(vsd * vsg),
    "Rainfall_Proxy": float(ndvi_df["NoData_Ratio"].iloc[-1]) if "NoData_Ratio" in ndvi_df.columns else np.nan,
    "NDVI_Lag_5": lag_5,
    "NDVI_Lag_10": lag_10,
  }


def _apply_scaler(request: Request, frame: pd.DataFrame) -> pd.DataFrame:
  scaler = request.app.state.scaler
  if scaler is None:
    return frame
  try:
    scaled = scaler.transform(frame.values)
    return pd.DataFrame(scaled, columns=frame.columns, index=frame.index)
  except Exception:
    return frame


def _heuristic_top_drivers(frame: pd.DataFrame) -> list[str]:
  candidates = [c for c in ["VSD", "VSG", "Coupling_Index", "NDVI_Lag_5", "NDVI_Lag_10"] if c in frame.columns]
  if not candidates:
    return ["Unknown", "Unknown", "Unknown"]
  values = frame[candidates].iloc[0].astype(float).abs().values
  order = np.argsort(values)[::-1][:3]
  chosen = [candidates[i] for i in order]
  while len(chosen) < 3:
    chosen.append("Unknown")
  return chosen


def _top_3_drivers(request: Request, frame: pd.DataFrame, risk_level: str) -> list[str]:
  if risk_level != "High":
    return _heuristic_top_drivers(frame)

  key = tuple(np.round(frame.iloc[0].fillna(0).astype(float).values, 6).tolist())
  cached = request.app.state.shap_cache.get(key)
  if cached is not None:
    return cached

  explainer = request.app.state.shap_explainer
  if explainer is None:
    drivers = _heuristic_top_drivers(frame)
    request.app.state.shap_cache[key] = drivers
    return drivers

  try:
    shap_values = explainer(frame)
    values = np.array(shap_values.values)[0]
    names = shap_values.feature_names or frame.columns.tolist()
    order = np.argsort(np.abs(values))[::-1][:3]
    drivers = [str(names[i]) for i in order]
  except Exception:
    drivers = _heuristic_top_drivers(frame)

  request.app.state.shap_cache[key] = drivers
  return drivers


def _risk_level(score: float) -> str:
  if score < RISK_THRESHOLD_LOW:
    return "Low"
  if score < RISK_THRESHOLD_HIGH:
    return "Medium"
  return "High"


def _stewardship_message(level: str, primary_driver: str) -> str:
  if level == "High":
    return f"High risk detected. Prioritize immediate field patrols. Primary driver: {primary_driver}."
  if level == "Medium":
    return f"Moderate risk. Increase monitoring frequency. Watch: {primary_driver}."
  return f"Low risk. Maintain routine stewardship checks. Current signal: {primary_driver}."


async def _predict_impl(request: Request, payload: PredictRequest) -> PredictResponse:
  if not hasattr(request.app.state, "model") or request.app.state.model is None:
    raise HTTPException(status_code=503, detail="Model unavailable.")

  ndvi_df = await _fetch_ndvi_last_15_days(request, payload.latitude, payload.longitude)
  features = _compute_features(ndvi_df)
  frame = pd.DataFrame([features])
  model_frame = _apply_scaler(request, frame)

  model = request.app.state.model
  try:
    if hasattr(model, "predict_proba"):
      score = float(model.predict_proba(model_frame)[0][1])
    else:
      score = float(model.predict(model_frame)[0])
  except Exception as exc:
    raise HTTPException(status_code=503, detail=f"Inference failed: {exc}") from exc

  level = _risk_level(score)
  top_drivers = _top_3_drivers(request, model_frame, level)
  primary = top_drivers[0] if top_drivers else "Unknown"

  return PredictResponse(
    request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
    model_version=MODEL_VERSION,
    location_id=payload.location_id,
    timestamp_utc=str(features["Timestamp"]),
    raw_prediction=round(score, 4),
    risk_score=round(score, 4),
    risk_level=level,
    stewardship_message=_stewardship_message(level, primary),
    primary_ecological_driver=primary,
    top_3_drivers=top_drivers,
    diagnostics={
      "ndvi": round(float(features["NDVI"]), 4),
      "vsd": round(float(features["VSD"]), 4),
      "vsg": round(float(features["VSG"]), 4),
      "coupling_index": round(float(features["Coupling_Index"]), 4),
      "ndvi_lag_5": round(float(features["NDVI_Lag_5"]), 4),
      "ndvi_lag_10": round(float(features["NDVI_Lag_10"]), 4),
      "rainfall_proxy": (
        round(float(features["Rainfall_Proxy"]), 4) if pd.notna(features["Rainfall_Proxy"]) else None
      ),
    },
  )


@app.get("/health")
async def health(request: Request):
  model_loaded = hasattr(request.app.state, "model") and request.app.state.model is not None
  sentinel_reachable = False
  sentinel_detail = None
  if model_loaded:
    try:
      await _get_access_token(request)
      sentinel_reachable = True
    except HTTPException as exc:
      sentinel_detail = str(exc.detail)

  healthy = bool(model_loaded and sentinel_reachable)
  code = 200 if healthy else 503
  return JSONResponse(
    status_code=code,
    content={
      "request_id": getattr(request.state, "request_id", str(uuid.uuid4())),
      "model_version": MODEL_VERSION,
      "healthy": healthy,
      "model_loaded": model_loaded,
      "sentinel_reachable": sentinel_reachable,
      "sentinel_detail": sentinel_detail,
    },
  )


@app.post("/predict")
@limiter.limit("30/minute")
async def predict(request: Request, payload: PredictRequest):
  result = await _predict_impl(request, payload)
  return JSONResponse(status_code=200, content=result.model_dump())


@app.post("/predict/stewardship")
@limiter.limit("30/minute")
async def predict_stewardship(request: Request, payload: PredictRequest):
  result = await _predict_impl(request, payload)
  return JSONResponse(status_code=200, content=result.model_dump())
