import argparse
import json
from pathlib import Path

from sentinelhub import CRS, DataCollection, Geometry, SentinelHubStatistical, SHConfig

EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B04", "B08", "SCL", "dataMask"]
    }],
    output: [
      { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function evaluatePixel(sample) {
  // Cloud and cloud-shadow classes in Sentinel-2 L2A SCL
  const cloudClasses = [3, 8, 9, 10];

  if (cloudClasses.includes(sample.SCL)) {
    return {
      ndvi: [0],
      dataMask: [0]
    };
  }

  const denominator = sample.B08 + sample.B04;
  const ndvi = denominator === 0 ? 0 : (sample.B08 - sample.B04) / denominator;

  return {
    ndvi: [ndvi],
    dataMask: [sample.dataMask]
  };
}
""".strip()


def load_first_polygon(geojson_path: Path) -> dict:
  with geojson_path.open("r", encoding="utf-8-sig") as f:
    geojson = json.load(f)

  if geojson.get("type") == "FeatureCollection":
    features = geojson.get("features", [])
    if not features:
      raise ValueError("GeoJSON FeatureCollection is empty.")
    geometry = features[0].get("geometry")
  elif geojson.get("type") == "Feature":
    geometry = geojson.get("geometry")
  else:
    geometry = geojson

  if not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
    raise ValueError("GeoJSON must contain a Polygon or MultiPolygon geometry.")

  return geometry


def build_request(geometry: Geometry, config: SHConfig) -> SentinelHubStatistical:
  aggregation = SentinelHubStatistical.aggregation(
    evalscript=EVALSCRIPT,
    time_interval=("2023-01-01", "2026-01-01"),
    aggregation_interval="P30D",
    size=(1000, 1000),
  )

  input_data = [
    SentinelHubStatistical.input_data(
      data_collection=DataCollection.SENTINEL2_L2A,
    )
  ]

  calculations = {
    "ndvi": {
      "statistics": {
        "default": {}
      }
    }
  }

  return SentinelHubStatistical(
    aggregation=aggregation,
    input_data=input_data,
    geometry=geometry,
    calculations=calculations,
    config=config,
  )


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Compute 3-year NDVI historical baseline stats (mean/stDev) using Sentinel Hub Statistical API."
  )
  parser.add_argument(
    "--geojson",
    required=True,
    type=Path,
    help="Path to a GeoJSON file containing your forest-edge polygon.",
  )
  parser.add_argument(
    "--output",
    default=Path("ndvi_baseline_stats.json"),
    type=Path,
    help="Output JSON file path (default: ndvi_baseline_stats.json).",
  )

  args = parser.parse_args()

  polygon = load_first_polygon(args.geojson)
  geometry = Geometry(polygon, CRS.WGS84)

  config = SHConfig()
  if not config.sh_client_id or not config.sh_client_secret:
    raise RuntimeError(
      "Missing Sentinel Hub credentials. Set SH_CLIENT_ID and SH_CLIENT_SECRET env vars "
      "or configure sentinelhub config file."
    )

  request = build_request(geometry, config)
  response = request.get_data()[0]

  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("w", encoding="utf-8") as f:
    json.dump(response, f, indent=2)

  print(f"Saved Statistical API response to: {args.output}")


if __name__ == "__main__":
  main()
