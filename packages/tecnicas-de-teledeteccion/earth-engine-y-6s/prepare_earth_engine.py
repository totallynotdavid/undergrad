"""Query the Peru Earth Engine examples and write their offline fixture."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import ee

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from earth_engine_support import initialize_earth_engine  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent / "peru_earth_engine.json"

ELEVATION_DATASET = "USGS/SRTMGL1_003"
LAND_COVER_DATASET = "MODIS/061/MCD12Q1"
LAND_SURFACE_TEMPERATURE_DATASET = "MODIS/061/MOD11A1"
LAND_COVER_DATE = "2024-01-01"
LST_YEAR = 2025
PLACES = (
    {
        "id": "lima",
        "name": "Lima",
        "description": "costa del Pacifico",
        "longitude": -77.0428,
        "latitude": -12.0464,
    },
    {
        "id": "cusco",
        "name": "Cusco",
        "description": "Andes altos",
        "longitude": -71.9675,
        "latitude": -13.5319,
    },
    {
        "id": "iquitos",
        "name": "Iquitos",
        "description": "Amazonia",
        "longitude": -73.2538,
        "latitude": -3.7437,
    },
)

LAND_COVER_LABELS = {
    0: "cuerpos de agua",
    1: "bosque siempreverde de coniferas",
    2: "bosque siempreverde latifoliado",
    3: "bosque deciduo de coniferas",
    4: "bosque deciduo latifoliado",
    5: "bosque mixto",
    6: "matorral cerrado",
    7: "matorral abierto",
    8: "sabana arbolada",
    9: "sabana",
    10: "pastizal",
    11: "humedal permanente",
    12: "cultivos",
    13: "urbano construido",
    14: "mosaico cultivo-vegetacion natural",
    15: "nieve y hielo",
    16: "suelo desnudo",
    17: "arbustos y pastizal",
}


def point_record(place: dict[str, Any]) -> tuple[ee.Geometry, dict[str, Any]]:
    coordinates = [place["longitude"], place["latitude"]]
    return ee.Geometry.Point(coordinates), {
        "place_id": place["id"],
        "place": place["name"],
        "longitude": place["longitude"],
        "latitude": place["latitude"],
    }


def mask_lst_image(image: ee.Image) -> ee.Image:
    quality = image.select("QC_Day")
    good_quality = (
        quality.bitwiseAnd(3).eq(0).And(quality.rightShift(2).bitwiseAnd(1).eq(0))
    )
    return (
        image.select("LST_Day_1km")
        .updateMask(good_quality)
        .multiply(0.02)
        .subtract(273.15)
    )


def query_live_data() -> dict[str, Any]:
    """Run the three lesson queries once and return JSON-serializable records."""

    initialize_earth_engine()
    elevation_image = ee.Image(ELEVATION_DATASET)
    land_cover_image = ee.Image(
        f"{LAND_COVER_DATASET}/{LAND_COVER_DATE[:4]}_01_01"
    ).select("LC_Type1")
    lst_collection = ee.ImageCollection(LAND_SURFACE_TEMPERATURE_DATASET)

    elevation: list[dict[str, Any]] = []
    land_cover: list[dict[str, Any]] = []
    land_surface_temperature: list[dict[str, Any]] = []
    for place in PLACES:
        point, record = point_record(place)
        elevation_value = (
            elevation_image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=30,
                maxPixels=10_000,
            )
            .getInfo()
            .get("elevation")
        )
        elevation.append(
            {
                **record,
                "dataset_id": ELEVATION_DATASET,
                "date": "2000-02-11 (SRTM mission)",
                "elevation_m": elevation_value,
            }
        )

        land_cover_value = (
            land_cover_image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=500,
                maxPixels=10_000,
            )
            .getInfo()
            .get("LC_Type1")
        )
        land_cover_code = (
            int(land_cover_value) if land_cover_value is not None else None
        )
        land_cover.append(
            {
                **record,
                "dataset_id": LAND_COVER_DATASET,
                "date": LAND_COVER_DATE,
                "class_code": land_cover_code,
                "class_name": LAND_COVER_LABELS.get(land_cover_code, "sin datos"),
            }
        )

        for month in range(1, 13):
            start = date(LST_YEAR, month, 1)
            end = date(
                LST_YEAR + (month == 12),
                1 if month == 12 else month + 1,
                1,
            )
            monthly = (
                lst_collection.filterDate(str(start), str(end))
                .filterBounds(point)
                .map(mask_lst_image)
                .median()
                .rename("lst_celsius")
            )
            statistics = monthly.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.count(), sharedInputs=True
                ),
                geometry=point.buffer(5_000),
                scale=1_000,
                maxPixels=100_000,
            ).getInfo()
            land_surface_temperature.append(
                {
                    **record,
                    "dataset_id": LAND_SURFACE_TEMPERATURE_DATASET,
                    "date": str(start),
                    "date_end": str(end),
                    "qc_mask": "QC_Day bits 0-1 == 00 and bit 2 == 0",
                    "lst_celsius": statistics.get("lst_celsius_mean"),
                    "valid_pixels": statistics.get("lst_celsius_count", 0),
                }
            )

    return {
        "generated_on": date.today().isoformat(),
        "description": "Peru Earth Engine lesson fixture; all coordinates are lon, lat.",
        "datasets": {
            "elevation": {"id": ELEVATION_DATASET, "date": "SRTM mission"},
            "land_cover": {"id": LAND_COVER_DATASET, "date": LAND_COVER_DATE},
            "land_surface_temperature": {
                "id": LAND_SURFACE_TEMPERATURE_DATASET,
                "dates": f"{LST_YEAR}-01-01/{LST_YEAR + 1}-01-01",
                "scale_factor": 0.02,
                "offset_kelvin": -273.15,
            },
        },
        "places": list(PLACES),
        "elevation": elevation,
        "land_cover": land_cover,
        "land_surface_temperature": land_surface_temperature,
    }


def write_fixture(output_path: Path = OUTPUT_PATH) -> None:
    output_path.write_text(
        json.dumps(query_live_data(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="fixture path (defaults to peru_earth_engine.json)",
    )
    args = parser.parse_args()
    write_fixture(args.output)
