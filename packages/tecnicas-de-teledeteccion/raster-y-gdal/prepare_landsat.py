"""Download and compress the small Landsat lesson fixture from Earth Engine."""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import ee
import rasterio

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from earth_engine_support import initialize_earth_engine  # noqa: E402

OUTPUT_PATH = (
    Path(__file__).resolve().parent / "landsat_scene_007068_20240427_reflectance.tif"
)
COLLECTIONS = (
    ("LANDSAT/LC08/C02/T1", "Collection 2 Tier 1"),
    ("LANDSAT/LC08/C02/T1_RT", "Collection 2 real-time tier"),
)
TARGET_PRODUCT_PREFIX = "LC08_L1TP_007068_20240427"
REFLECTANCE_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
CRS = "EPSG:32618"
SCALE_METRES = 30
DOWNLOAD_TIMEOUT_SECONDS = 120
MAX_RETRIES = 4

# Keep the clip window explicit so the lesson can connect downloaded pixels
# to the Earth Engine query.
CLIP_WINDOW_LON_LAT = (
    (-76.40, -11.62),
    (-76.40, -11.52),
    (-76.28, -11.52),
    (-76.28, -11.62),
    (-76.40, -11.62),
)


def download_with_retry(url: str, output_path: Path) -> None:
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(
                url, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                output_path.write_bytes(response.read())
            return
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2**attempt)


def find_scene() -> tuple[ee.Image, str, str]:
    for collection_id, tier in COLLECTIONS:
        collection = (
            ee.ImageCollection(collection_id)
            .filterDate("2024-04-27", "2024-04-28")
            .filter(
                ee.Filter.stringStartsWith("LANDSAT_PRODUCT_ID", TARGET_PRODUCT_PREFIX)
            )
        )
        count = collection.size().getInfo()
        if count == 1:
            return ee.Image(collection.first()), collection_id, tier
        if count > 1:
            raise RuntimeError(
                f"The requested Landsat scene was not unique in {collection_id}"
            )
    raise RuntimeError("The requested Landsat scene was not found in Tier 1 or T1_RT")


def write_fixture(output_path: Path = OUTPUT_PATH) -> None:
    initialize_earth_engine()
    image, collection_id, tier = find_scene()
    image_info = image.toDictionary(["LANDSAT_PRODUCT_ID", "system:index"]).getInfo()
    source_path = output_path.with_suffix(".download.tif")
    url = image.select(REFLECTANCE_BANDS).getDownloadURL(
        {
            "name": output_path.stem,
            "bands": REFLECTANCE_BANDS,
            "region": [list(coordinate) for coordinate in CLIP_WINDOW_LON_LAT],
            "scale": SCALE_METRES,
            "crs": CRS,
            "format": "GEO_TIFF",
            "filePerBand": False,
        }
    )
    download_with_retry(url, source_path)

    try:
        with rasterio.open(source_path) as source:
            profile = source.profile.copy()
            profile.update(
                compress="DEFLATE",
                driver="GTiff",
                nodata=0,
                predictor=2,
                tiled=True,
                zlevel=9,
            )
            with rasterio.open(output_path, "w", **profile) as translated:
                for index in range(1, source.count + 1):
                    translated.write(source.read(index), index)
                translated.descriptions = tuple(REFLECTANCE_BANDS)
                translated.update_tags(
                    EARTH_ENGINE_COLLECTION=collection_id,
                    EARTH_ENGINE_TIER=tier,
                    LANDSAT_PRODUCT_ID=image_info["LANDSAT_PRODUCT_ID"],
                )
    finally:
        source_path.unlink(missing_ok=True)

    print(f"Scene: {image_info['system:index']} ({image_info['LANDSAT_PRODUCT_ID']})")
    print(f"Collection: {collection_id} ({tier})")
    print(f"Clip window: {CLIP_WINDOW_LON_LAT}")
    print(f"Wrote {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    write_fixture(args.output)
