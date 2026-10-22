"""Prepare the small, offline Gaofen-1 fixture used by ``main.py``.

The large scene is downloaded to ``.cache/`` unless a verified local data
directory is configured. Only the small clip and the Earth Engine response JSON
belong in the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.dom import minidom

import ee
import numpy as np
import rasterio
from atmosfera import Atmospheric
from correction import (
    SCENE_NAME,
    load_clip,
    load_metadata,
    parse_start_time,
    run_correction,
)
from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
CACHE_DIR = PACKAGE_DIR / ".cache"
CLIP_PATH = PACKAGE_DIR / "gaofen1_clip.tif"
INPUTS_PATH = PACKAGE_DIR / "gaofen1_inputs.json"
XML_PATH = PACKAGE_DIR / "GF1D_PMS_W73.1_S3.9_20200914_L1A1256801630-MUX.xml"
EE_DEADLINE_MS = 120_000
EE_MAX_RETRIES = 3

DRIVE_FILES = {
    "ORTO_GF1D_PMS_W73.1_S3.9_20200914_L1A1256801630.tif": {
        "id": "1_za1jr01lwhGE4aB6nN_I6oUNgxFRE37",
        "bytes": 597_213_549,
        "sha256": "545ecad971e70f1ac7c215b867b6151afdb401679005c3cc20f8f557910cda60",
    },
    "GF1D_PMS_W73.1_S3.9_20200914_L1A1256801630-MUX.xml": {
        "id": "1kyiFVj2ZcfmHw1fyeoaNQgEztgfhUsnl",
        "bytes": 3_368,
        "sha256": "c195bbbb3608947646138d8e213fa8471e22e67a1c3a0c172d06ff9ad0b86541",
    },
}

# These pixel offsets select a representative full-resolution window.
WINDOW = {"row_off": 4_850, "col_off": 4_100, "height": 512, "width": 512}

DATASETS = {
    "elevation": "USGS/SRTMGL1_003:elevation",
    "water_vapour": "ECMWF/ERA5/HOURLY:total_column_water_vapour",
    "ozone": "TOMS/MERGED_V4:ozone",
    "aerosol": "MODIS/061/MOD08_M3:Aerosol_Optical_Depth_Land_Ocean_Mean_Mean",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_drive_file(name: str, metadata: dict[str, object]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = CACHE_DIR / name
    expected_bytes = int(metadata["bytes"])
    expected_sha = str(metadata["sha256"])
    if destination.exists():
        if (
            destination.stat().st_size == expected_bytes
            and sha256(destination) == expected_sha
        ):
            return destination
        destination.unlink()

    url = (
        "https://drive.usercontent.google.com/download?id="
        f"{metadata['id']}&export=download&confirm=t"
    )
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            request = Request(url, headers={"User-Agent": "undergrad-gaofen-prepare"})
            with urlopen(request, timeout=60) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if (
                partial.stat().st_size != expected_bytes
                or sha256(partial) != expected_sha
            ):
                raise RuntimeError(
                    f"SHA-256 mismatch for {name}: expected {expected_sha}, "
                    f"got {sha256(partial)} (size {partial.stat().st_size})"
                )
            partial.replace(destination)
            return destination
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as error:
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {name} after 3 attempts: {last_error}")


def data_file(name: str) -> Path:
    load_dotenv(PACKAGE_DIR.parent / ".env")
    metadata = DRIVE_FILES[name]
    data_dir = os.environ.get("GAOFEN1_DATA_DIR", "").strip()
    if data_dir:
        candidate = Path(data_dir).expanduser() / name
        if (
            candidate.is_file()
            and candidate.stat().st_size == int(metadata["bytes"])
            and sha256(candidate) == str(metadata["sha256"])
        ):
            return candidate
    return download_drive_file(name, metadata)


def parse_scene_metadata(xml_path: Path) -> tuple[datetime, float, float]:
    xml = minidom.parse(str(xml_path))
    date = parse_start_time(xml.getElementsByTagName("StartTime")[0].firstChild.data)
    lat = float(xml.getElementsByTagName("CenterLatitude")[0].firstChild.data)
    lon = float(xml.getElementsByTagName("CenterLongitude")[0].firstChild.data)
    return date, lat, lon


def write_clip(scene_path: Path) -> None:
    with rasterio.open(scene_path) as source:
        window = rasterio.windows.Window(**WINDOW)
        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            height=WINDOW["height"],
            width=WINDOW["width"],
            transform=rasterio.windows.transform(window, source.transform),
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        with rasterio.open(CLIP_PATH, "w", **profile) as destination:
            destination.write(source.read(window=window))


def earth_engine_inputs(
    date: datetime, latitude: float, longitude: float
) -> dict[str, object]:
    load_dotenv(PACKAGE_DIR.parent / ".env")
    project = os.environ.get("EE_PROJECT", "").strip()
    if not project:
        raise RuntimeError(
            "EE_PROJECT must be set in the package .env before preparation"
        )
    ee.Initialize(project=project)
    ee.data.setDeadline(EE_DEADLINE_MS)
    ee.data.setMaxRetries(EE_MAX_RETRIES)
    ee_date = ee.Date(date.isoformat())
    point = ee.Geometry.Point(longitude, latitude)
    elevation = (
        ee.Image("USGS/SRTMGL1_003")
        .reduceRegion(reducer=ee.Reducer.mean(), geometry=point)
        .get("elevation")
        .getInfo()
    )
    water_vapour = Atmospheric.water(point, ee_date).getInfo()
    ozone = Atmospheric.ozone(point, ee_date).getInfo()
    aerosol = Atmospheric.aerosol(point, ee_date).getInfo()
    return {
        "date": date.isoformat(),
        "point": {"latitude": latitude, "longitude": longitude},
        "datasets": DATASETS,
        "values": {
            "elevation_km": float(elevation) / 1000,
            "water_vapour_g_cm2": float(water_vapour),
            "ozone_atm_cm": float(ozone),
            "aerosol_optical_depth_550": float(aerosol),
        },
    }


def prepare() -> None:
    scene_path = data_file(SCENE_NAME)
    downloaded_xml = data_file(XML_PATH.name)
    if downloaded_xml.read_bytes() != XML_PATH.read_bytes():
        print(
            "Supplied XML differs from the committed XML; preserving the committed "
            "XML and using the supplied copy only for validation."
        )
    write_clip(scene_path)
    date, latitude, longitude = parse_scene_metadata(downloaded_xml)
    inputs = earth_engine_inputs(date, latitude, longitude)
    sensor, _ = load_clip()
    corrected = run_correction(sensor, load_metadata(), inputs)
    finite = np.isfinite(corrected)
    reference = {
        "shape": list(corrected.shape),
        "mean": [float(np.nanmean(corrected[index])) for index in range(4)],
        "max": [float(np.nanmax(corrected[index])) for index in range(4)],
        "sample": [float(corrected[index, 200, 200]) for index in range(4)],
        "finite_fraction": [float(finite[index].mean()) for index in range(4)],
    }
    inputs["reference"] = reference
    INPUTS_PATH.write_text(
        json.dumps(inputs, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {CLIP_PATH} ({CLIP_PATH.stat().st_size} bytes)")
    print(f"wrote {INPUTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()


if __name__ == "__main__":
    main()
