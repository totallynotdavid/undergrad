"""Offline and live data paths for the Gaofen-1 atmospheric correction."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from xml.dom import minidom

import numpy as np
import rasterio
from gf1pms import BANDS
from Py6S import AeroProfile, AtmosProfile, Geometry, SixS, Wavelength

PACKAGE_DIR = Path(__file__).resolve().parent
CLIP_PATH = PACKAGE_DIR / "gaofen1_clip.tif"
INPUTS_PATH = PACKAGE_DIR / "gaofen1_inputs.json"
XML_PATH = PACKAGE_DIR / "GF1D_PMS_W73.1_S3.9_20200914_L1A1256801630-MUX.xml"
SCENE_NAME = "ORTO_GF1D_PMS_W73.1_S3.9_20200914_L1A1256801630.tif"
SIXS_PATH = (
    Path(importlib.util.find_spec("sixs_bin").submodule_search_locations[0])
    / "sixsV1.1"
)
GAIN = np.array([0.1490, 0.1328, 0.1311, 0.1217])
BIAS = np.zeros(4)


def load_inputs(path: Path = INPUTS_PATH) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata(path: Path = XML_PATH) -> dict[str, object]:
    xml = minidom.parse(str(path))

    def value(name: str) -> str:
        return xml.getElementsByTagName(name)[0].firstChild.data.strip()

    return {
        "date": datetime.strptime(value("StartTime"), "%Y-%m-%d %H:%M:%S"),
        "solar_zenith": float(value("SolarZenith")),
        "solar_azimuth": float(value("SolarAzimuth")),
        "view_zenith": float(value("SatelliteZenith")),
        "view_azimuth": float(value("SatelliteAzimuth")),
        "latitude": float(value("CenterLatitude")),
        "longitude": float(value("CenterLongitude")),
    }


def load_clip(path: Path = CLIP_PATH) -> tuple[np.ndarray, dict[str, object]]:
    with rasterio.open(path) as dataset:
        return dataset.read(), dataset.profile.copy()


def _sixs(metadata: dict[str, object], inputs: dict[str, object]) -> SixS:
    values = inputs["values"]
    date = metadata["date"]
    sixs = SixS(path=str(SIXS_PATH))
    sixs.atmos_profile = AtmosProfile.UserWaterAndOzone(
        values["water_vapour_g_cm2"], values["ozone_atm_cm"]
    )
    sixs.aero_profile = AeroProfile.Continental
    sixs.aot550 = values["aerosol_optical_depth_550"]
    sixs.geometry = Geometry.User()
    sixs.geometry.solar_z = metadata["solar_zenith"]
    sixs.geometry.solar_a = metadata["solar_azimuth"]
    sixs.geometry.view_z = metadata["view_zenith"]
    sixs.geometry.view_a = metadata["view_azimuth"]
    sixs.geometry.month = date.month
    sixs.geometry.day = date.day
    sixs.altitudes.set_sensor_satellite_level()
    sixs.altitudes.set_target_custom_altitude(values["elevation_km"])
    return sixs


def run_correction(
    sensor: np.ndarray,
    metadata: dict[str, object],
    inputs: dict[str, object],
) -> np.ndarray:
    result = np.empty(sensor.shape, dtype=np.float64)
    sixs = _sixs(metadata, inputs)
    for index, bandname in enumerate(("B1", "B2", "B3", "B4")):
        sixs.wavelength = Wavelength(*BANDS[bandname])
        sixs.run()
        direct = sixs.outputs.direct_solar_irradiance
        diffuse = sixs.outputs.diffuse_solar_irradiance
        path_radiance = sixs.outputs.atmospheric_intrinsic_radiance
        gas = sixs.outputs.trans["global_gas"].upward
        scattering = sixs.outputs.trans["total_scattering"].upward
        radiance = (sensor[index].astype(np.float64) / GAIN[index]) + BIAS[index]
        reflectance = (
            (radiance - path_radiance) * np.pi / (gas * scattering * (direct + diffuse))
        )
        result[index] = np.where(reflectance <= 0, np.nan, reflectance)
    return result


def live_data() -> tuple[
    np.ndarray, dict[str, object], dict[str, object], dict[str, object]
]:
    import prepare
    from dotenv import load_dotenv

    load_dotenv(PACKAGE_DIR.parent / ".env")
    scene_path = prepare.download_drive_file(
        SCENE_NAME, prepare.DRIVE_FILES[SCENE_NAME]
    )
    date, latitude, longitude = prepare.parse_scene_metadata(
        prepare.download_drive_file(
            prepare.XML_PATH.name, prepare.DRIVE_FILES[prepare.XML_PATH.name]
        )
    )
    inputs = prepare.earth_engine_inputs(date, latitude, longitude)
    with rasterio.open(scene_path) as dataset:
        sensor = dataset.read()
        profile = dataset.profile.copy()
    return sensor, profile, load_metadata(), inputs
