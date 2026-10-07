#!/usr/bin/env bash
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get is required (Debian/Ubuntu)." >&2
  exit 1
fi

GDAL_REQUIREMENT=$(awk -F'"' '$2 ~ /^gdal==/ {sub(/^gdal==/, "", $2); print $2; exit}' \
  packages/tecnicas-de-teledeteccion/pyproject.toml)
if [[ -z "${GDAL_REQUIREMENT}" ]]; then
  echo "Could not find an exact GDAL pin in packages/tecnicas-de-teledeteccion/pyproject.toml." >&2
  exit 1
fi

PACKAGES=(
  gfortran-13
  gdal-bin
  libgdal-dev
)
SUDO=""
if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
fi

$SUDO apt-get update
$SUDO apt-get install -y "${PACKAGES[@]}"

echo "Installed:"
gfortran-13 --version | head -n 1
if ! command -v gdal-config >/dev/null 2>&1; then
  echo "gdal-config is missing after installing libgdal-dev ${GDAL_REQUIREMENT}." >&2
  exit 1
fi
GDAL_NATIVE_VERSION=$(gdal-config --version)
if [[ "${GDAL_NATIVE_VERSION}" != "${GDAL_REQUIREMENT}" ]]; then
  echo "GDAL version mismatch: pyproject.toml pins ${GDAL_REQUIREMENT}, but gdal-config reports ${GDAL_NATIVE_VERSION}." >&2
  echo "Install libgdal-dev ${GDAL_REQUIREMENT} and retry." >&2
  exit 1
fi
echo "GDAL ${GDAL_NATIVE_VERSION}"
