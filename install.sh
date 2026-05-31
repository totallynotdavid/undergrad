#!/usr/bin/env bash
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get is required (Debian/Ubuntu)." >&2
  exit 1
fi

GFORTRAN_MAJOR="${1:-13}"
PACKAGE="gfortran-${GFORTRAN_MAJOR}"
SUDO=""
if [[ "${EUID}" -ne 0 ]]; then
  SUDO="sudo"
fi

$SUDO apt-get update
$SUDO apt-get install -y "${PACKAGE}"

echo "Installed:"
"${PACKAGE}" --version | head -n 1
