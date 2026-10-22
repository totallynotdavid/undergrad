"""Descarga del inventario de peligros geológicos de INGEMMET.

El servidor solo envía su certificado y no el intermedio de Sectigo, así que
la verificación TLS falla con las CA por defecto. `sectigo_ov_r36.pem` es ese
intermedio, publicado por Sectigo en la URL AIA del certificado del servidor.
"""

import datetime as dt
import json
import ssl
from collections.abc import Iterator
from pathlib import Path

import certifi
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LAYER_URL = (
    "https://geocatmin.ingemmet.gob.pe/arcgis/rest/services/"
    "SERV_PELIGROS_GEOLOGICOS/MapServer/0"
)
DEFAULT_WHERE = "NM_DPTO='Ancash'"
INTERMEDIATE_CA = Path(__file__).with_name("sectigo_ov_r36.pem")

# La capa admite hasta 2000 registros por consulta, pero no admite
# `resultOffset`. Las consultas usan lotes de `OBJECTID`.
PAGE_SIZE = 500
TIMEOUT = (10, 60)
RETRIES = 5


class _TlsAdapter(HTTPAdapter):
    def __init__(self, context: ssl.SSLContext, **kwargs):
        self._context = context
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, ssl_context=self._context, **kwargs)


def make_session(ca_file: Path | None = INTERMEDIATE_CA) -> requests.Session:
    """Sesión que verifica TLS con certifi más el intermedio, con reintentos."""
    context = ssl.create_default_context(cafile=certifi.where())
    if ca_file is not None:
        context.load_verify_locations(cafile=str(ca_file))
    retry = Retry(
        total=RETRIES,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", _TlsAdapter(context, max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def get_json(session: requests.Session, url: str, params: dict) -> dict:
    """GET con tiempo límite. ArcGIS responde los errores con HTTP 200."""
    response = session.get(url, params={**params, "f": "json"}, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"El servicio respondió {payload['error']}")
    return payload


def fetch_layer(session: requests.Session, layer_url: str = LAYER_URL) -> dict:
    return get_json(session, layer_url, {})


def object_ids(
    session: requests.Session, where: str, layer_url: str = LAYER_URL
) -> list[int]:
    payload = get_json(
        session, f"{layer_url}/query", {"where": where, "returnIdsOnly": "true"}
    )
    return sorted(payload["objectIds"] or [])


def iter_features(
    session: requests.Session,
    ids: list[int],
    layer_url: str = LAYER_URL,
    batch_size: int = PAGE_SIZE,
) -> Iterator[dict]:
    """Pide los puntos por lotes de OBJECTID, con coordenadas WGS84."""
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        payload = get_json(
            session,
            f"{layer_url}/query",
            {
                "objectIds": ",".join(map(str, batch)),
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": 4326,
            },
        )
        if payload.get("exceededTransferLimit"):
            raise RuntimeError("El lote superó maxRecordCount; reduzca batch_size")
        yield from payload["features"]


def domains(layer: dict) -> dict[str, dict[str, str]]:
    """Tablas código a nombre de los campos con dominio de valores codificados."""
    tables = {}
    for field in layer["fields"]:
        domain = field.get("domain")
        if domain and domain.get("type") == "codedValue":
            tables[field["name"]] = {
                str(item["code"]): item["name"] for item in domain["codedValues"]
            }
    return tables


def _iso_date(milliseconds: int | None) -> str | None:
    if milliseconds is None:
        return None
    moment = dt.datetime.fromtimestamp(milliseconds / 1000, dt.UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def to_geojson(features: list[dict], layer: dict) -> dict:
    """GeoJSON con todos los atributos. Los campos de fecha pasan a ISO 8601."""
    date_fields = {
        f["name"] for f in layer["fields"] if f["type"] == "esriFieldTypeDate"
    }
    out = []
    for feature in features:
        attributes = dict(feature["attributes"])
        for name in date_fields:
            attributes[name] = _iso_date(attributes.get(name))
        geometry = feature.get("geometry")
        out.append(
            {
                "type": "Feature",
                "geometry": (
                    {"type": "Point", "coordinates": [geometry["x"], geometry["y"]]}
                    if geometry
                    else None
                ),
                "properties": attributes,
            }
        )
    return {"type": "FeatureCollection", "features": out}


NOTA = """\
# Peligros geológicos de Áncash

Los datos son del Instituto Geológico, Minero y Metalúrgico (INGEMMET),
inventario nacional de peligros geológicos.

- Fuente:
  {layer_url}
- Capa: Peligros Geológicos (capa 0 del servicio SERV_PELIGROS_GEOLOGICOS)
- Consulta: `where={where}`, `outFields=*`, `outSR=4326`
- Registros: {rows}
- Fecha de descarga: {taken}
- Archivos: `peligros_ancash.geojson` (puntos en WGS84 con todos los atributos
  de la capa) y `dominios.json` (código a nombre de cada campo codificado).

Para volver a descargar:
`uv run --package fundamentos-de-riesgos-y-desastres python -m ingemmet_peligros.inventario`.
"""


def download(
    where: str = DEFAULT_WHERE,
    session: requests.Session | None = None,
    layer_url: str = LAYER_URL,
    batch_size: int = PAGE_SIZE,
) -> tuple[dict, dict[str, dict[str, str]]]:
    """Puntos como GeoJSON y tablas de dominios, leídos del servicio."""
    session = session or make_session()
    layer = fetch_layer(session, layer_url)
    ids = object_ids(session, where, layer_url)
    features = list(iter_features(session, ids, layer_url, batch_size))
    if sorted(f["attributes"]["OBJECTID"] for f in features) != ids:
        raise RuntimeError(
            f"El servicio listó {len(ids)} registros y entregó {len(features)}"
        )
    return to_geojson(features, layer), domains(layer)


def write_snapshot(
    out_dir: Path,
    where: str = DEFAULT_WHERE,
    session: requests.Session | None = None,
    layer_url: str = LAYER_URL,
) -> int:
    """Descarga los puntos y escribe GeoJSON, dominios y nota de datos."""
    geojson, tables = download(where, session, layer_url)
    rows = len(geojson["features"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "peligros_ancash.geojson").write_text(
        json.dumps(geojson, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (out_dir / "dominios.json").write_text(
        json.dumps(tables, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "NOTA.md").write_text(
        NOTA.format(
            layer_url=layer_url,
            where=where,
            rows=rows,
            taken=dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
        ),
        encoding="utf-8",
    )
    return rows


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "peligros-geologicos" / "datos"
    print(f"{write_snapshot(target)} registros en {target}")
