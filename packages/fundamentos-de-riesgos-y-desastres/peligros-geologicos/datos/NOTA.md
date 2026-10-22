# Peligros geológicos de Áncash

Los datos son del Instituto Geológico, Minero y Metalúrgico (INGEMMET),
inventario nacional de peligros geológicos.

- Fuente:
  https://geocatmin.ingemmet.gob.pe/arcgis/rest/services/SERV_PELIGROS_GEOLOGICOS/MapServer/0
- Capa: Peligros Geológicos (capa 0 del servicio SERV_PELIGROS_GEOLOGICOS)
- Consulta: `where=NM_DPTO='Ancash'`, `outFields=*`, `outSR=4326`
- Registros: 1648
- Fecha de descarga: 2026-10-22
- Archivos: `peligros_ancash.geojson` (puntos en WGS84 con todos los atributos
  de la capa) y `dominios.json` (código a nombre de cada campo codificado).

Para volver a descargar:
`uv run --package fundamentos-de-riesgos-y-desastres python -m ingemmet_peligros.inventario`.
