# Proyecto Final: Corrección Atmosférica GaoFen-1

Este directorio contiene el notebook marimo para corrección atmosférica con 6S:

- `main.py`

## Requisitos de Earth Engine

El notebook lee `EE_PROJECT` desde `técnicas-de-teledetección/.env`.
Si no existe, muestra un formulario para guardar ese valor automáticamente.

## Archivos de entrada esperados

- `GF1D_PMS_W73.1_S3.9_20200914_L1A1256801630-MUX.xml`
- `ORTO_GF1D_PMS_W73.1_S3.9_20200914_L1A1256801630.tif`

El notebook detecta estos archivos automáticamente en este mismo directorio.

## Dependencias locales del flujo

- `atmosfera.py`
- `wavelengths.py`

## Salidas

El notebook escribe productos de salida en `public/` dentro de este directorio.
