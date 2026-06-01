# Física y Técnicas de Teledetección

Este módulo usa notebooks en marimo dentro del workspace con `uv`.

## Ejecución reproducible

1. Crear entorno y dependencias del workspace:
   `uv sync`
2. Configurar Earth Engine la primera vez:
   - Abre un notebook (`Clase_1.py` o `correccion-atmosferica-gaofen1/main.py`)
   - Ingresa `EE_PROJECT` en el formulario y pulsa `Guardar en .env`
3. Autenticar Earth Engine (una sola vez por máquina/usuario):
   `uv run --package tecnicas-de-teledeteccion earthengine authenticate`

## Abrir notebooks

- `uv run --package tecnicas-de-teledeteccion marimo edit técnicas-de-teledetección/Clase_0.py`
- `uv run --package tecnicas-de-teledeteccion marimo edit técnicas-de-teledetección/Clase_1.py`
- `uv run --package tecnicas-de-teledeteccion marimo edit técnicas-de-teledetección/correccion-atmosferica-gaofen1/main.py`
