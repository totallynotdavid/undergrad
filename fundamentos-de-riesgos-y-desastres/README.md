# Extractor de Datos SIGRID

Este proyecto proporciona una herramienta para extraer datos de SIGRID, la base de datos de CENEPRED para desastres en Perú (específicamente huaicos, otros endpoints no son considerados), y guardarlos en formatos CSV y XLSX.

Nota: ¡Otros endpoints pueden o deberían de ser añadidos en versiones futuras! Acepto PRs si estás dispuesto a colaborar.

## Requisitos

- Python 3.10 o superior (<3.13)
- Poetry

## Instalación

1. Clona este repositorio en tu máquina local.
    ```sh
    git clone https://github.com/dadch1404/sigrid-data-extractor.git
    cd sigrid_data_extractor
    ```
2. Instala las dependencias del proyecto.
    ```sh
    poetry install
    ```

## Uso

1. Abre el archivo `extractor.py` ubicado en la carpeta `sigrid_data_extractor`.
2. Modifica los parámetros en la sección de parámetros del script según tus necesidades. Por ejemplo, puedes cambiar la región, el tipo de desastre, etc.
3. Guarda los cambios y ejecuta el script.
    ```sh
    poetry run start
    ```
4. Los resultados se guardarán en el directorio principal del proyecto con los nombres `output.csv` y `output.xlsx`.

## Parámetros

- `service_endpoint`: URL del servicio de SIGRID.
- `geometry`: Define la geometría espacial para la consulta. Especifica una región en sí, no necesariamente un departamento.
- `where`: Condición WHERE para la consulta.
- `out_fields`: Campos a ser recuperados en la consulta.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue si encuentras un bug o tienes una sugerencia de mejora.

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.