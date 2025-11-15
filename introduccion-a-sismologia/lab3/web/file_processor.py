import os
import shutil
import logging
import subprocess
from collections import defaultdict

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_available_filters(ruta_origen):
    """
    Gets the available sensors and components from the raw files.
    """
    available_sensors = set()
    available_components = set()
    for nombre_archivo in os.listdir(ruta_origen):
        if nombre_archivo.endswith(".SAC"):
            try:
                red, estacion, file_sensor, file_component, *_ = nombre_archivo.split(
                    "."
                )
                available_sensors.add(file_sensor)
                available_components.add(file_component)
            except ValueError:
                logging.warning(
                    f"Unexpected filename: {nombre_archivo}"
                )  # Handle filenames that don't match
    return sorted(list(available_sensors)), sorted(list(available_components))


def copy_and_group_sac_files(
    ruta_origen,
    ruta_destino,
    filter_single_file_stations=False,
    sensors=None,
    components=None,
):
    """
    Copies, groups, and processes SAC files. Supports multi-filter mechanism.
    """
    try:
        os.makedirs(ruta_destino, exist_ok=True)
    except OSError as e:
        logging.error(f"Error creating directory {ruta_destino}: {e}")
        return None

    archivos_por_estacion = defaultdict(list)

    try:
        for nombre_archivo in os.listdir(ruta_origen):
            if nombre_archivo.endswith(".SAC"):
                try:
                    red, estacion, file_sensor, file_component, *_ = (
                        nombre_archivo.split(".")
                    )

                    # Check if sensor and component match the selected filters
                    sensor_match = (
                        sensors is None or not sensors or file_sensor in sensors
                    )
                    component_match = (
                        components is None
                        or not components
                        or file_component in components
                    )

                    if sensor_match and component_match:
                        origen_archivo = os.path.join(ruta_origen, nombre_archivo)
                        destino_archivo = os.path.join(ruta_destino, nombre_archivo)

                        if not os.path.exists(destino_archivo):
                            try:
                                shutil.copy2(origen_archivo, destino_archivo)
                            except OSError as e:
                                logging.error(f"Error copying {nombre_archivo}: {e}")

                        archivos_por_estacion[f"{red}.{estacion}"].append(
                            destino_archivo
                        )
                except ValueError:
                    logging.warning(f"Unexpected filename: {nombre_archivo}")

    except Exception as e:
        logging.error(f"General error during copy: {e}")
        return None

    if filter_single_file_stations:
        archivos_por_estacion = {
            estacion: archivos
            for estacion, archivos in archivos_por_estacion.items()
            if len(archivos) > 1
        }

    for estacion, archivos in archivos_por_estacion.items():
        create_sac_ps(ruta_destino, {estacion: archivos})
        convert_ps_to_png(ruta_destino, {estacion: archivos})

    return archivos_por_estacion


def create_sac_ps(ruta_destino, archivos_por_estacion):
    """
    Creates PS images from SAC files.
    """
    for estacion, archivos in archivos_por_estacion.items():
        for archivo in archivos:
            nombre_ps = f"{os.path.splitext(os.path.basename(archivo))[0]}.ps"
            ruta_ps = os.path.join(ruta_destino, nombre_ps)

            logging.debug(f"Processing file: {archivo}")
            logging.debug(f"PS file path: {ruta_ps}")

            if not os.path.exists(ruta_ps):
                try:
                    commands = [
                        f"r {os.path.basename(archivo)}",
                        "p",
                        f"saveimg {nombre_ps}",
                        "q\n",
                    ]
                    sac_command = ["sac"]
                    subprocess.run(
                        sac_command,
                        input="\n".join(commands),
                        cwd=ruta_destino,
                        capture_output=True,
                        text=True,
                        check=True,
                    )

                except subprocess.CalledProcessError as e:
                    logging.error(f"Error running SAC for {archivo}: {e.stderr}")


def convert_ps_to_png(ruta_destino, archivos_por_estacion):
    """
    Converts PS to PNG.
    """
    for estacion, archivos_sac in archivos_por_estacion.items():
        for archivo_sac in archivos_sac:
            nombre_ps = f"{os.path.splitext(os.path.basename(archivo_sac))[0]}.ps"
            ruta_ps = os.path.join(ruta_destino, nombre_ps)
            nombre_png = f"{os.path.splitext(os.path.basename(archivo_sac))[0]}.png"
            ruta_png = os.path.join(ruta_destino, nombre_png)

            if (
                os.path.exists(ruta_ps)
                and os.path.getsize(ruta_ps) > 0
                and not os.path.exists(ruta_png)
            ):
                command = [
                    "magick",
                    ruta_ps,
                    "-density",
                    "300",
                    "-colorspace",
                    "Gray",
                    "-background",
                    "white",
                    "-flatten",
                    ruta_png,
                ]

                try:
                    subprocess.run(
                        command,
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                    logging.debug(f"Converted {ruta_ps} to {ruta_png}")

                    os.remove(ruta_ps)
                except subprocess.CalledProcessError as e:
                    logging.error(
                        f"Error converting {ruta_ps} to PNG: {e}\n"
                        f"Command: {' '.join(command)}\n"
                    )
                except FileNotFoundError as e:
                    logging.error(f"ImageMagick's 'magick' not found: {e}")
