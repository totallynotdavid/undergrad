import os
import shutil
from collections import defaultdict

def mover_datos_estacion(directorio_origen, directorio_destino):
    # Crear el directorio de destino si no existe
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)

    # Diccionario para agrupar archivos por estación
    archivos_por_estacion = defaultdict(list)

    # Listar todos los archivos en el directorio de origen
    archivos = os.listdir(directorio_origen)

    # Agrupar archivos por estación
    for archivo in archivos:
        if archivo.endswith('.SAC'):
            partes = archivo.split('.')
            if len(partes) >= 5:
                red, estacion = partes[:2]
                clave_estacion = f"{red}.{estacion}"
                archivos_por_estacion[clave_estacion].append(archivo)

    # Procesar cada estación
    for estacion, archivos in archivos_por_estacion.items():
        if len(archivos) > 1:
            # Crear un subdirectorio para esta estación
            directorio_estacion = os.path.join(directorio_destino, estacion)
            os.makedirs(directorio_estacion, exist_ok=True)
            
            # Mover archivos a este subdirectorio
            for archivo in archivos:
                ruta_origen = os.path.join(directorio_origen, archivo)
                ruta_destino = os.path.join(directorio_estacion, archivo)
                shutil.move(ruta_origen, ruta_destino)
                print(f"Movido para comparación: {archivo}")

    print("Movimiento y separación de datos completada.")

# Uso
directorio_origen = r'C:\Users\david\Downloads\sismologia\lab3\archivos'
directorio_destino = r'C:\Users\david\Downloads\sismologia\lab3\archivos\comparacion'

mover_datos_estacion(directorio_origen, directorio_destino)
