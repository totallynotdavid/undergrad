import os
import shutil

def copiar_archivos_sac(directorio_origen, directorio_destino):
    # Crear el directorio de destino si no existe
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)

    # Listar todos los archivos en el directorio de origen
    archivos = os.listdir(directorio_origen)

    # Filtrar y copiar archivos
    for archivo in archivos:
        if archivo.endswith('.SAC'):
            partes = archivo.split('.')
            if len(partes) >= 5:
                red, estacion, sensor, componente = partes[:4]
                
                # Verificar si el archivo cumple con los criterios
                if componente == 'BHZ' and sensor == '00':
                    ruta_origen = os.path.join(directorio_origen, archivo)
                    ruta_destino = os.path.join(directorio_destino, archivo)
                    shutil.copy2(ruta_origen, ruta_destino)
                    print(f"Copiado: {archivo}")

# Definir directorios
directorio_origen = r'C:\Users\david\Downloads\sismologia\2007-08-15-mw80-near-coast-of-peru'
directorio_destino = r'C:\Users\david\Downloads\sismologia\lab3\archivos'

copiar_archivos_sac(directorio_origen, directorio_destino)
print("Copia de archivos completada.")
