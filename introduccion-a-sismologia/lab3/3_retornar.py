import os
import shutil

def retornar_archivos(directorio_procesado, directorio_destino):
    # Recorrer todos los subdirectorios y archivos en el directorio procesado
    for root, dirs, files in os.walk(directorio_procesado, topdown=False):
        for file in files:
            # Ruta completa del archivo actual
            ruta_origen = os.path.join(root, file)
            ruta_destino = os.path.join(directorio_destino, file)
            
            # Mover el archivo al directorio de destino
            shutil.move(ruta_origen, ruta_destino)
            print(f"Archivo '{file}' movido a '{directorio_destino}'")
        
        # Después de mover los archivos, intentar eliminar el directorio si está vacío
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
                print(f"Directorio '{dir_path}' eliminado")
            except OSError:
                pass  # El directorio no está vacío o no se pudo eliminar

    # Intentar eliminar el directorio procesado si está vacío
    try:
        os.rmdir(directorio_procesado)
        print(f"Directorio '{directorio_procesado}' eliminado")
    except OSError:
        pass  # El directorio no está vacío o no se pudo eliminar

    print("Todos los archivos han sido retornados al directorio original.")

# Uso
directorio_procesado = r'C:\Users\david\Downloads\sismologia\lab3\archivos\comparacion'
directorio_destino = r'C:\Users\david\Downloads\sismologia\lab3\archivos'

retornar_archivos(directorio_procesado, directorio_destino)
