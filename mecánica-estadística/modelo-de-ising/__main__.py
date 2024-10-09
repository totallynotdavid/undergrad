import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure
from numba import njit
import os
import subprocess

@njit
def generar_rejilla_3D(N):
    """
    Genera una rejilla 3D con giros aleatorios para un sistema NxNxN
    """
    init_aleatorio_3D = np.random.random((N, N, N))
    rejilla_3D = np.where(init_aleatorio_3D >= 0.5, 1, -1)
    return rejilla_3D

def obtener_energia_3D(rejilla):
    """
    Calcula la energía de una configuración de rejilla 3D dada
    """
    kern = generate_binary_structure(3, 1) 
    kern[1, 1, 1] = False
    arr = -rejilla * convolve(rejilla, kern, mode='constant', cval=0)
    return arr.sum()

@njit
def metropolis_3D_paso_individual(arr_giro, BJ, energia, N):
    """
    Realiza un paso del algoritmo de metropolis
    """
    arr_giro = arr_giro.copy()

    x, y, z = np.random.randint(0, N, size=3)
    giro_i = arr_giro[x, y, z]
    giro_f = -giro_i

    E_i = 0
    E_f = 0
    for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < N and 0 <= ny < N and 0 <= nz < N:
            E_i += -giro_i*arr_giro[nx, ny, nz]
            E_f += -giro_f*arr_giro[nx, ny, nz]

    dE = E_f - E_i
    if dE > 0 and np.random.random() < np.exp(-BJ * dE):
        arr_giro[x, y, z] = giro_f
        energia += dE
    elif dE <= 0:
        arr_giro[x, y, z] = giro_f
        energia += dE

    giro_neto = arr_giro.sum()
    return arr_giro, giro_neto, energia

def ejecutar_metropolis_3D(N, pasos, BJ):
    """
    Ejecuta el algoritmo de Metropolis por un número específico de pasos
    """
    rejilla = generar_rejilla_3D(N)
    energia = obtener_energia_3D(rejilla)

    giros = np.zeros(pasos)
    energias = np.zeros(pasos)

    for t in range(pasos):
        rejilla, giros[t], energias[t] = metropolis_3D_paso_individual(rejilla, BJ, energia, N)
        np.save(f"config_giro_{t}.npy", rejilla)

    return rejilla, giros, energias

def visualizar_rejilla_3D(rejilla, paso):
    """
    Visualiza una sola configuración de rejilla 3D
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = np.indices(rejilla.shape)
    c = np.where(rejilla > 0, 'r', 'b')
    a = np.where(rejilla > 0, 0.5, 0.5)  # 50% transparency for both spins

    ax.scatter(x, y, z, c=c.flatten(), s=100)
    ax.quiver(x, y, z, 0, 0, rejilla, color=c.flatten(), alpha=a.flatten())

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Configuración de giro en el paso {paso}')
    plt.savefig(f"config_giro_{paso}.png")
    plt.close()

def crear_video(carpeta_entrada, video_salida):
    """
    Crea un video a partir de imágenes usando ffmpeg
    """
    imagenes = sorted([img for img in os.listdir(carpeta_entrada) if img.endswith(".png")])
    cuadro = carpeta_entrada + imagenes[0]
    comprobar_tamaño_cuadro_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {cuadro}'
    tamaño_cuadro = subprocess.check_output(comprobar_tamaño_cuadro_cmd, shell=True).decode("utf-8").strip()

    os.system(f"ffmpeg -framerate 10 -y -s {tamaño_cuadro} -i '{carpeta_entrada}/config_giro_%d.png' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {video_salida}")

def graficar_cambio_energia(energias, figura_salida):
    """
    Grafica el cambio total de energía
    """
    plt.figure(figsize=(10, 6))
    plt.plot(energias)
    plt.title('Cambio en la Energía Total')
    plt.xlabel('Paso')
    plt.ylabel('Energía Total')
    plt.savefig(figura_salida)
    plt.close()

if __name__ == "__main__":
    N = 10
    pasos = 1000
    BJ = 0.2

    rejilla, giros_3D, energias_3D = ejecutar_metropolis_3D(N, pasos, BJ)

    for t in range(pasos):
        config_giro_n = np.load(f"config_giro_{t}.npy")
        visualizar_rejilla_3D(config_giro_n, t)

    np.save("rejilla_final.npy", rejilla)
    np.save("giros_3D.npy", giros_3D)
    np.save("energias_3D.npy", energias_3D)

    crear_video('./', 'output.mp4')
    graficar_cambio_energia(energias_3D, 'cambio_energia.png')
