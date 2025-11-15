"""
Autor: David Duran
Fecha: 2023-02-21
Descripción: Este programa simula el comportamiento 
de partículas en una caja utilizando el potencial de
Lennard-Jones y el algoritmo de velocidad de Verlet
para la integración de ecuaciones diferenciales.
El programa utiliza la biblioteca numba para cálculos
de fuerza más rápidos y la biblioteca matplotlib para
visualizar la simulación a través de animación.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numba

# Simulación de interacciones molecular de Lennard-Jones
np.random.seed(0)

# Definiendo las constantes
sig = 1
ep = 0.4
m = 1

# Longitud de la caja
lx = ly = lz = 20

# Parámetros de la simulación
n_particulas = 729
n_pasos = 50000
dt = 0.005
va = np.sqrt(12)
rc = 2.5 * sig
avx = avy = avz = 0
t2 = (dt ** 2) / (2 * m)
t1 = dt / (2 * m)
fc = ep * (24 / rc ** 7) * ((2 / rc ** 6) - 1)
Vc = ep * (4 / rc ** 6) * ((1 / rc ** 6) - 1)
rs = 8
max_neigh = 1000

# Función para inicializar las posiciones
@numba.jit
def inicializar_posiciones():
    # Arreglos vacíos para las posiciones
    x = np.zeros(n_particulas)
    y = np.zeros(n_particulas)
    z = np.zeros(n_particulas)
    
    # Establecer los rangos de las posiciones y el paso
    p1, p2, dp = 2, 19, 2
    
    # Iterar sobre todas las posiciones posibles y llenar los arreglos de posiciones
    kk = 0
    for p in range(p1, p2+1, dp):
        for q in range(p1, p2+1, dp):
            for r in range(p1, p2+1, dp):
                x[kk], y[kk], z[kk] = p, q, r
                kk += 1
                
                # Retornar los arreglos de posiciones si se alcanza el número máximo de partículas
                if kk >= n_particulas:
                    return x, y, z
    
    # Retornar los arreglos de posiciones si todas las posiciones posibles están llenas
    return x, y, z

# Función para inicializar las velocidades
@numba.jit
def inicializar_velocidades():
    vx = va * (np.random.rand(n_particulas) - 0.5)
    vy = va * (np.random.rand(n_particulas) - 0.5)
    vz = va * (np.random.rand(n_particulas) - 0.5)

    # Calcular y restar la velocidad promedio
    avx, avy, avz = np.mean(vx), np.mean(vy), np.mean(vz)
    vx, vy, vz = vx - avx, vy - avy, vz - avz
    KEN = 0.5 * np.sum(vx**2 + vy**2 + vz**2)
    return vx, vy, vz, KEN

# Función para inicializar y calcular las fuerzas
@numba.jit
def calcular_fuerzas(x, y, z, neigh, nlist):
    # Inicializar los arreglos de fuerzas
    fx, fy, fz = np.zeros(n_particulas), np.zeros(n_particulas), np.zeros(n_particulas)
    PEN = 0 # Energía potencial
    
    # Iterar sobre todas las partículas y calcular las fuerzas y la energía potencial
    for i in range(n_particulas - 1):
        for j in range(neigh[i]):
            pp = nlist[i, j]
            dx, dy, dz = x[i] - x[pp], y[i] - y[pp], z[i] - z[pp]
            if abs(dx) > (lx / 2):
                dx = (lx - abs(dx)) * (-dx) / abs(dx)
            if abs(dy) > (ly / 2):
                dy = (ly - abs(dy)) * (-dy) / abs(dy)
            if abs(dz) > (lz / 2):
                dz = (lz - abs(dz)) * (-dz) / abs(dz)
            
            # Calcular la distancia entre partículas
            r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            
            # Calcular las fuerzas y actualizar los arreglos de fuerzas
            if r <= rc and r > 0:
                f = ep * (24 / r ** 7) * ((2 / r ** 6) - 1)
                fm = f - fc
                fx[i] += fm * (dx / r)
                fy[i] += fm * (dy / r)
                fz[i] += fm * (dz / r)
                fx[pp] -= fm * (dx / r)
                fy[pp] -= fm * (dy / r)
                fz[pp] -= fm * (dz / r)
                
                # Calcular la energía potencial y actualizar la energía potencial total
                V = ep * (4 / r ** 6) * ((1 / r ** 6) - 1)
                Vm = V - Vc + r * fc - rc * fc
                PEN += Vm
    
    # Calcular la energía potencial promedio por partícula y retornar los arreglos de fuerzas y 
    # la energía potencial
    return fx, fy, fz, PEN

# Función para actualizar las posiciones
@numba.jit
def actualizar_posiciones(x, y, z, vx, vy, vz, fx, fy, fz):
    # Iterar sobre todas las partículas y actualizar las posiciones y las velocidades
    for i in range(n_particulas):
        x[i] += vx[i] * dt + fx[i] * t2
        y[i] += vy[i] * dt + fy[i] * t2
        z[i] += vz[i] * dt + fz[i] * t2

        # Aplicar condiciones de frontera periódicas y actualizar las velocidades si la partícula
        # choca con una "pared"
        if x[i] > lx:
            x[i] = 2 * lx - x[i]
            vx[i] = -vx[i]
        elif x[i] < 0:
            x[i] = -x[i]
            vx[i] = -vx[i]
        if y[i] > ly:
            y[i] = 2 * ly - y[i]
            vy[i] = -vy[i]
        elif y[i] < 0:
            y[i] = -y[i]
            vy[i] = -vy[i]
        if z[i] > lz:
            z[i] = 2 * lz - z[i]
            vz[i] = -vz[i]
        elif z[i] < 0:
            z[i] = -z[i]
            vz[i] = -vz[i]
    
    # Retornar los arreglos de posiciones actualizadas
    return x, y, z

# Función para actualizar las velocidades
@numba.jit
def actualizar_velocidades(vx, vy, vz, fx, fy, fz, fox, foy, foz):
    # Iterar sobre todas las partículas y actualizar las velocidades
    for i in range(n_particulas):
        vx[i] = vx[i] + (fox[i] + fx[i]) * t1
        vy[i] = vy[i] + (foy[i] + fy[i]) * t1
        vz[i] = vz[i] + (foz[i] + fz[i]) * t1
    
    # Calcular la energía cinética y retornar los arreglos de velocidades actualizados y la energía
    # cinética promedio por partícula
    KEN = 0
    for i in range(n_particulas):
        KEN += 0.5 * (vx[i] ** 2 + vy[i] ** 2 + vz[i] ** 2)
    return vx, vy, vz, KEN

@numba.jit(nopython=True)
def Thermostat(vx, vy, vz, KE):
    SF = np.sqrt(KEi / KE)
    for i in range(n_particulas):
        vx[i] *= SF
        vy[i] *= SF
        vz[i] *= SF
    return vx, vy, vz

@numba.jit(nopython=True)
def neighbour(x, y, z):
    npoints = x.shape[0]
    neigh = np.zeros(npoints, dtype=np.int64)
    nlist = np.zeros((npoints, npoints), dtype=np.int64)
    for i in range(npoints-1):
        x1, y1, z1 = x[i], y[i], z[i]
        for j in range(i+1, npoints):
            x2, y2, z2 = x[j], y[j], z[j]
            dx, dy, dz = x2-x1, y2-y1, z2-z1
            if abs(dx) > (lx/2):
                dx = (lx - abs(dx)) * (-dx) / abs(dx)
            if abs(dy) > (ly/2):
                dy = (ly - abs(dy)) * (-dy) / abs(dy)
            if abs(dz) > (lz/2):
                dz = (lz - abs(dz)) * (-dz) / abs(dz)
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            if r <= rs:
                neigh[i] += 1
                nlist[i, neigh[i]-1] = j
                neigh[j] += 1
                nlist[j, neigh[j]-1] = i
    return neigh, nlist

# PROGRAMA PRINCIPAL
# Esta parte del código encarga de ejecutar las funciones anteriores y graficar los resultados
# Inicializar las condiciones iniciales
x, y, z = inicializar_posiciones()
x1, y1, z1 = x.copy(), y.copy(), z.copy()
vx, vy, vz, KEN = inicializar_velocidades()
KEi = KEN / n_particulas
neigh, nlist = neighbour(x, y, z)
fx, fy, fz, PEN = calcular_fuerzas(x, y, z, neigh, nlist)

# Inicializar los arreglos de energía potencial y cinética y el número de iteraciones
PE = np.empty(n_pasos)
KE = np.empty(n_pasos)
nit = np.arange(n_pasos)

# Iterar sobre el número de pasos y actualizar las posiciones, fuerzas, velocidades y energía
# utilizando las funciones anteriores
for k in range(n_pasos):
    x, y, z = actualizar_posiciones(x, y, z, vx, vy, vz, fx, fy, fz)
    fox, foy, foz = fx.copy(), fy.copy(), fz.copy()
    fx, fy, fz, PEN = calcular_fuerzas(x, y, z, neigh, nlist)
    vx, vy, vz, KEN = actualizar_velocidades(vx, vy, vz, fx, fy, fz, fox, foy, foz)
    PE[k] = PEN / n_particulas
    KE[k] = KEN / n_particulas
    nit[k] = k
    print('Iteraciones:', k+1)
    if k % 100 == 0:
      vx, vy, vz = Thermostat(vx, vy, vz, KE[k])
      neigh, nlist = neighbour(x, y, z)


# Utilizamos velocity para calcular la velocidad promedio de las partículas
# Esta variable se utiliza después para el cmap de la gráficas
velocity = np.sqrt(vx**2 + vy**2 + vz**2)

# ############################
# GRÁFICAS
# ############################
# Las gráficas forman parte de una gráfica de 2x2
# Se utiliza add_subplot para agregar las gráficas a la figura principal

# Configuración de la gráfica
plt.style.use('seaborn-v0_8-colorblind')

# Activar el uso de LaTeX para la renderización de texto
# Estos parámetros deben ser establecidos antes de crear la primera gráfica, de lo contrario serán ignorados
plt.rcParams.update({
    "ytick.color" : "black",
    "xtick.color" : "black",
    "axes.labelcolor" : "black",
    "axes.edgecolor" : "black",
    "text.usetex": True,
    "font.family": "serif",
})

# Inicializar la figura
fig = plt.figure(figsize=(16, 12))

# Graficar las posiciones iniciales
ax1 = fig.add_subplot(2, 2, 1, projection='3d')                 # 2 filas, 2 columnas, gráfica 1
ax1.scatter(x1, y1, z1, s=10, color='gray')                     # s es el tamaño de los puntos
ax1.set_xlim([0, lx])
ax1.set_ylim([0, ly])
ax1.set_zlim([0, lz])
ax1.set_title('Posiciones iniciales de las partículas', fontsize=12)
ax1.tick_params(labelsize=7)                                  # Tamaño de los números en los ejes
ax1.set_box_aspect((1,1,1))                                   # Aspecto de la gráfica
ax1.set_proj_type('ortho')                                    # Proyección ortogonal
ax1.view_init(elev=20, azim=-135)                             # Elevación y ángulo de la vista

# Graficar las posiciones finales
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
sc = ax2.scatter(x, y, z, s=10, c=velocity, cmap='viridis', alpha=0.5)  # Velocidad como color de los puntos
ax2.set_xlim([0, lx])
ax2.set_ylim([0, ly])
ax2.set_zlim([0, lz])
ax2.set_title('Posiciones finales de las partículas', fontsize=12)
ax2.tick_params(labelsize=7)
ax2.set_box_aspect((1,1,1))
ax2.set_proj_type('ortho')
ax2.view_init(elev=20, azim=-135)

# Añadir la barra de colores a la gráfica 2
cbar = fig.colorbar(sc, ax=ax2, shrink=0.5)
# cbar.ax.set_xlabel('Velocidad', fontsize=12)

TE = KE + PE # Energía total

# Graficar la evolución de la energía en el tiempo
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(nit, KE, linestyle='--', linewidth=2, color='C0')
ax3.plot(nit, PE, linestyle=':', linewidth=2, color='C1')
ax3.plot(nit, TE, linestyle='-', linewidth=2, color='C2')
ax3.legend(['Energía cinética', 'Energía potencial', 'Energía total'], fontsize=8)
ax3.set_xlabel('Número de iteraciones', fontsize=10)
ax3.set_ylabel('Energía', fontsize=10)
ax3.set_title('Energía vs. Iteraciones', fontsize=12)
ax3.tick_params(axis='both', labelsize=8)
ax3.grid(True, linestyle='--', alpha=0.5)

# Graficar la distribución de rapidez de las partículas
ax4 = fig.add_subplot(2, 2, 4)
ax4.hist(velocity, bins=20, density=True, alpha=0.75, color='gray')
ax4.set_xlabel('Rapidez de las partículas', fontsize=10)
ax4.set_ylabel('Densidad', fontsize=10)
ax4.set_title('Distribución de la rapidez', fontsize=12)
ax4.tick_params(axis='both', labelsize=8)
ax4.grid(True, linestyle='--', alpha=0.5)

# Guardar la figura como un archivo eps
fig.savefig('Simulación Molecular.eps', format='eps', dpi=1000, bbox_inches='tight')

# Crear la figura y los ejes
fig = plt.figure(figsize=(6, 6))

# ############################
# ANIMACIÓN
# ############################

# Inicializar la figura en 3D
ax = fig.add_subplot(111, projection='3d')

# Definir el colormap
cmap = plt.cm.jet

# Función para actualizar la posición de las partículas
def update(frame):
    # No se debería de utilizar variables globales, pero en este caso no encontré otra forma
    global x, y, z, vx, vy, vz, fx, fy, fz

    # Actualizar las posiciones, fuerzas y velocidades
    x, y, z = actualizar_posiciones(x, y, z, vx, vy, vz, fx, fy, fz)
    fx, fy, fz, PEN = calcular_fuerzas(x, y, z, neigh, nlist)
    vx, vy, vz, KEN = actualizar_velocidades(vx, vy, vz, fx, fy, fz, fx, fy, fz)

    # Rapidez actual de las partículas para el cmap
    actual_velocity = np.sqrt(vx**2 + vy**2 + vz**2)

    # Actualizar la gráfica
    ax.clear()
    # Limitar los ejes
    ax.set_xlim([0, lx])
    ax.set_ylim([0, ly])
    ax.set_zlim([0, lz])
    ax.scatter(x, y, z, s=10, c=actual_velocity, cmap='viridis', alpha=0.5)
    ax.set_title('Simulación 3D de Dinámica Molecular', fontsize=12)
    ax.tick_params(labelsize=7)
    ax.set_box_aspect((1,1,1))
    ax.set_proj_type('ortho')

    # Rotar la cámara
    angle = frame / 10.0                    # Ajustar la velocidad de rotación
    ax.view_init(elev=30.0, azim=angle)     # Ángulo de elevación y rotación
    
    # Mostrar el frame que se está graficando en la consola
    print(f"Frame {frame+1}/{n_pasos} completado")

    return ax

# Crear el objeto de animación
anim = animation.FuncAnimation(fig, update, frames=n_pasos, interval=50)

# Guardar la animación como un archivo mp4
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='David Duran'), bitrate=1800)
anim.save('MD_simulation.mp4', writer=writer, dpi=300)

# Mostrar la animación
plt.show()
