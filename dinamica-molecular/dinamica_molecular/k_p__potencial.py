import csv
import numpy as np
from math import sqrt,exp
import matplotlib.pyplot as plt

# Definición de constantes para la molécula de potasio (K)
eps = (1.51/1000)*(27.21138386/2.0) # eV
gam = (19.3/1000)*(27.21138386/2.0) # eV
r0 = (8.253*0.52917721)             # angstroms
p = 10.58
q = 1.34

# Función para el potencial interatómico Ucoh = Urep - Uel
# Retorna el valor de la función para una distancia de separación constante d
def P(n,d):    
    # Términos para la iteración
    Urep = 0    # Potencial de repulsión
    Uel = 0     # Potencial de atracción

    sigma = r0*np.power(2,1/6)
    epsilon = 4*eps*np.power(q/p,2)*np.power(sigma/r0,12)*(np.power(sigma/r0,6)-1)
    
    # Iterar sobre la doble suma de la función de energía
    for i in range(n):
        Uel0 = 0
        
        for j in range(n):
            if j != i:
                # Hallar los términos repulsivos y atractivos de la función de energía
                Urep = Urep + eps*(exp(-p*((d/r0)-1.0)))
                Uel0 = Uel0 + (gam**2.0)*(exp(-2.0*q*((d/r0)-1.0)))
        
        Uel = Uel + sqrt(Uel0)
    
    # Hallar la energía cohesiva total
    Ucoh = Urep - Uel

    return Ucoh, sigma, epsilon


N = [2,3,4,5,6,7,8,9,10]      # Valores de N para los que se calculará el potencial
dist = np.arange(3,11,0.01)   # Valores de la distancia entre átomos para los que se calculará el potencial
PE = []

# Configuración de la gráfica
plt.style.use('seaborn-v0_8-colorblind')
plt.figure(dpi=100)

# Activar el uso de LaTeX para la renderización de texto
# Estos parámetros deben ser establecidos antes de crear la primera gráfica, de lo contrario serán ignorados
plt.rcParams.update({
    "ytick.color" : "black",
    "xtick.color" : "black",
    "axes.labelcolor" : "black",
    "axes.edgecolor" : "black",
    "text.usetex": True,
    "font.family": "serif",
    "font.monospace": 'Computer Modern Serif'
})

# Crear el archivo CSV que se usará para almacenar los datos
with open('data.csv', mode='w', newline='') as csv_file:
    fieldnames = ['N', 'min_PE', 'min_dist', 'sigma', 'epsilon']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Iterar sobre los valores de N para obtener las energías
    for n in N:
        PE = []
        for i in dist:
            PEi, sigma, epsilon = P(n,i)
            PE.append(PEi)
        
        min_PE = min(PE)
        min_dist = dist[np.argmin(PE)]
        sigma_val = sigma
        epsilon_val = epsilon
        
        writer.writerow({'N': n, 'min_PE': min_PE, 'min_dist': min_dist, 'sigma': sigma_val, 'epsilon': epsilon_val})
        
        """
        print(f"Iteración: {n+1}/{len(N)}")
        print('Para N = {}, el mínimo PE es {:.3f} eV a una distancia de {:.3f} angstroms.'.format(n, min_PE, min_dist))
        print('El valor de sigma es {:.3f} angstroms y el valor de epsilon es {:.3f} eV.'.format(sigma, epsilon))
        print('-'*50)
        """
        
        plt.plot(dist,PE,label='N={} átomos'.format(n))

    plt.plot([3, 11], [0, 0], 'k--')
    plt.xlabel('Separación Atómica (angstroms)', fontsize=12)
    plt.ylabel('Potencial Interatómico (eV)', fontsize=12)
    plt.legend(fontsize=12)
    plt.title('Potencial Lennard-Jones para\ndiferentes números de átomos', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Potencial Lennard-Jones.eps', format='eps')
    # plt.show()
