from numba.typed import List
# Import necessary libraries
import numpy as np
import numba
from numba import njit
from mayavi import mlab
from tqdm.notebook import tqdm

# Define the lattice size and initialize it
size = 10
lattice = np.ones((size, size, size))

# Define Metropolis function with spin flip tracking
@njit("(i8[:,:,:], i8, f8, f8)", nogil=True)
def metropolis(lattice, steps, BJ, energy):
    # Copy the lattice to not modify the original
    lattice = lattice.copy()

    # Create arrays to store the magnetization and the energy
    magnetization = np.zeros(steps)
    total_energy = np.zeros(steps)

    # Create a list to store the spin flips
    spin_flips = List()

    # Loop to update the lattice
    for t in range(steps):
        # Select a random point in the lattice and flip the spin
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        z = np.random.randint(0, size)

        initial_spin = lattice[x, y, z]
        proposed_spin = -initial_spin

        # Calculate the change in energy
        initial_energy = 0
        final_energy = 0
        neighbors = [(x - 1, y, z), (x + 1, y, z), (x, y - 1, z), 
                     (x, y + 1, z), (x, y, z - 1), (x, y, z + 1)]
        for nx, ny, nz in neighbors:
            if 0 <= nx < size and 0 <= ny < size and 0 <= nz < size:
                initial_energy += -initial_spin * lattice[nx, ny, nz]
                final_energy += -proposed_spin * lattice[nx, ny, nz]

        # Update the state with the designated probabilities
        delta_energy = final_energy - initial_energy
        if (delta_energy > 0) * (np.random.random() < np.exp(-BJ * delta_energy)):
            lattice[x, y, z] = proposed_spin
            energy += delta_energy
            spin_flips.append((t, x, y, z))
        elif delta_energy <= 0:
            lattice[x, y, z] = proposed_spin
            energy += delta_energy
            spin_flips.append((t, x, y, z))

        # Update the magnetization and the total energy
        magnetization[t] = lattice.sum()
        total_energy[t] = energy

    return magnetization, total_energy, spin_flips


# Function to animate spins
def animate_spins(spin_flips, output_dir):
    lattice = np.ones((size, size, size))
    last_t = -1

    for t, x, y, z in sorted(spin_flips):
        if t != last_t:
            visualize_spins(lattice, t, output_dir)
            last_t = t
        lattice[x, y, z] *= -1  # Flip the spin


# Function to simulate and visualize
def simulate_and_visualize(lattice, steps, BJ, energy, output_dir="."):
    lattice = lattice.astype(np.int64)
    magnetization, total_energy, spin_flips = metropolis(lattice, steps, BJ, energy)
    animate_spins(spin_flips, output_dir)

def visualize_spins(lattice, t, output_dir):
    mlab.figure(bgcolor=(1, 1, 1), size=(400, 308))

    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            for k in range(lattice.shape[2]):
                spin = lattice[i, j, k]
                color = (0, 0, 1) if spin == 1 else (1, 0, 0)
                mlab.points3d(i, j, k, color=color, scale_factor=0.2)

    mlab.savefig(f"{output_dir}/frame_{t:04d}.png")
    mlab.close()  # Close the figure to free memory

# Increase the temperature by increasing `BJ`
simulate_and_visualize(lattice, 10000, 1.0, 0)

