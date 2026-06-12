import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from inspect import cleandoc

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.express as px
    from numba import njit

    return cleandoc, mo, njit, np, plt, px


@app.cell(hide_code=True)
def _(cleandoc, mo):
    mo.md(
        cleandoc(r"""
        # Modelo de Ising 3D: muestreo de Metropolis

        Hamiltoniano (sin campo externo):
        $$
        H(\sigma) = -J\sum_{\langle i,j \rangle} s_i s_j,
        \qquad s_i \in \{-1,+1\}
        $$

        Regla de aceptación de Metropolis para un cambio local con $\Delta E$:
        $$
        P_{\text{aceptar}} = \min\left(1, e^{-\beta \Delta E}\right)
        $$

        En este notebook usamos unidades con $J=1$ y controlamos $\beta J$ directamente.
    """)
    )
    return


@app.cell
def _(mo):
    L = mo.ui.slider(start=4, stop=20, step=1, value=10, label="L")
    betaJ = mo.ui.slider(start=0.05, stop=0.80, step=0.01, value=0.22, label="betaJ")
    sweeps = mo.ui.slider(start=50, stop=4000, step=50, value=1200, label="Barridos")
    burn_in = mo.ui.slider(start=0, stop=3000, step=50, value=300, label="Burn-in")
    seed = mo.ui.number(value=7, label="Semilla", start=0, step=1)
    snapshot_stride = mo.ui.slider(
        start=10, stop=400, step=10, value=50, label="Stride de snapshots"
    )
    max_snapshots = mo.ui.slider(
        start=10, stop=200, step=10, value=80, label="Máx snapshots"
    )
    render_step = mo.ui.slider(
        start=1, stop=20, step=1, value=3, label="Submuestreo 3D"
    )
    return (
        L,
        betaJ,
        burn_in,
        max_snapshots,
        render_step,
        seed,
        snapshot_stride,
        sweeps,
    )


@app.cell(hide_code=True)
def _(
    L,
    betaJ,
    burn_in,
    max_snapshots,
    mo,
    render_step,
    seed,
    snapshot_stride,
    sweeps,
):
    mo.vstack(
        [
            mo.hstack([L, betaJ, sweeps, burn_in, seed], justify="start"),
            mo.hstack([snapshot_stride, max_snapshots, render_step], justify="start"),
        ]
    )
    return


@app.cell
def _(njit, np):
    @njit
    def initial_lattice(L: int) -> np.ndarray:
        spins = np.ones((L, L, L), dtype=np.int8)
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    if np.random.random() < 0.5:
                        spins[i, j, k] = -1
        return spins

    @njit
    def delta_energy_periodic(
        spins: np.ndarray, i: int, j: int, k: int, J: float = 1.0
    ) -> float:
        L = spins.shape[0]
        s = spins[i, j, k]
        nn_sum = (
            spins[(i + 1) % L, j, k]
            + spins[(i - 1) % L, j, k]
            + spins[i, (j + 1) % L, k]
            + spins[i, (j - 1) % L, k]
            + spins[i, j, (k + 1) % L]
            + spins[i, j, (k - 1) % L]
        )
        return 2.0 * J * s * nn_sum

    @njit
    def total_energy_periodic(spins: np.ndarray, J: float = 1.0) -> float:
        L = spins.shape[0]
        e = 0.0
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    s = spins[i, j, k]
                    e -= (
                        J
                        * s
                        * (
                            spins[(i + 1) % L, j, k]
                            + spins[i, (j + 1) % L, k]
                            + spins[i, j, (k + 1) % L]
                        )
                    )
        return e

    @njit
    def run_ising_3d(
        L: int, betaJ: float, sweeps: int, snapshot_stride: int, max_snapshots: int
    ):
        spins = initial_lattice(L)
        n_sites = L * L * L

        energy = total_energy_periodic(spins)
        magnet = spins.sum()

        energies = np.empty(sweeps, dtype=np.float64)
        magnetizations = np.empty(sweeps, dtype=np.float64)
        acceptances = np.empty(sweeps, dtype=np.float64)
        snapshots = np.empty((max_snapshots, L, L, L), dtype=np.int8)
        snapshot_steps = np.empty(max_snapshots, dtype=np.int64)
        snapshot_count = 0

        for t in range(sweeps):
            accepted = 0
            for _ in range(n_sites):
                i = np.random.randint(0, L)
                j = np.random.randint(0, L)
                k = np.random.randint(0, L)

                dE = delta_energy_periodic(spins, i, j, k)
                if dE <= 0.0 or np.random.random() < np.exp(-betaJ * dE):
                    old_s = spins[i, j, k]
                    spins[i, j, k] = -old_s
                    energy += dE
                    magnet += -2 * old_s
                    accepted += 1

            energies[t] = energy / n_sites
            magnetizations[t] = magnet / n_sites
            acceptances[t] = accepted / n_sites
            if t % snapshot_stride == 0 and snapshot_count < max_snapshots:
                snapshots[snapshot_count] = spins
                snapshot_steps[snapshot_count] = t
                snapshot_count += 1

        return (
            spins,
            energies,
            magnetizations,
            acceptances,
            snapshots,
            snapshot_steps,
            snapshot_count,
        )

    return (run_ising_3d,)


@app.cell
def _(
    L,
    betaJ,
    burn_in,
    max_snapshots,
    np,
    run_ising_3d,
    seed,
    snapshot_stride,
    sweeps,
):
    np.random.seed(int(seed.value))
    (
        final_spins,
        energies,
        magnetizations,
        acceptances,
        snapshots_all,
        snapshot_steps_all,
        snapshot_count,
    ) = run_ising_3d(
        int(L.value),
        float(betaJ.value),
        int(sweeps.value),
        int(snapshot_stride.value),
        int(max_snapshots.value),
    )
    snapshots = snapshots_all[:snapshot_count]
    snapshot_steps = snapshot_steps_all[:snapshot_count]

    burn = min(int(burn_in.value), len(energies) - 1)
    energies_eq = energies[burn:]
    magnetizations_eq = magnetizations[burn:]
    acceptance_eq = acceptances[burn:]

    stats = {
        "E_mean": float(np.mean(energies_eq)),
        "E_std": float(np.std(energies_eq)),
        "|M|_mean": float(np.mean(np.abs(magnetizations_eq))),
        "|M|_std": float(np.std(np.abs(magnetizations_eq))),
        "acc_mean": float(np.mean(acceptance_eq)),
    }
    return (
        acceptances,
        burn,
        energies,
        final_spins,
        magnetizations,
        snapshot_steps,
        snapshots,
        stats,
    )


@app.cell(hide_code=True)
def _(burn, mo, stats):
    mo.md(f"""
    Burn-in aplicado: **{burn}** barridos.

    - $\\langle E/N \\rangle$: **{stats["E_mean"]:.4f} ± {stats["E_std"]:.4f}**
    - $\\langle |M|/N \\rangle$: **{stats["|M|_mean"]:.4f} ± {stats["|M|_std"]:.4f}**
    - Aceptación media (post burn-in): **{stats["acc_mean"]:.3f}**
    """)
    return


@app.cell
def _(acceptances, burn, energies, magnetizations, np, plt):
    t = np.arange(len(energies))

    _fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(t, energies, lw=1.2)
    axes[0].axvline(burn, color="k", ls="--", lw=1)
    axes[0].set_ylabel("E/N")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, magnetizations, lw=1.2)
    axes[1].axvline(burn, color="k", ls="--", lw=1)
    axes[1].set_ylabel("M/N")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, acceptances, lw=1.2)
    axes[2].axvline(burn, color="k", ls="--", lw=1)
    axes[2].set_ylabel("Aceptación")
    axes[2].set_xlabel("Barrido")
    axes[2].grid(alpha=0.3)

    _fig.tight_layout()
    plt.show()
    return


@app.cell
def _(final_spins, plt):
    lattice_size = final_spins.shape[0]
    k = lattice_size // 2

    _fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(final_spins[:, :, k], cmap="bwr", vmin=-1, vmax=1, origin="lower")
    ax.set_title(f"Corte z={k} de la configuración final")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.show()
    return


@app.cell
def _(px, render_step, snapshot_steps, snapshots):
    stride = int(render_step.value)
    records = []
    for frame_idx in range(len(snapshots)):
        sampled = snapshots[frame_idx][::stride, ::stride, ::stride]
        x, y, z = sampled.nonzero()
        spins = sampled[x, y, z]
        sweep = int(snapshot_steps[frame_idx])
        for idx in range(len(x)):
            records.append(
                {
                    "x": int(x[idx]),
                    "y": int(y[idx]),
                    "z": int(z[idx]),
                    "spin": "up" if spins[idx] > 0 else "down",
                    "sweep": sweep,
                }
            )

    fig3d = px.scatter_3d(
        records,
        x="x",
        y="y",
        z="z",
        color="spin",
        animation_frame="sweep",
        color_discrete_map={"up": "crimson", "down": "royalblue"},
        opacity=0.85,
        title="Configuración 3D por barrido",
    )
    fig3d.update_traces(marker={"size": 4})
    fig3d.update_layout(
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
        height=560,
        legend_title_text="Spin",
    )
    return


@app.cell(hide_code=True)
def _(cleandoc, mo):
    mo.md(
        cleandoc(r"""
        Notas:
        - Se usan condiciones de frontera periódicas.
        - El tiempo de Monte Carlo está medido en barridos completos (un intento por sitio en promedio).
        - La celda de exportación (frames/video) se deja fuera del flujo principal de clase.
    """)
    )
    return


if __name__ == "__main__":
    app.run()
