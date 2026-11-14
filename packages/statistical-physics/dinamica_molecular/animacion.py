"""Animación GIF de una trayectoria periódica."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def gif_trayectoria(
    cuadros: np.ndarray,
    lado: float,
    *,
    grosor_capa: float = 0.25,
    pixeles: int = 240,
    ms_por_cuadro: int = 80,
) -> bytes:
    """GIF de las partículas que empiezan en una capa delgada de la caja.

    Cada partícula conserva el color de su posición inicial en x, de modo que
    la mezcla de colores muestra la difusión. Se proyecta sobre el plano xy con
    las posiciones envueltas en la caja. Devuelve los bytes del GIF.
    """
    capa = cuadros[0, :, 2] < grosor_capa * lado
    color = cuadros[0, capa, 0] / lado
    dpi = 100
    fig, ax = plt.subplots(figsize=(pixeles / dpi, pixeles / dpi), dpi=dpi)
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.98)
    ax.set_xlim(0, lado)
    ax.set_ylim(0, lado)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    puntos = ax.scatter(
        np.zeros(color.size),
        np.zeros(color.size),
        c=color,
        cmap="turbo",
        vmin=0,
        vmax=1,
        s=18,
    )
    imagenes = []
    for cuadro in cuadros:
        puntos.set_offsets(np.mod(cuadro[capa, :2], lado))
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        imagenes.append(Image.fromarray(rgba).convert("RGB"))
    plt.close(fig)

    # Una paleta común y sin tramado mantiene el GIF pequeño.
    paleta = imagenes[0].quantize(colors=48, dither=Image.Dither.NONE)
    imagenes = [
        im.quantize(palette=paleta, dither=Image.Dither.NONE) for im in imagenes
    ]

    salida = io.BytesIO()
    imagenes[0].save(
        salida,
        format="GIF",
        save_all=True,
        append_images=imagenes[1:],
        duration=ms_por_cuadro,
        loop=0,
        optimize=True,
    )
    return salida.getvalue()
