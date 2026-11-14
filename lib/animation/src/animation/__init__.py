"""Figuras de Plotly cuyos cuadros se reproducen en el navegador, sin Python."""

import plotly.graph_objects as go

# marimo incrusta la figura como JSON escapado dos veces, así que cada byte
# repetido en cuadros y pasos pesa casi el triple en la página. Esta plantilla
# reemplaza a plotly_white (unos 7 KB) con lo que las figuras usan de ella.
_EJE = {
    "gridcolor": "#ebf0f8",
    "linecolor": "#ebf0f8",
    "zerolinecolor": "#ebf0f8",
    "zerolinewidth": 2,
    "automargin": True,
}
PLANTILLA = go.layout.Template(
    layout={
        "font": {"color": "#2a3f5f"},
        "colorway": ["#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a"],
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "hovermode": "closest",
        "xaxis": _EJE,
        "yaxis": _EJE,
    }
)


def figura_animada(
    trazas,
    cuadros,
    tiempos,
    *,
    animadas=None,
    ms_por_cuadro=50,
    formato="{:.2f}",
    prefijo="t = ",
    sufijo=" s",
    **layout,
):
    """Figura con botones de reproducir y pausar y un deslizador de tiempo.

    `cuadros` tiene, para cada instante de `tiempos`, un dict por cada traza de
    `animadas` (todas las de `trazas` si se omite) con sólo los campos que
    cambian. Plotly combina cada cuadro con las trazas, así que un campo que el
    cuadro omite, como una `y` fija, conserva su valor inicial.
    """
    indices = list(range(len(trazas))) if animadas is None else list(animadas)
    sin_transicion = {"duration": 0}
    frames = [
        go.Frame(data=cuadro, traces=indices, name=str(i))
        for i, cuadro in enumerate(cuadros)
    ]
    reproducir = {
        "label": "▶ Reproducir",
        "method": "animate",
        "args": [
            None,
            {
                "frame": {"duration": ms_por_cuadro, "redraw": False},
                "transition": sin_transicion,
                "fromcurrent": True,
            },
        ],
    }
    pausar = {
        "label": "❚❚ Pausa",
        "method": "animate",
        "args": [[None], {"mode": "immediate"}],
    }
    pasos = [
        {
            "label": formato.format(t),
            "method": "animate",
            "args": [[str(i)], {"mode": "immediate", "transition": sin_transicion}],
        }
        for i, t in enumerate(tiempos)
    ]
    figura = go.Figure(data=trazas, frames=frames)
    figura.update_layout(
        template=PLANTILLA,
        margin={"l": 50, "r": 20, "t": 50, "b": 20},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "buttons": [reproducir, pausar],
                "showactive": False,
                "x": 0,
                "xanchor": "left",
                "y": 0,
                "yanchor": "top",
                "pad": {"t": 140, "r": 10},
            }
        ],
        sliders=[
            {
                "steps": pasos,
                "y": 0,
                "yanchor": "top",
                "pad": {"t": 45},
                "currentvalue": {
                    "prefix": prefijo,
                    "suffix": sufijo,
                    "xanchor": "right",
                },
                "transition": sin_transicion,
            }
        ],
    )
    figura.update_layout(**layout)
    return figura
