import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import warnings
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from obspy import UTCDateTime, read
    from obspy.signal.trigger import classic_sta_lta
    from obspy.taup import TauPyModel

    # ObsPy redondea el delta de precisión simple de SAC y emite esta advertencia.
    warnings.filterwarnings("ignore", message="Sample spacing read from SAC file")

    DATOS = Path(__file__).resolve().parent / "archivos"
    return DATOS, TauPyModel, UTCDateTime, classic_sta_lta, mo, np, plt, read


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Procesamiento de señales sísmicas con ObsPy

    Registros de la componente vertical (BHZ) del terremoto de Pisco del 15 de
    agosto de 2007 (Mw 8.0), en formato SAC. Este cuaderno reemplaza el flujo de
    SAC y GrafSac del laboratorio 3. Lee los archivos, quita la tendencia, filtra,
    ordena los registros por distancia, calcula espectros y pica la llegada de la
    onda P con un detector STA/LTA.

    Todo se calcula al construir la página, a partir de los archivos de
    `lab3/archivos/`.
    """)
    return


@app.cell(hide_code=True)
def _(DATOS, UTCDateTime, np, read):
    # IU.TUC...234933_filtered.SAC es una salida de SAC (rmean, trans, bp, int).
    # No es un registro crudo y repetiría la estación.
    archivos = sorted(
        p for p in DATOS.glob("*.SAC") if not p.stem.endswith("_filtered")
    )
    trazas = [read(p, format="SAC")[0] for p in archivos]

    # El centro de datos escribió el tiempo origen del evento como O, en
    # segundos desde la hora de referencia de cada archivo.
    origenes = np.array(
        [(tr.stats.starttime + tr.stats.sac.o).timestamp for tr in trazas]
    )
    origen_tipico = np.median(origenes)
    principal = np.abs(origenes - origen_tipico) < 60
    origen_principal = UTCDateTime(round(origen_tipico))

    def etiqueta(tr):
        return f"{tr.stats.network}.{tr.stats.station}"

    otro_evento = [etiqueta(tr) for tr, p in zip(trazas, principal) if not p]
    otro_origen = (
        UTCDateTime(round(origenes[~principal][0])) if (~principal).any() else None
    )
    return etiqueta, origen_principal, otro_evento, otro_origen, principal, trazas


@app.cell(hide_code=True)
def _(etiqueta, mo, origen_principal, otro_evento, otro_origen, principal, trazas):
    _estaciones = ", ".join(sorted(etiqueta(tr) for tr in trazas))
    mo.md(f"""
    ## Los datos

    Se leyeron **{len(trazas)} registros** BHZ. Son los archivos crudos de la
    carpeta; el archivo `..._filtered.SAC` de IU.TUC es un resultado de SAC y no
    se vuelve a leer.

    {_estaciones}.

    **Un evento distinto.** Las cabeceras de **{", ".join(otro_evento)}** fijan
    un origen en {otro_origen.strftime("%Y-%m-%d %H:%M:%S")} UTC y una profundidad de 24.7 km.
    Los otros {int(principal.sum())} registros fijan
    {origen_principal.strftime("%Y-%m-%d %H:%M:%S")} UTC y 41.2 km. Esas dos estaciones
    registraron otro terremoto, no el de Pisco. Se pican igual, pero no entran
    en la sección sísmica ni en los espectros.

    **Redes duplicadas.** KIP y SFJD aparecen en dos redes (G o GE, e IU) con la
    misma ubicación. Son registros distintos de la misma estación y se tratan
    como dos.
    """)
    return


@app.cell(hide_code=True)
def _(etiqueta, mo, trazas):
    _escalas = [float(tr.stats.sac.scale) for tr in trazas]
    _sensores = sorted({tr.stats.sac.kinst.strip().title() for tr in trazas})
    mo.md(f"""
    ## Amplitudes en cuentas

    La respuesta instrumental solo se puede quitar si se conoce. Las cabeceras SAC
    de estos archivos traen un factor `SCALE` por registro (entre
    {min(_escalas):.2g} y {max(_escalas):.2g}) y un texto de instrumento
    (`KINST`: {", ".join(_sensores)}). No traen polos ni ceros, y el repositorio
    no tiene los archivos `SACPZ` ni `RESP` que mencionaba la guía del
    laboratorio. Sin eso no hay función de transferencia que deconvolucionar.

    **Todas las amplitudes de este cuaderno están en cuentas (counts) del
    digitalizador.** Los sismómetros y digitalizadores difieren entre estaciones,
    así que una amplitud en cuentas no se compara de forma cuantitativa entre
    dos estaciones. Sí se pueden comparar formas de onda, tiempos y contenido de
    frecuencias.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocesamiento y sección sísmica

    Cada registro se prepara así: se quita la tendencia lineal (esto incluye la
    media), se aplica una ventana de coseno (Hann) al 5 % de cada extremo para
    que el filtro no vea un salto, y se filtra con un pasabanda Butterworth de
    0.01 a 0.2 Hz (periodos de 5 a 100 s), el mismo rango que usa la guía del
    laboratorio (`bp n 5 corner 0.01 0.2`). El filtro es de fase cero: no
    desplaza la forma de onda, pero deja energía antes de la llegada. Para
    picar la onda P más abajo se usa otro filtro, causal.

    La sección ordena las estaciones por distancia epicentral (campo `GCARC`) y
    normaliza cada traza por su máximo. El eje de tiempo se mide desde el origen
    del terremoto.
    """)
    return


@app.cell(hide_code=True)
def _(etiqueta, plt, principal, trazas):
    def preparar(tr, banda=(0.01, 0.2)):
        x = tr.copy()
        x.detrend("linear")
        x.taper(0.05)
        x.filter(
            "bandpass", freqmin=banda[0], freqmax=banda[1], corners=5, zerophase=True
        )
        return x

    def tiempo_desde_origen(tr):
        # times() cuenta desde el inicio del registro.
        # O y B son relativos a la misma hora de referencia.
        return tr.times() + tr.stats.sac.b - tr.stats.sac.o

    seccion = sorted(
        (tr for tr, p in zip(trazas, principal) if p), key=lambda tr: tr.stats.sac.gcarc
    )
    _fig, _ax = plt.subplots(figsize=(7, 7.2), dpi=72)
    for _i, _tr in enumerate(seccion):
        _x = preparar(_tr)
        # Tras el filtro no queda energía sobre 0.2 Hz: 1 muestra por segundo
        # basta y mantiene la imagen pequeña.
        _paso = max(1, int(round(_x.stats.sampling_rate)))
        _ax.plot(
            tiempo_desde_origen(_x)[::_paso],
            _i + 0.45 * _x.data[::_paso] / abs(_x.data).max(),
            lw=0.5,
            c="k",
        )
    _ax.set_yticks(range(len(seccion)))
    _ax.set_yticklabels(
        [f"{etiqueta(tr)}  {tr.stats.sac.gcarc:.1f}°" for tr in seccion], fontsize=8
    )
    _ax.set_ylim(-1, len(seccion))
    _ax.set_xlabel("Tiempo desde el origen (s)")
    _ax.set_title("Pisco 2007, BHZ, pasabanda 0.01 a 0.2 Hz, ordenado por distancia")
    _ax.grid(axis="x", alpha=0.3)
    _fig.tight_layout()
    _fig
    return preparar, seccion, tiempo_desde_origen


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Picado de la onda P con STA/LTA

    El detector STA/LTA compara la energía de una ventana corta (STA) con la de
    una ventana larga (LTA) que la precede. En el ruido el cociente es cercano a
    1. Cuando llega una señal, la ventana corta la ve antes y el cociente sube.
    Se declara la llegada en la primera muestra en que el cociente supera un
    umbral.

    Parámetros, los mismos para todas las trazas:

    - Filtro: pasabanda Butterworth causal de 1 a 3 Hz, de cuatro polos. Un
      filtro de fase cero adelantaría el pico.
    - STA de 2 s, LTA de 20 s, umbral 3.
    - La búsqueda empieza a los 40 s (el doble de la LTA). Antes de eso la LTA
      no está llena y el transitorio del filtro dispara falsas alarmas.

    Estos valores se escogieron probando varias combinaciones y se comparan
    abajo con una referencia independiente, así que el acuerdo no es una prueba
    ciega del método.

    **Referencia.** Las cabeceras SAC pueden guardar picks en los campos `A` y
    `T0` a `T9`. Aquí se revisan todos los registros y se cuentan los que tienen
    alguno. Como referencia independiente se usa además el tiempo teórico de la
    onda P en el modelo iasp91 (módulo `obspy.taup`, que trae el modelo y no
    usa red) para la distancia y la profundidad de cada cabecera.
    """)
    return


@app.cell(hide_code=True)
def _(TauPyModel, classic_sta_lta, np, tiempo_desde_origen, trazas):
    CAMPOS_PICK = ("a", *(f"t{i}" for i in range(10)))
    BANDA_PICK = (1.0, 3.0)
    STA, LTA, UMBRAL = 2.0, 20.0, 3.0

    def picar_p(tr):
        x = tr.copy()
        x.detrend("linear")
        x.filter(
            "bandpass",
            freqmin=BANDA_PICK[0],
            freqmax=BANDA_PICK[1],
            corners=4,
            zerophase=False,
        )
        fs = x.stats.sampling_rate
        razon = classic_sta_lta(x.data, int(STA * fs), int(LTA * fs))
        inicio = int(2 * LTA * fs)
        sobre = np.flatnonzero(razon[inicio:] > UMBRAL)
        indice = inicio + sobre[0] if sobre.size else None
        t = tiempo_desde_origen(x)
        return {
            "t": t,
            "senal": x.data,
            "razon": razon,
            "pick": None if indice is None else t[indice],
        }

    def picks_de_cabecera(tr):
        # Tiempos de los picks guardados en la cabecera, desde el origen.
        sac = tr.stats.sac
        return {c: sac[c] - sac.o for c in CAMPOS_PICK if c in sac}

    modelo = TauPyModel("iasp91")

    def p_teorica(tr):
        llegadas = modelo.get_travel_times(
            source_depth_in_km=float(tr.stats.sac.evdp),
            distance_in_degree=float(tr.stats.sac.gcarc),
            phase_list=["P"],
        )
        return llegadas[0].time

    # Un pick a más de TOLERANCIA de la P teórica se marca para revisión.
    TOLERANCIA = 5.0

    resultados = []
    for _tr in trazas:
        _p = picar_p(_tr)
        _teorica = p_teorica(_tr)
        resultados.append(
            {
                "traza": _tr,
                "picado": _p,
                "teorica": _teorica,
                "dif": None if _p["pick"] is None else _p["pick"] - _teorica,
                "cabecera": picks_de_cabecera(_tr),
            }
        )
    return CAMPOS_PICK, LTA, STA, TOLERANCIA, UMBRAL, resultados


@app.cell(hide_code=True)
def _(CAMPOS_PICK, TOLERANCIA, etiqueta, mo, np, principal, resultados):
    _con_cabecera = [
        r for r in resultados if r["cabecera"] and r["picado"]["pick"] is not None
    ]
    # Si hay varios picks, se compara con el primero: A antes que T0 y los demás.
    _dif_cabecera = [
        r["picado"]["pick"] - r["cabecera"][next(iter(r["cabecera"]))]
        for r in _con_cabecera
    ]
    _frase_cabecera = (
        f"Contra esos picks, STA/LTA difiere en una mediana de {np.median(_dif_cabecera):+.1f} s."
        if _con_cabecera
        else "No hay picks de analista contra los cuales comparar, así que la comparación se hace con el tiempo teórico de iasp91."
    )
    _sin_pick = [
        etiqueta(r["traza"]) for r in resultados if r["picado"]["pick"] is None
    ]
    _filas = []
    for _r, _p in zip(resultados, principal):
        _tr = _r["traza"]
        _t = _r["picado"]["pick"]
        _dif = _r["dif"]
        _marca = "" if _p else " (otro evento)"
        if _dif is None:
            _estado = "sin pick"
        else:
            _estado = "revisar" if abs(_dif) > TOLERANCIA else "ok"
        _filas.append(
            f"| {etiqueta(_tr)}{_marca} | {_tr.stats.sac.gcarc:.1f} | {_r['teorica']:.1f} "
            f"| {'' if _t is None else f'{_t:.1f}'} | {'' if _dif is None else f'{_dif:+.1f}'} | {_estado} |"
        )
    _buenos = np.array(
        [
            r["dif"]
            for r in resultados
            if r["dif"] is not None and abs(r["dif"]) <= TOLERANCIA
        ]
    )
    mo.md(
        f"""
    ### Resultado

    **Picks en las cabeceras.** {len(_con_cabecera)} de {len(resultados)} registros
    tienen un valor en alguno de los campos {", ".join(f"`{c.upper()}`" for c in CAMPOS_PICK)}.
    {_frase_cabecera}

    **STA/LTA contra iasp91.** {len(_buenos)} de {len(resultados)} picks quedan a
    menos de {TOLERANCIA:g} s de la onda P teórica. La mediana de esos {len(_buenos)} es
    {np.median(_buenos):+.1f} s: el detector llega tarde porque necesita que la
    ventana STA se llene de señal, y la onda P telesísmica empieza de forma
    gradual. {len(resultados) - len(_buenos)} picks quedan marcados "revisar"
    (más de {TOLERANCIA:g} s de diferencia) y {len(_sin_pick)} registros no tienen pick.

    | Estación | Distancia (°) | P teórica (s) | P STA/LTA (s) | Diferencia (s) | Estado |
    | --- | ---: | ---: | ---: | ---: | --- |
    {chr(10).join(_filas)}

    Los tiempos se miden desde el origen del terremoto de cada registro.
    """
    )
    return


@app.cell(hide_code=True)
def _(TOLERANCIA, etiqueta, plt, principal, resultados):
    _fig, _ax = plt.subplots(figsize=(7, 3.6), dpi=72)
    _rotulos = {}
    for _r, _p in zip(resultados, principal):
        if _r["dif"] is None:
            continue
        _x = _r["traza"].stats.sac.gcarc
        _revisar = abs(_r["dif"]) > TOLERANCIA
        _ax.scatter(
            _x,
            _r["dif"],
            c="tab:red" if _revisar else "tab:blue",
            marker="o" if _p else "s",
            s=28,
        )
        if _revisar:
            # Registros de la misma estación coinciden en el gráfico.
            _rotulos.setdefault((_x, _r["dif"]), []).append(etiqueta(_r["traza"]))
    for (_x, _y), _nombres in _rotulos.items():
        _ax.annotate(
            " y ".join(_nombres),
            (_x, _y),
            fontsize=7,
            xytext=(-6, 4),
            textcoords="offset points",
            ha="right",
        )
    _ax.axhline(0, c="k", lw=0.6)
    _ax.axhspan(-TOLERANCIA, TOLERANCIA, color="0.9", zorder=0)
    _ax.set_xlabel("Distancia epicentral (°)")
    _ax.set_ylabel("P STA/LTA menos P teórica (s)")
    _ax.set_title("Círculos: Pisco. Cuadrados: otro evento. Rojo: revisar")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(LTA, STA, TOLERANCIA, UMBRAL, etiqueta, plt, resultados):
    _por_nombre = {etiqueta(r["traza"]): r for r in resultados}
    _malos = [
        n
        for n, r in _por_nombre.items()
        if r["dif"] is not None and abs(r["dif"]) > TOLERANCIA
    ]
    _elegidos = ["IU.TUC", _malos[0] if _malos else "IU.ANMO"]
    _fig, _ax = plt.subplots(2, 2, figsize=(8, 4.8), dpi=72, sharex=True)
    for _j, _nombre in enumerate(_elegidos):
        _r = _por_nombre[_nombre]
        _p = _r["picado"]
        _m = _p["t"] < _r["teorica"] + 70
        _ax[0, _j].plot(_p["t"][_m], _p["senal"][_m], lw=0.5, c="k")
        _ax[1, _j].plot(_p["t"][_m], _p["razon"][_m], lw=0.8, c="tab:blue")
        _ax[1, _j].axhline(
            UMBRAL, c="tab:orange", ls="--", lw=0.8, label=f"umbral {UMBRAL:g}"
        )
        for _k in (0, 1):
            _ax[_k, _j].axvline(
                _r["teorica"], c="tab:green", lw=1, label="P teórica (iasp91)"
            )
            if _p["pick"] is not None:
                _ax[_k, _j].axvline(_p["pick"], c="tab:red", lw=1, label="pick STA/LTA")
        _ax[0, _j].set_title(f"{_nombre}, filtro 1 a 3 Hz")
        _ax[1, _j].set_xlabel("Tiempo desde el origen (s)")
    _ax[0, 0].set_ylabel("Cuentas")
    _ax[1, 0].set_ylabel(f"STA/LTA ({STA:g} s / {LTA:g} s)")
    _ax[1, 1].legend(fontsize=7, loc="upper left")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(TOLERANCIA, etiqueta, mo, resultados):
    _dif = {etiqueta(r["traza"]): r["dif"] for r in resultados}
    mo.md(f"""
    ### Dónde falla el detector

    La figura de la izquierda es un caso que funciona. IU.TUC tiene un ruido previo
    muy bajo en 1 a 3 Hz, la señal lo supera con claridad y el cociente salta de 2
    a casi 9 en pocos segundos.

    Los picos marcados "revisar" (más de {TOLERANCIA:g} s) son de tres tipos:

    - **G.KIP, IU.KIP y G.TAOE** ({_dif["G.KIP"]:+.1f}, {_dif["IU.KIP"]:+.1f} y
      {_dif["G.TAOE"]:+.1f} s). El ruido en 1 a 3 Hz es comparable con la
      señal de la onda P y el cociente nunca sube con claridad. Es el caso de la
      figura de la derecha. El pick es la primera fluctuación que supera el umbral
      y no la llegada. Se necesitaría otra banda de frecuencia o un umbral por
      estación.
    - **IU.PMSA** ({_dif["IU.PMSA"]:+.1f} s). Un pulso de ruido antes de la P cruza el
      umbral. Es una falsa alarma.
    - **IU.POHA** ({_dif["IU.POHA"]:+.1f} s). La onda P es emergente: empieza con
      amplitud pequeña que crece despacio, y el cociente cruza el umbral después
      de la llegada real.

    Un detector con parámetros fijos para 30 registros de ruido y distancia tan
    distintos no puede dar siempre la misma calidad. Por eso la tabla marca cuáles
    picks revisar a mano.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Espectros de amplitud

    Se comparan tres estaciones a distancias distintas. Para cada una se toma una
    ventana de la señal, de 5 s antes del pick STA/LTA a 115 s después, y una
    ventana de ruido, los primeros 40 s del registro, que anteceden a la P. A cada
    ventana se le quita la tendencia lineal, se le aplica una ventana de Hann y se
    calcula la transformada de Fourier. Para que las ventanas de longitud
    distinta se puedan comparar, el espectro se expresa como densidad de amplitud,
    en cuentas por raíz de hercio.
    """)
    return


@app.cell(hide_code=True)
def _(etiqueta, np, plt, principal, resultados):
    def densidad_amplitud(x, dt):
        x = x - np.polyval(np.polyfit(np.arange(x.size), x, 1), np.arange(x.size))
        w = np.hanning(x.size)
        espectro = np.fft.rfft(x * w)
        f = np.fft.rfftfreq(x.size, dt)
        return f[1:], np.abs(espectro[1:]) * np.sqrt(2 * dt / (x.size * np.mean(w**2)))

    _por_nombre = {etiqueta(r["traza"]): r for r, p in zip(resultados, principal) if p}
    _fig, _ax = plt.subplots(1, 3, figsize=(10, 3.6), dpi=72, sharey=True)
    for _a, _nombre in zip(_ax, ["G.MPG", "IU.TUC", "IU.TSUM"]):
        _r = _por_nombre[_nombre]
        _tr = _r["traza"]
        _fs = _tr.stats.sampling_rate
        _t0 = _r["picado"]["pick"] - _tr.stats.sac.b + _tr.stats.sac.o
        _i0 = int((_t0 - 5) * _fs)
        _senal = _tr.data[_i0 : _i0 + int(120 * _fs)].astype(float)
        _ruido = _tr.data[: int(40 * _fs)].astype(float)
        _f, _s = densidad_amplitud(_senal, _tr.stats.delta)
        _fn, _n = densidad_amplitud(_ruido, _tr.stats.delta)
        _a.loglog(_f, _s, lw=0.8, c="k", label="señal (P + 115 s)")
        _a.loglog(_fn, _n, lw=0.8, c="tab:red", label="ruido (40 s previos)")
        _a.set_xlim(0.01, 5)
        _a.set_xlabel("Frecuencia (Hz)")
        _a.set_title(f"{_nombre}, {_tr.stats.sac.gcarc:.1f}°")
        _a.grid(alpha=0.3, which="both")
    _ax[0].set_ylabel("Densidad de amplitud (cuentas/√Hz)")
    _ax[0].legend(fontsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    En las tres estaciones la densidad de amplitud de la señal es casi plana por
    debajo de unos 0.1 Hz y baja más de dos órdenes de magnitud entre 0.1 y 2 Hz,
    como corresponde a una fuente que radia sobre todo a periodos largos. La
    señal supera al ruido por un factor de decenas a miles bajo 1 Hz. Hacia 3 a
    5 Hz las dos curvas se acercan, lo que explica por qué el detector STA/LTA
    de 1 a 3 Hz se apoya en la parte débil del espectro y falla en las
    estaciones ruidosas.

    Las alturas absolutas no se comparan entre estaciones: los instrumentos
    tienen sensibilidades distintas y el espectro está en cuentas.
    """)
    return


if __name__ == "__main__":
    app.run()
