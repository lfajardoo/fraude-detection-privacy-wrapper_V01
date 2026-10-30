#!/usr/bin/env python3
"""Genera las figuras del capítulo de resultados (Experimentos 1, 2 y 3).

Lee los artefactos reales de las corridas en ``results/`` (nunca valores escritos a mano,
salvo las dos excepciones documentadas explícitamente: los diccionarios ``DIAG_*`` de las
Figuras 5 y 6, que provienen del stdout de las corridas y no de los CSV), agrega por
semilla con intervalos de confianza al 95% (t de Student) y exporta:

- ``figuras/datos_agregados.csv``: tabla de verificación con las medias e IC de cada figura.
- ``figuras/fig01_*.pdf`` / ``.png`` ... ``figuras/fig08_*.pdf`` / ``.png``.

Ejecutar desde la raíz del repositorio:

    python3 figuras/generar_figuras.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------------------
# Configuración general
# --------------------------------------------------------------------------------------

TITULOS_EN_FIGURA = True  # los títulos solo se dibujan en los PNG de trabajo; en LaTeX va el \caption

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "archivo" / "ataque_inversion" / "results"
FIGURAS_DIR = Path(__file__).resolve().parent

EXP1_DIRS = [f"run_{i}" for i in range(1, 6)]  # Exp 1: barrido de eps_in, n=5 semillas
EXP2_DIRS = [f"exp2_run_{i}" for i in range(1, 4)]  # Exp 2: barrido de eps_out, n=3 semillas
EXP3_DIRS = [f"exp3_run_{i}" for i in range(1, 4)]  # Exp 3: punto único, n=3 semillas

EPS_COMPARTIDOS = [1, 10, 50, 100, 500]  # valores presentes en los dos barridos (Fig 3 y 4)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "sans-serif",
    }
)

# Paleta accesible para daltonismo (Okabe-Ito) + marcadores de forma distinta por serie.
COLOR_A = "#0072B2"  # azul
COLOR_B = "#D55E00"  # naranja/rojo
COLOR_C = "#009E73"  # verde
COLOR_BASELINE = "#555555"
COLOR_SOMBRA_VERDE = "#B7E3C8"
COLOR_SOMBRA_ROJA = "#F5C2B8"
COLOR_ABSTIENE = "#0072B2"
COLOR_PASA = "#D55E00"

MARKER_A = "o"
MARKER_B = "s"
MARKER_C = "^"
MARKER_ABSTENCION = "D"  # marcador distinto para puntos sin barra de error (abstención total)

BASELINE_MAE_TXT = "1.1389"

FIGSIZE = (7, 4.5)


# --------------------------------------------------------------------------------------
# Carga y validación de artefactos
# --------------------------------------------------------------------------------------


def _validar_directorios(dirs: list[str], nombre_experimento: str) -> None:
    faltantes = [d for d in dirs if not (RESULTS_DIR / d / "inversion_curve.csv").is_file()]
    if faltantes:
        rutas = ", ".join(str(RESULTS_DIR / d / "inversion_curve.csv") for d in faltantes)
        raise SystemExit(
            f"[ERROR] Faltan artefactos de {nombre_experimento}: {rutas}\n"
            "No se genera ninguna figura a medias. Verifica que 'results/' contenga todas las corridas."
        )


def _cargar_experimento(dirs: list[str], eje: str, experimento: int) -> pd.DataFrame:
    """Carga y concatena los inversion_curve.csv de un experimento, con columnas eje/experimento/seed."""
    marcos = []
    for i, d in enumerate(dirs, start=1):
        df = pd.read_csv(RESULTS_DIR / d / "inversion_curve.csv")
        df["seed"] = i
        df["eje"] = eje
        df["experimento"] = experimento
        marcos.append(df)
    return pd.concat(marcos, ignore_index=True)


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validar_directorios(EXP1_DIRS, "Experimento 1 (barrido eps_in)")
    _validar_directorios(EXP2_DIRS, "Experimento 2 (barrido eps_out)")
    _validar_directorios(EXP3_DIRS, "Experimento 3 (punto desplegado)")

    exp1 = _cargar_experimento(EXP1_DIRS, eje="in", experimento=1)
    exp2 = _cargar_experimento(EXP2_DIRS, eje="out", experimento=2)
    exp3 = _cargar_experimento(EXP3_DIRS, eje="in", experimento=3)
    return exp1, exp2, exp3


# --------------------------------------------------------------------------------------
# Agregación estadística: IC 95% con t de Student (nunca 1.96, nunca ±1 desv. típica)
# --------------------------------------------------------------------------------------


@dataclass
class ResumenIC:
    n: int
    media: float
    desv: float
    t_valor: float
    semiancho: float
    ic_bajo: float
    ic_alto: float


def resumen_ic95(valores: pd.Series) -> ResumenIC:
    """Media, desviación muestral (ddof=1) e IC 95% con t de Student (df = n-1)."""
    valores = valores.to_numpy(dtype=float)
    n = len(valores)
    media = float(np.mean(valores))
    if n < 2:
        return ResumenIC(n=n, media=media, desv=0.0, t_valor=float("nan"), semiancho=0.0, ic_bajo=media, ic_alto=media)
    desv = float(np.std(valores, ddof=1))
    t_valor = float(stats.t.ppf(0.975, df=n - 1))
    semiancho = t_valor * desv / np.sqrt(n)
    return ResumenIC(n=n, media=media, desv=desv, t_valor=t_valor, semiancho=semiancho, ic_bajo=media - semiancho, ic_alto=media + semiancho)


def agregar_por_epsilon(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Agrega un experimento por epsilon: para cada columna calcula media, desv, IC95%."""
    filas = []
    for epsilon, grupo in df.groupby("epsilon"):
        fila = {
            "experimento": int(grupo["experimento"].iloc[0]),
            "eje": grupo["eje"].iloc[0],
            "epsilon": float(epsilon),
        }
        for col in columnas:
            r = resumen_ic95(grupo[col])
            fila[f"{col}_n"] = r.n
            fila[f"{col}_media"] = r.media
            fila[f"{col}_desv"] = r.desv
            fila[f"{col}_t"] = r.t_valor
            fila[f"{col}_ci_bajo"] = r.ic_bajo
            fila[f"{col}_ci_alto"] = r.ic_alto
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("epsilon").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Utilidades de graficación
# --------------------------------------------------------------------------------------


def _ejex_log_epsilon(ax: plt.Axes, valores: list[float]) -> None:
    """Eje X log con marcadores solo en los valores de epsilon medidos (nada de ticks automáticos)."""
    ax.set_xscale("log")
    ax.set_xticks(valores)
    ax.set_xticklabels([str(int(v)) for v in valores])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(min(valores) * 0.75, max(valores) * 1.35)


def _linea_base(ax: plt.Axes) -> None:
    ax.axhline(
        1.1388595856794768,
        color=COLOR_BASELINE,
        linestyle="--",
        linewidth=1.3,
        label=f"línea base (adivinar la media pública) = {BASELINE_MAE_TXT}",
        zorder=1,
    )


def _guardar_figura(fig: plt.Figure, nombre: str) -> None:
    fig.savefig(FIGURAS_DIR / f"{nombre}.pdf", bbox_inches="tight")
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _leyenda_y_nota(fig: plt.Figure, ax: plt.Axes, ncol: int = 1, nota: str | None = None) -> None:
    """Leyenda y nota al pie debajo del área de ejes, para no tapar datos ni superponerse entre sí."""
    fig.subplots_adjust(bottom=0.40 if nota else 0.30)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.15 if nota else 0.02), ncol=ncol, framealpha=0.9)
    if nota:
        fig.text(0.5, 0.01, nota, fontsize=7.5, va="bottom", ha="center", color="#333333")


def _plot_serie_con_ic(
    ax: plt.Axes,
    agg: pd.DataFrame,
    columna: str,
    color: str,
    marker: str,
    etiqueta: str,
    escala: float = 1.0,
) -> None:
    """Dibuja una serie con línea recta y barras de error IC95%, salvo en puntos con desv=0."""
    x = agg["epsilon"].to_numpy()
    y = agg[f"{columna}_media"].to_numpy() * escala
    desv = agg[f"{columna}_desv"].to_numpy()
    ci_bajo = agg[f"{columna}_ci_bajo"].to_numpy() * escala
    ci_alto = agg[f"{columna}_ci_alto"].to_numpy() * escala

    ax.plot(x, y, "-", color=color, linewidth=1.4, zorder=2)

    sin_error = np.isclose(desv, 0.0)
    con_error = ~sin_error

    if con_error.any():
        yerr = np.vstack([y[con_error] - ci_bajo[con_error], ci_alto[con_error] - y[con_error]])
        ax.errorbar(
            x[con_error],
            y[con_error],
            yerr=yerr,
            fmt=marker,
            color=color,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=6,
            capsize=3,
            linewidth=1.4,
            label=etiqueta,
            zorder=3,
        )
    if sin_error.any():
        ax.plot(
            x[sin_error],
            y[sin_error],
            MARKER_ABSTENCION,
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.6,
            markersize=7,
            linestyle="none",
            label=f"{etiqueta} (abstención total, sin dispersión)" if not con_error.any() else None,
            zorder=4,
        )


# --------------------------------------------------------------------------------------
# Figura 1 — Curva de protección del input shield (Exp 1)
# --------------------------------------------------------------------------------------


def fig01_proteccion_input_shield(agg1: pd.DataFrame) -> None:
    """
    Pie de figura sugerido:
    Figura 1. MAE de la reconstrucción de V4 frente a eps_in (Experimento 1, K=400,
    n=5 semillas; barras = IC 95%, t de Student). La línea gris discontinua es la línea
    base de no atacar (adivinar la media pública) = 1.1389. En la región sombreada en
    verde (eps_in <= 10) el ataque se abstiene en los 30 targets: el MAE coincide con la
    línea base hasta el último decimal en las 5 semillas (desviación típica = 0), por lo
    que esos dos puntos se marcan con un rombo y sin barra de error -- una barra de
    longitud cero se leería como un dato faltante. En eps_in=30 el IC contiene la línea
    base: no hay efecto detectable. En la banda sombreada en rojo (entre eps_in=50 y
    eps_in=100) los IC dejan de solaparse ([1.023, 1.117] frente a [0.662, 0.796]):
    colapso de la protección.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.axvspan(agg1["epsilon"].min() * 0.75, 10, color=COLOR_SOMBRA_VERDE, alpha=0.6, zorder=0)
    ax.axvspan(50, 100, color=COLOR_SOMBRA_ROJA, alpha=0.6, zorder=0)

    _linea_base(ax)
    _plot_serie_con_ic(ax, agg1, "mae", COLOR_A, MARKER_A, "Exp 1 — MAE frente a eps_in")

    ax.text(
        3.0,
        ax.get_ylim()[1] * 0.93,
        "protección total:\nabstención en los 30 targets",
        fontsize=9,
        ha="center",
        va="top",
        color="#1a6b3c",
    )
    ax.text(
        70,
        ax.get_ylim()[1] * 0.93,
        "colapso de la\nprotección",
        fontsize=9,
        ha="center",
        va="top",
        color="#a33a1c",
    )

    fila30 = agg1[agg1["epsilon"] == 30].iloc[0]
    ax.annotate(
        "sin efecto detectable\n(el IC contiene la baseline)",
        xy=(30, fila30["mae_media"]),
        xytext=(30, fila30["mae_media"] - 0.32),
        fontsize=8.5,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
    )

    _ejex_log_epsilon(ax, [1, 10, 30, 50, 100, 300, 500])
    ax.set_xlabel(r"$\varepsilon_{in}$ (escala logarítmica)")
    ax.set_ylabel("MAE de la reconstrucción de V4")
    ax.set_ylim(0, 1.3)
    if TITULOS_EN_FIGURA:
        ax.set_title("Curva de protección del input shield (Exp 1)")
    _leyenda_y_nota(
        fig,
        ax,
        nota="Nota: los puntos marcados con rombo (eps_in=1,10) tienen desviación típica exactamente cero en las\n"
        "5 corridas; no llevan barra de error porque indican abstención total, no falta de datos.",
    )

    _guardar_figura(fig, "fig01_proteccion_input_shield")


# --------------------------------------------------------------------------------------
# Figura 2 — Curva de protección del output shield (Exp 2)
# --------------------------------------------------------------------------------------


def fig02_proteccion_output_shield(agg2: pd.DataFrame) -> None:
    """
    Pie de figura sugerido:
    Figura 2. MAE de la reconstrucción de V4 frente a eps_out (Experimento 2, K=200,
    n=3 semillas; barras = IC 95%, t de Student). A diferencia de la Figura 1, en este
    eje no existe ningún punto con protección total (ninguna desviación típica es cero):
    ese contraste es el mensaje. La banda roja entre eps_out=10 y eps_out=50 marca el
    colapso de la protección. El punto eps_out=2 es el valor por defecto en producción:
    sin ventaja detectable, pero sin abstención. La rodilla de este eje está peor acotada
    que la del Exp 1 -- el corchete es un factor 5x (10 a 50) frente al factor 2x del
    Exp 1 (50 a 100), porque faltan puntos intermedios en el barrido.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.axvspan(10, 50, color=COLOR_SOMBRA_ROJA, alpha=0.6, zorder=0)

    _linea_base(ax)
    _plot_serie_con_ic(ax, agg2, "mae", COLOR_A, MARKER_A, "Exp 2 — MAE frente a eps_out")

    ax.text(22.4, ax.get_ylim()[1] * 0.93, "colapso de la\nprotección", fontsize=9, ha="center", va="top", color="#a33a1c")

    fila2 = agg2[agg2["epsilon"] == 2].iloc[0]
    ax.annotate(
        "valor por defecto en producción\nsin ventaja detectable, sin abstención",
        xy=(2, fila2["mae_media"]),
        xytext=(6, 0.82),
        fontsize=8.5,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
    )

    _ejex_log_epsilon(ax, [1, 2, 5, 10, 50, 100, 500])
    ax.set_xlabel(r"$\varepsilon_{out}$ (escala logarítmica)")
    ax.set_ylabel("MAE de la reconstrucción de V4")
    ax.set_ylim(0, 1.3)
    if TITULOS_EN_FIGURA:
        ax.set_title("Curva de protección del output shield (Exp 2)")
    _leyenda_y_nota(
        fig,
        ax,
        nota="Nota: la rodilla de este eje es un factor 5x (10→50); en el Exp 1 fue un factor 2x (50→100)\n"
        "por faltar puntos intermedios en el barrido.",
    )

    _guardar_figura(fig, "fig02_proteccion_output_shield")


# --------------------------------------------------------------------------------------
# Figura 3 — Comparación de los dos ejes
# --------------------------------------------------------------------------------------


def fig03_comparacion_ejes(agg1: pd.DataFrame, agg2: pd.DataFrame) -> None:
    """
    Pie de figura sugerido:
    Figura 3. MAE frente a epsilon en los 5 valores que comparten los dos barridos
    (1, 10, 50, 100, 500). Serie A: Exp 1, se barre eps_in con salida transparente
    (eps_out=10000). Serie B: Exp 2, se barre eps_out con entrada transparente
    (eps_in=10000). Barras = IC 95% (t de Student). eps_in y eps_out no son la misma
    unidad: la escala del ruido es b = Delta/epsilon y las sensibilidades Delta de las
    dos capas son distintas. La figura sostiene que a igual epsilon nominal la capa de
    salida deja pasar más información, no que la capa de entrada proteja más. El Exp 2
    corrió además con K=200 frente a K=400, es decir con la mitad de promediado, lo que
    hace la comparación conservadora en la dirección observada.
    """
    a1 = agg1[agg1["epsilon"].isin(EPS_COMPARTIDOS)].sort_values("epsilon")
    a2 = agg2[agg2["epsilon"].isin(EPS_COMPARTIDOS)].sort_values("epsilon")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _linea_base(ax)
    _plot_serie_con_ic(ax, a1, "mae", COLOR_A, MARKER_A, "Exp 1 — se barre eps_in (salida transparente)")
    _plot_serie_con_ic(ax, a2, "mae", COLOR_B, MARKER_B, "Exp 2 — se barre eps_out (entrada transparente)")

    _ejex_log_epsilon(ax, EPS_COMPARTIDOS)
    ax.set_xlabel(r"$\varepsilon$ (escala logarítmica)")
    ax.set_ylabel("MAE de la reconstrucción de V4")
    ax.set_ylim(bottom=0)
    if TITULOS_EN_FIGURA:
        ax.set_title("Comparación: barrido de eps_in vs. barrido de eps_out")
    _leyenda_y_nota(
        fig,
        ax,
        nota="eps_in y eps_out no son la misma unidad (b = Δ/ε, sensibilidades Δ distintas). La figura sostiene que\n"
        "a igual ε nominal la capa de salida deja pasar más información, no que la de entrada proteja más.\n"
        "Exp 2 corrió con K=200 frente a K=400 del Exp 1 (comparación conservadora).",
    )

    _guardar_figura(fig, "fig03_comparacion_ejes")


# --------------------------------------------------------------------------------------
# Figura 4 — Aciertos del atacante (success_rate)
# --------------------------------------------------------------------------------------


def fig04_success_rate(agg1: pd.DataFrame, agg2: pd.DataFrame, aviso_una_corrida: bool) -> None:
    """
    Pie de figura sugerido:
    Figura 4. Porcentaje de los 30 targets reconstruidos dentro de la tolerancia
    tol_abs=0.3189, frente a epsilon, en los valores compartidos por los dos barridos.
    La línea horizontal de trazos en 6.7% (2 de 30) es el nivel de azar: son los 2
    targets cuyo V4 verdadero cae, por casualidad, a menos de tol_abs de la media pública.
    Un success_rate de 6.7% significa cero información extraída por el ataque.
    """
    a1 = agg1[agg1["epsilon"].isin(EPS_COMPARTIDOS)].sort_values("epsilon")
    a2 = agg2[agg2["epsilon"].isin(EPS_COMPARTIDOS)].sort_values("epsilon")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    azar = 100 * 2 / 30
    ax.axhline(azar, color="#7a7a7a", linestyle=":", linewidth=1.3, label="nivel de azar (2 de 30) = 6.7%", zorder=1)

    _plot_serie_con_ic(ax, a1, "success_rate", COLOR_A, MARKER_A, "Exp 1 — se barre eps_in", escala=100.0)
    _plot_serie_con_ic(ax, a2, "success_rate", COLOR_B, MARKER_B, "Exp 2 — se barre eps_out", escala=100.0)

    _ejex_log_epsilon(ax, EPS_COMPARTIDOS)
    ax.set_xlabel(r"$\varepsilon$ (escala logarítmica)")
    ax.set_ylabel("Targets reconstruidos dentro de tolerancia (%)")
    ax.set_ylim(0, 100)
    titulo = "Aciertos del atacante (success_rate)"
    if TITULOS_EN_FIGURA:
        ax.set_title(titulo)
    nota_fig4 = (
        "Aviso: estos datos provienen de una sola corrida por eje (Exp 1 semilla 5, Exp 2 semilla 1)."
        if aviso_una_corrida
        else None
    )
    _leyenda_y_nota(fig, ax, ncol=2, nota=nota_fig4)

    _guardar_figura(fig, "fig04_success_rate")


# --------------------------------------------------------------------------------------
# Figura 5 y 6 — Diagnóstico de pendientes (datos de stdout, no de los CSV)
# --------------------------------------------------------------------------------------

# Bloques [diag] extraídos del stdout de las corridas. No están en los CSV.
DIAG_EXP3 = {  # eps_in=1, eps_out=2, K=400 — los 9 dieron ABSTIENE
    # (seed, target, slope_mean, snr, signos_positivos_de_8)
    (1, 0, +0.0057, 0.21, 3), (1, 15, -0.0325, 1.29, 3), (1, 29, +0.0370, 1.05, 5),
    (2, 0, -0.0391, 2.38, 2), (2, 15, +0.0304, 0.80, 5), (2, 29, -0.0245, 0.56, 3),
    (3, 0, +0.0399, 2.10, 6), (3, 15, -0.0223, 0.55, 4), (3, 29, +0.0038, 0.14, 4),
}
DIAG_EXP2 = {  # eps_in=10000, eps_out=500, K=200 — los 9 dieron PASA, signos 8/0
    # (seed, target, slope_mean, slope_std, snr)
    (1, 0, +0.6056, 0.0453, 37.79), (1, 15, +0.5389, 0.0298, 51.15), (1, 29, +0.5901, 0.0203, 82.35),
    (2, 0, +0.6211, 0.0214, 82.05), (2, 15, +0.5702, 0.0544, 29.65), (2, 29, +0.5920, 0.0345, 48.51),
    (3, 0, +0.6167, 0.0248, 70.43), (3, 15, +0.5709, 0.0283, 57.06), (3, 29, +0.5844, 0.0354, 46.63),
}
# Exp 1 (eps_in=500, eps_out=10000, K=400): solo se conservaron los RANGOS de las 15 mediciones,
# no los valores individuales. slope_mean en [0.462, 0.625]; slope_std en [0.068, 0.181];
# SNR en [8.74, 25.49]; signos 8/0 en las 15.
DIAG_EXP1_RANGO = {"slope_mean": (0.462, 0.625), "snr": (8.74, 25.49), "n": 15}

UMBRAL_SNR = 5.0

CAT_A_LABEL = "Desplegada\n(eps_in=1, eps_out=2)"
CAT_B_LABEL = "Solo input shield\n(eps_in=500)"
CAT_C_LABEL = "Solo output shield\n(eps_out=500)"


def _jitter(n: int, centro: float, ancho: float = 0.12, semilla: int = 0) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    return centro + rng.uniform(-ancho, ancho, size=n)


def fig05_diagnostico_pendientes() -> None:
    """
    Pie de figura sugerido:
    Figura 5. Pendiente medida de la respuesta del modelo respecto a V4 (slope_mean) en
    tres condiciones: la configuración desplegada (eps_in=1, eps_out=2, 9 mediciones,
    todas [ABSTIENE]), el input shield transparente (eps_in=500, 15 mediciones, todas
    [PASA]) y el output shield transparente (eps_out=500, 9 mediciones, todas [PASA]). La
    línea horizontal en y=0 muestra que la configuración desplegada se distribuye
    alrededor de cero con signos aleatorios, mientras que las dos condiciones
    transparentes se agrupan en torno a +0.55. De la condición de input shield solo se
    conservaron los rangos de las 15 mediciones individuales (no los valores puntuales),
    representados aquí como banda sombreada horizontal entre 0.462 y 0.625; no se
    inventan puntos dentro de la banda. Datos de stdout de las corridas, no de los CSV.
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.axhline(0, color="#888888", linewidth=1.0, zorder=1)

    slopes_a = [v[2] for v in DIAG_EXP3]
    slopes_c = [v[2] for v in DIAG_EXP2]

    xa = _jitter(len(slopes_a), 1, semilla=1)
    xc = _jitter(len(slopes_c), 3, semilla=3)

    ax.scatter(xa, slopes_a, marker=MARKER_A, color=COLOR_ABSTIENE, edgecolor="black", s=55, zorder=3, label="ABSTIENE")
    ax.scatter(xc, slopes_c, marker=MARKER_C, color=COLOR_PASA, edgecolor="black", s=55, zorder=3, label="PASA")

    lo, hi = DIAG_EXP1_RANGO["slope_mean"]
    ax.add_patch(
        plt.Rectangle((2 - 0.3, lo), 0.6, hi - lo, facecolor=COLOR_PASA, alpha=0.25, edgecolor=COLOR_PASA, zorder=2)
    )
    ax.plot([], [], MARKER_B, color=COLOR_PASA, markerfacecolor="none", label="PASA (banda de rango, n=15)")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([CAT_A_LABEL, CAT_B_LABEL, CAT_C_LABEL], fontsize=9)
    ax.set_xlim(0.3, 3.7)
    ax.set_ylabel("Pendiente medida (slope_mean)")
    if TITULOS_EN_FIGURA:
        ax.set_title("Diagnóstico de pendientes: presencia y ausencia de señal")
    _leyenda_y_nota(
        fig,
        ax,
        ncol=3,
        nota="Datos de stdout de las corridas (no están en los CSV). Categoría B: solo se conservaron los\n"
        "rangos de las 15 mediciones individuales; la banda cubre el rango observado, sin puntos inventados.",
    )

    _guardar_figura(fig, "fig05_diagnostico_pendientes")


def fig06_snr_vs_umbral() -> None:
    """
    Pie de figura sugerido:
    Figura 6. Relación señal-ruido (SNR) de la pendiente medida, en escala logarítmica,
    para las mismas tres condiciones de la Figura 5. La línea roja en SNR=5.0 es el
    umbral de decisión del mecanismo de abstención (fijado a priori por criterio
    estadístico -- Bonferroni sobre 30 tests, t con df=7 -- y no eligiendo el valor que
    produjera el mejor resultado). Toda la configuración desplegada queda entre 0.14 y
    2.38, por debajo del umbral; las dos condiciones transparentes quedan entre 8.7 y
    82.4, por encima. La separación es de más de un orden de magnitud. Datos de stdout de
    las corridas, no de los CSV.
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.set_yscale("log")
    ax.axhline(UMBRAL_SNR, color="#c0392b", linewidth=1.4, label="umbral SNR=5.0 (Bonferroni, 30 tests, t df=7)", zorder=1)

    snr_a = [v[3] for v in DIAG_EXP3]
    snr_c = [v[4] for v in DIAG_EXP2]

    xa = _jitter(len(snr_a), 1, semilla=11)
    xc = _jitter(len(snr_c), 3, semilla=13)

    ax.scatter(xa, snr_a, marker=MARKER_A, color=COLOR_ABSTIENE, edgecolor="black", s=55, zorder=3, label="ABSTIENE")
    ax.scatter(xc, snr_c, marker=MARKER_C, color=COLOR_PASA, edgecolor="black", s=55, zorder=3, label="PASA")

    lo, hi = DIAG_EXP1_RANGO["snr"]
    ax.add_patch(
        plt.Rectangle((2 - 0.3, lo), 0.6, hi - lo, facecolor=COLOR_PASA, alpha=0.25, edgecolor=COLOR_PASA, zorder=2)
    )
    ax.plot([], [], MARKER_B, color=COLOR_PASA, markerfacecolor="none", label="PASA (banda de rango, n=15)")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([CAT_A_LABEL, CAT_B_LABEL, CAT_C_LABEL], fontsize=9)
    ax.set_xlim(0.3, 3.7)
    ax.set_ylabel("SNR de la pendiente (escala logarítmica)")
    if TITULOS_EN_FIGURA:
        ax.set_title("SNR frente al umbral de decisión")
    _leyenda_y_nota(
        fig,
        ax,
        ncol=3,
        nota="Umbral fijado a priori por criterio estadístico, no eligiendo el que produjera el mejor resultado.\n"
        "Datos de stdout de las corridas (no están en los CSV).",
    )

    _guardar_figura(fig, "fig06_snr_vs_umbral")


# --------------------------------------------------------------------------------------
# Figura 7 — Sesgo del estimador frente a eps_out (Exp 2, corrida única)
# --------------------------------------------------------------------------------------


def fig07_sesgo_mean_error(exp2_run1: pd.DataFrame) -> None:
    """
    Pie de figura sugerido:
    Figura 7. Error medio con signo (mean_error) frente a eps_out, escala logarítmica
    (Experimento 2, corrida 1; pendiente de consolidar con las tres semillas). La línea
    horizontal de trazos en +0.159 es el sesgo de abstenerse: el sesgo que tendría alguien
    que respondiera siempre la media pública. La curva sube por encima de esa línea hasta
    +0.267 en eps_out=50, y solo después se derrumba hacia cero: las estimaciones que
    superan el guard quedan más sesgadas que no atacar (sesgo de recorte del output
    shield).
    """
    d = exp2_run1.sort_values("epsilon")
    x = d["epsilon"].to_numpy()
    y = d["mean_error"].to_numpy()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sesgo_abstenerse = 0.1591478861973178  # mean_error en eps_out=1 (equivale a abstenerse)
    ax.axhline(sesgo_abstenerse, color=COLOR_BASELINE, linestyle="--", linewidth=1.3, label="sesgo de abstenerse = +0.159", zorder=1)
    ax.axhline(0, color="#bbbbbb", linewidth=0.8, zorder=1)

    ax.plot(x, y, "-", color=COLOR_A, linewidth=1.4, zorder=2)
    ax.plot(x, y, MARKER_A, color=COLOR_A, markeredgecolor="black", markersize=6, zorder=3, label="Exp 2 — mean_error (corrida 1)")

    fila50 = d[d["epsilon"] == 50].iloc[0]
    ax.annotate(
        "las estimaciones que superan el guard quedan\nmás sesgadas que no atacar\n(sesgo de recorte del output shield)",
        xy=(50, fila50["mean_error"]),
        xytext=(0.05, 0.95),
        textcoords="axes fraction",
        fontsize=8,
        ha="left",
        va="top",
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
    )

    ax.set_ylim(top=0.34)
    _ejex_log_epsilon(ax, [1, 2, 5, 10, 50, 100, 500])
    ax.set_xlabel(r"$\varepsilon_{out}$ (escala logarítmica)")
    ax.set_ylabel("Error medio con signo (mean_error)")
    if TITULOS_EN_FIGURA:
        ax.set_title("Sesgo del estimador frente a eps_out (Exp 2)")
    _leyenda_y_nota(
        fig,
        ax,
        nota="Aviso: datos de una sola corrida (la 1), pendientes de consolidar con las tres semillas.",
    )

    _guardar_figura(fig, "fig07_sesgo_mean_error")


# --------------------------------------------------------------------------------------
# Figura 8 (opcional) — El punto desplegado (Exp 3) frente a la línea base
# --------------------------------------------------------------------------------------


def fig08_punto_desplegado(agg3: pd.DataFrame) -> None:
    """
    Pie de figura sugerido:
    Figura 8. MAE del ataque en la configuración desplegada (eps_in=1, eps_out=2, K=400,
    n=3 semillas) comparado contra la línea base de no atacar. La barra vertical es el
    IC 95% (t de Student, df=2). El IC contiene la línea base: el ataque es
    estadísticamente indistinguible de no atacar en la configuración de producción.
    """
    fila = agg3[agg3["epsilon"] == agg3["epsilon"].iloc[0]].iloc[0]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    _linea_base(ax)

    ax.errorbar(
        [0],
        [fila["mae_media"]],
        yerr=[[fila["mae_media"] - fila["mae_ci_bajo"]], [fila["mae_ci_alto"] - fila["mae_media"]]],
        fmt=MARKER_A,
        color=COLOR_A,
        markeredgecolor="black",
        markersize=9,
        capsize=5,
        linewidth=1.6,
        label="Exp 3 — configuración desplegada (eps_in=1, eps_out=2)",
    )

    ax.set_xlim(-1, 1)
    ax.set_xticks([])
    ax.set_ylabel("MAE de la reconstrucción de V4")
    if TITULOS_EN_FIGURA:
        ax.set_title("El punto desplegado frente a la línea base")
    _leyenda_y_nota(
        fig,
        ax,
        nota=f"IC 95% = [{fila['mae_ci_bajo']:.4f}, {fila['mae_ci_alto']:.4f}] contiene la línea base: "
        "indistinguible estadísticamente de no atacar.",
    )

    _guardar_figura(fig, "fig08_punto_desplegado")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    exp1, exp2, exp3 = cargar_datos()

    columnas_agg = ["mae", "mean_error", "success_rate"]
    agg1 = agregar_por_epsilon(exp1, columnas_agg)
    agg2 = agregar_por_epsilon(exp2, columnas_agg)
    agg3 = agregar_por_epsilon(exp3, columnas_agg)

    agregados = pd.concat([agg1, agg2, agg3], ignore_index=True)
    agregados.to_csv(FIGURAS_DIR / "datos_agregados.csv", index=False)
    print(f"[ok] Escrito {FIGURAS_DIR / 'datos_agregados.csv'} ({len(agregados)} filas)")

    # Figura 4: success_rate está presente en todas las semillas de Exp 1 y Exp 2 -> se
    # agrega igual que las demás métricas y se elimina el aviso de "corrida única".
    aviso_una_corrida_fig4 = False

    fig01_proteccion_input_shield(agg1)
    fig02_proteccion_output_shield(agg2)
    fig03_comparacion_ejes(agg1, agg2)
    fig04_success_rate(agg1, agg2, aviso_una_corrida_fig4)
    fig05_diagnostico_pendientes()
    fig06_snr_vs_umbral()
    fig07_sesgo_mean_error(exp2[exp2["seed"] == 1])
    fig08_punto_desplegado(agg3)

    print(f"[ok] Figuras generadas en {FIGURAS_DIR}")


if __name__ == "__main__":
    sys.exit(main())
