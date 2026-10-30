"""CLI de validación: ejecuta los niveles 2 y/o 3 y escribe JSON/CSV.

Uso (desde privacy-api/):

    python -m validacion.run_validacion --nivel todos --T 2000 --alpha 0.01 \\
        --seed 1 --feature V4 --out results/validacion/

Una ruta --out relativa se interpreta respecto a la raíz del repositorio.
Código de salida: 0 si todo es compatible y los controles se comportan como se
espera; 1 si hay VIOLACIÓN o un control falla (hallazgo, no se ajusta el test).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import analitica, distinguibilidad
from .mecanismo import FEATURE_BOUNDS, N_FEATURES, PRIVACY_API_DIR

REPO_ROOT = PRIVACY_API_DIR.parent
OUT_POR_DEFECTO = REPO_ROOT / "results" / "validacion"


# --------------------------------------------------------------------------
# Nivel 2
# --------------------------------------------------------------------------

def ejecutar_nivel2(
    feature: str,
    eps_in_barrido: Sequence[float],
    eps_out_barrido: Sequence[float],
    k_consultas: int = 100,
) -> Dict:
    low, high = FEATURE_BOUNDS[feature]
    delta = high - low
    v = 0.5 * (low + high)
    escalas: List[Dict] = []
    for eps_in in eps_in_barrido:
        b = analitica.escala_entrada(eps_in, delta=delta, n=N_FEATURES)
        a = analitica.atenuacion(v, b, low, high)
        escalas.append({
            "capa": "entrada", "feature": feature, "eps": eps_in,
            "eps_por_feature": analitica.eps_por_feature(eps_in, N_FEATURES),
            "delta": delta, "b": b, "v_referencia": v, "atenuacion": a,
            "consultas_necesarias": analitica.consultas_necesarias(a),
            f"composicion_{k_consultas}_consultas": analitica.composicion_basica(eps_in, k_consultas),
        })
    for eps_out in eps_out_barrido:
        escalas.append({
            "capa": "salida", "feature": None, "eps": eps_out, "eps_por_feature": None,
            "delta": analitica.DELTA_SALIDA, "b": analitica.escala_salida(eps_out),
            "v_referencia": None, "atenuacion": None, "consultas_necesarias": None,
            f"composicion_{k_consultas}_consultas": analitica.composicion_basica(eps_out, k_consultas),
        })
    garantias = [
        {
            "eps_in": ei, "eps_out": eo,
            "garantia_registro": analitica.garantia_por_consulta(ei, eo, "registro"),
            "garantia_una_feature": analitica.garantia_por_consulta(ei, eo, "una_feature"),
        }
        for ei in eps_in_barrido for eo in eps_out_barrido
    ]
    return {"escalas": escalas, "garantias": garantias, "k_consultas": k_consultas}


# --------------------------------------------------------------------------
# Escritura
# --------------------------------------------------------------------------

def _limpiar(obj):
    """Convierte numpy/inf/nan a tipos JSON válidos (inf/nan → None)."""
    if isinstance(obj, dict):
        return {k: _limpiar(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_limpiar(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _escribir_json(ruta: Path, datos: Dict) -> None:
    ruta.write_text(json.dumps(_limpiar(datos), indent=2, ensure_ascii=False), encoding="utf-8")


def _escribir_csv(ruta: Path, filas: List[Dict]) -> None:
    columnas: List[str] = []
    for fila in filas:
        for k in fila:
            if k not in columnas:
                columnas.append(k)
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas)
        w.writeheader()
        for fila in filas:
            w.writerow(_limpiar(fila))


def _tabla(filas: List[Dict]) -> str:
    cab = ["capa", "tipo", "eps_teo", "p0_hat", "p1_hat", "eps_lb", "techo", "veredicto", "control", "t(s)"]
    rows = [cab]
    for f in filas:
        rows.append([
            f["capa"], f["tipo"], f"{f['eps_teorico']:.3g}", f"{f['p0_hat']:.4f}",
            f"{f['p1_hat']:.4f}", f"{f['eps_lb']:.3f}", f"{f['techo']:.3f}",
            f["veredicto"], {True: "OK", False: "HALLAZGO", None: "-"}[f.get("control_ok")],
            f"{f['tiempo_s']:.2f}",
        ])
    anchos = [max(len(r[i]) for r in rows) for i in range(len(cab))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, anchos)) for r in rows)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def _lista(texto: str) -> List[float]:
    return [float(x) for x in texto.split(",") if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--nivel", choices=["2", "3", "todos"], default="todos")
    p.add_argument("--T", type=int, default=2000, help="consultas por entrada vecina")
    p.add_argument("--alpha", type=float, default=0.01, help="nivel global (se reparte α/2 por proporción)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--feature", default="V4", help="feature auditada (clave de FEATURE_BOUNDS)")
    p.add_argument("--out", default=str(OUT_POR_DEFECTO))
    p.add_argument("--eps-in", type=_lista, default=list(distinguibilidad.BARRIDO_EPS_IN),
                   help="barrido de ε_in total, separado por comas")
    p.add_argument("--eps-out", type=_lista, default=list(distinguibilidad.BARRIDO_EPS_OUT),
                   help="barrido de ε_out, separado por comas")
    p.add_argument("--e2e-url", default=None,
                   help="opcional: URL de privacy-api (mode=legacy) para una corrida extremo a extremo")
    p.add_argument("--e2e-eps-in", type=float, default=45.0)
    p.add_argument("--e2e-eps-out", type=float, default=2.0)
    args = p.parse_args(argv)

    if args.feature not in FEATURE_BOUNDS:
        p.error(f"--feature debe estar en {sorted(FEATURE_BOUNDS)}")
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    inicio = time.perf_counter()
    exit_code = 0
    meta = {"T": args.T, "alpha": args.alpha, "seed": args.seed, "feature": args.feature}

    if args.nivel in ("2", "todos"):
        n2 = ejecutar_nivel2(args.feature, args.eps_in, args.eps_out)
        _escribir_json(out / "nivel2_analitica.json", {**meta, **n2})
        _escribir_csv(out / "nivel2_analitica.csv", n2["escalas"])
        _escribir_csv(out / "nivel2_garantias.csv", n2["garantias"])
        print("== Nivel 2: escalas analíticas ==")
        for e in n2["escalas"]:
            a = "" if e["atenuacion"] is None else f"  a={e['atenuacion']:.4f}"
            print(f"  {e['capa']:<8} eps={e['eps']:<6g} b={e['b']:<10.4g}{a}")

    if args.nivel in ("3", "todos"):
        n3 = distinguibilidad.ejecutar_auditoria(
            args.T, args.alpha, args.seed, args.feature, args.eps_in, args.eps_out,
            progreso=None,
        )
        filas = n3["filas"]
        if args.e2e_url:
            filas.append(distinguibilidad.auditar_e2e(
                args.e2e_url, args.feature, args.e2e_eps_in, args.e2e_eps_out,
                args.T, args.alpha, args.seed,
            ))
        _escribir_json(out / "nivel3_distinguibilidad.json", {**n3, "filas": filas})
        _escribir_csv(out / "nivel3_distinguibilidad.csv", filas)
        print(f"\n== Nivel 3: distinguibilidad (T={args.T}, alpha={args.alpha}, "
              f"techo eps_lb={n3['techo']:.3f}) ==")
        print(_tabla(filas))
        violaciones = [f for f in filas if f["tipo"] in ("barrido", "e2e") and f["veredicto"] == distinguibilidad.VIOLACION]
        fallos = [f for f in filas if f.get("control_ok") is False]
        if violaciones:
            print(f"\n[HALLAZGO] {len(violaciones)} auditoría(s) con VIOLACIÓN: el mecanismo supera su ε teórico.")
        for f in fallos:
            print(f"[HALLAZGO] control {f['tipo']} ({f['capa']}, eps={f['eps_param']}): {f['control_detalle']}")
        if violaciones or fallos:
            exit_code = 1

    print(f"\nResultados en {out} ({time.perf_counter() - inicio:.1f} s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
