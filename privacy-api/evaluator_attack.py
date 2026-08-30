"""Harness investigador para el ataque de inversión inter-request.

Cambios problema-4 (probes limpios):
  - Los probes ya NO se derivan del secreto de las transacciones objetivo.
    Se calculan sobre una MUESTRA PÚBLICA DISJUNTA (pool), separada de los
    targets. Supuesto de amenaza estándar en ataques de inversión: el atacante
    dispone de una muestra auxiliar de la misma distribución, pero NO conoce el
    secreto de la víctima concreta.
  - Se usan PERCENTILES (p5/p95 por defecto) en vez de min/max: robusto a
    outliers y evita anclar los probes en el recorte ±5 del input shield, donde
    la pendiente lineal se distorsiona.
  - La tolerancia de success_rate se normaliza por feature (fracción de la
    desviación pública del secreto), no un 0.5 fijo en unidades crudas.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from attacks.api_client import HttpAttackClient
from attacks.inversion_attack import denoise, invert_feature
from attacks.metrics_attack import build_curve


# Rango amplio: los valores bajos (1, 10) muestran la zona saturada/protegida;
# los altos (>=100) llevan el ruido al tamaño de la feature para ver la rotura.
# Graficar con eje epsilon logarítmico.
DEFAULT_EPSILONS = (1.0, 10.0, 30.0, 50.0, 100.0, 300.0, 500.0)
PUBLIC_FIELDS = ["Time"] + [f"V{i}" for i in range(1, 15)] + ["Amount"]

# Rango público de recorte de cada feature (el atacante lo conoce: es metadato
# publicado por /privacy-info). Sirve para acotar la estimación a valores
# posibles: sin esto, cuando la pendiente medida es casi cero, la división
# explota a valores absurdos (p. ej. 125 para una V que vive en [-5, 5]).
PUBLIC_BOUNDS = {f"V{i}": (-5.0, 5.0) for i in range(1, 15)}
PUBLIC_BOUNDS["Amount"] = (0.0, 25000.0)


def load_transactions(dataset_path: Path, limit: int) -> List[dict]:
    with dataset_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))[:limit]
    return [{field: float(row[field]) for field in PUBLIC_FIELDS} for row in rows]


def split_public_targets(
    transactions: List[dict], public_size: int
) -> Tuple[List[dict], List[dict]]:
    """Parte el conjunto en (pool público, targets) DISJUNTOS.

    El pool alimenta los probes; los targets son las víctimas a invertir.
    """
    if public_size < 1:
        raise ValueError("public_size debe ser >= 1")
    if public_size >= len(transactions):
        raise ValueError(
            f"public_size={public_size} deja 0 targets "
            f"(hay {len(transactions)} transacciones). Baja --public-size o sube --limit."
        )
    return transactions[:public_size], transactions[public_size:]


def public_probe_range(
    pool: List[dict],
    secret_name: str,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
) -> Tuple[float, float]:
    """Probes desde percentiles del secreto en la muestra pública (disjunta)."""
    values = np.array([row[secret_name] for row in pool], dtype=float)
    if values.size == 0:
        raise ValueError("Pool público vacío: no hay con qué calcular probes")
    lo, hi = np.percentile(values, [lo_pct, hi_pct])
    if hi == lo:  # degenerado: ensancha con la desviación
        spread = float(values.std()) or 1.0
        lo, hi = lo - spread, hi + spread
    return float(lo), float(hi)


def secret_scale(pool: List[dict], secret_name: str) -> float:
    """Desviación pública del secreto, para normalizar la tolerancia."""
    values = np.array([row[secret_name] for row in pool], dtype=float)
    std = float(values.std())
    return std if std > 0 else 1.0


def run_experiment(
    base_url: str,
    pool: List[dict],
    targets: List[dict],
    secret_name: str = "V4",
    K: int = 20,
    epsilons=DEFAULT_EPSILONS,
    tol_frac: float = 0.25,
    epsilon_in_fixed: float = 2,
) -> List[dict]:
    probe_lo, probe_hi = public_probe_range(pool, secret_name)
    tol_abs = tol_frac * secret_scale(pool, secret_name)

    # Conocimiento público del atacante: rango válido y valor "por defecto"
    # (la media del pool) al que recurre cuando no puede estimar nada.
    lo, hi = PUBLIC_BOUNDS.get(secret_name, (probe_lo, probe_hi))
    public_mean = float(np.mean([row[secret_name] for row in pool]))

    # Línea base: error de NO atacar, solo adivinar la media pública.
    # Es la vara contra la cual se juzga el ataque: si el ataque no baja de
    # aquí, no recuperó nada útil.
    baseline_errors = [public_mean - t[secret_name] for t in targets]
    baseline_mae = float(np.mean(np.abs(baseline_errors))) if baseline_errors else 0.0

    results: Dict[float, List[float]] = {}
    for epsilon in epsilons:
        # epsilon_in_fixed desactiva (o fija) el input shield; epsilon barre el output shield
        client = HttpAttackClient(base_url, epsilon_in=epsilon_in_fixed, epsilon_out=epsilon)
        print(f"  [barrido] epsilon_in={epsilon_in_fixed}, epsilon_out={epsilon}...", flush=True)
        errors = []
        for target in targets:
            known = {k: v for k, v in target.items() if k != secret_name}
            p_target = denoise(client, target, K)
            estimate = invert_feature(
                client, known, secret_name, probe_lo, probe_hi, K, p_target
            )
            if estimate is None:
                # pendiente indistinguible de cero → el atacante no puede
                # estimar: recurre a la media pública (su mejor apuesta ciega).
                estimate = public_mean
            else:
                # acota a valores posibles: evita las explosiones numéricas.
                estimate = min(hi, max(lo, estimate))
            errors.append(estimate - target[secret_name])
        results[epsilon] = errors

    curve = build_curve(results, tol=tol_abs)
    # anexa la línea base a cada fila para poder compararla de un vistazo
    for row in curve:
        row["baseline_mae"] = baseline_mae
        row["beats_baseline"] = row["mae"] < baseline_mae
    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--dataset", type=Path,
                        default=Path(__file__).parent.parent / "modelo-python" / "dataset" / "creditcard.csv")
    parser.add_argument("--secret", default="V4")
    parser.add_argument("--limit", type=int, default=220,
                        help="Total de filas a cargar (pool + targets).")
    parser.add_argument("--public-size", type=int, default=200,
                        help="Filas reservadas al pool público para los probes.")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPSILONS),
                        help="Lista de epsilon_in a barrer, separados por espacio.")
    parser.add_argument("--tol-frac", type=float, default=0.25,
                        help="Tolerancia de éxito como fracción de la std pública del secreto.")
    parser.add_argument("--epsilon-in", type=float, default=1.0,
                        help="Epsilon fijo del input shield durante el barrido (default=1.0, valor del governor actual).")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    transactions = load_transactions(args.dataset, args.limit)
    pool, targets = split_public_targets(transactions, args.public_size)
    curve = run_experiment(
        args.base_url, pool, targets, args.secret, args.K,
        epsilons=args.epsilons, tol_frac=args.tol_frac,
        epsilon_in_fixed=args.epsilon_in,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inversion_curve.json").write_text(
        json.dumps(curve, indent=2), encoding="utf-8"
    )
    fieldnames = list(curve[0].keys()) if curve else [
        "epsilon_out", "mae", "rmse", "mean_error", "n", "success_rate", "tol_abs"
    ]
    with (args.output_dir / "inversion_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve)

    print(f"pool={len(pool)}  targets={len(targets)}  secret={args.secret}  K={args.K}  epsilon_in_fijo={args.epsilon_in}")
    if curve:
        print(f"linea_base (adivinar la media) MAE = {curve[0]['baseline_mae']:.4f}")
        print("  -> el ataque solo 'gana' en las filas donde MAE < linea_base\n")
    print("epsilon_out\tMAE_ataque\t¿gana?")
    for row in curve:
        verdict = "SI" if row["beats_baseline"] else "no"
        print(f"{row['epsilon_out']:>8.0f}\t{row['mae']:.4f}\t\t{verdict}")


if __name__ == "__main__":
    main()