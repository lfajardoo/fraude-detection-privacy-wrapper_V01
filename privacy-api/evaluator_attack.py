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
from attacks.inversion_attack import _logit, denoise, invert_feature
from attacks.metrics_attack import build_bias_variance_curve, build_curve


# Rango amplio: los valores bajos (1, 10) muestran la zona saturada/protegida;
# los altos (>=100) llevan el ruido al tamaño de la feature para ver la rotura.
# Graficar con eje epsilon logarítmico.
DEFAULT_EPSILONS = (1.0, 10.0, 30.0, 50.0, 100.0, 300.0, 500.0)
# Experimento 2 (salida aislada): barrido de epsilon_out con el input transparente.
# Incluye 2.0 (default del governor) para leer el punto desplegado directamente.
DEFAULT_EPSILONS_OUT = (1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0)
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
    epsilons_in=DEFAULT_EPSILONS,
    tol_frac: float = 0.25,
    epsilon_out_fixed: float = 50.0,
    R: int = 8,
    snr_threshold: float = 2.0,
    probe_lo_pct: float = 25.0,
    probe_hi_pct: float = 75.0,
) -> List[dict]:
    # CAMINO 2: span de sondeo ajustable por percentiles.  Percentiles menos
    # extremos (25/75 en vez de 5/95) mantienen probe_lo y probe_hi en la zona
    # lineal del logit, donde p no está tan cerca de 0/1 que el logit se dispare.
    probe_lo, probe_hi = public_probe_range(
        pool, secret_name, lo_pct=probe_lo_pct, hi_pct=probe_hi_pct
    )
    tol_abs = tol_frac * secret_scale(pool, secret_name)
    lo, hi = PUBLIC_BOUNDS.get(secret_name, (probe_lo, probe_hi))
    public_mean = float(np.mean([row[secret_name] for row in pool]))
    baseline_errors = [public_mean - t[secret_name] for t in targets]
    baseline_mae = float(np.mean(np.abs(baseline_errors))) if baseline_errors else 0.0

    print(f"  [probes] {secret_name} en p{probe_lo_pct:.0f}/p{probe_hi_pct:.0f} = "
          f"[{probe_lo:.4f}, {probe_hi:.4f}]  (span={probe_hi - probe_lo:.4f})  R={R}")

        # --- Diagnóstico del guard: reproduce las R pendientes que evalúa invert_feature ---
    if targets:
        diag_eps_in = max(epsilons_in)
        diag_client = HttpAttackClient(
            base_url, epsilon_in=diag_eps_in, epsilon_out=epsilon_out_fixed
        )
        print(f"  [diag] ε_in={diag_eps_in:g}, ε_out={epsilon_out_fixed:g}, R={R}, K={K}")
        for i in [0, len(targets)//2, len(targets)-1]:
            t = targets[i]
            known = {k: v for k, v in t.items() if k != secret_name}
            slopes = []
            for _ in range(R):
                p_lo = denoise(diag_client, {**known, secret_name: probe_lo}, K)
                p_hi = denoise(diag_client, {**known, secret_name: probe_hi}, K)
                slopes.append((_logit(p_hi) - _logit(p_lo)) / (probe_hi - probe_lo))
            m = sum(slopes) / R
            sd = (sum((s - m) ** 2 for s in slopes) / (R - 1)) ** 0.5
            sem = sd / (R ** 0.5)
            snr = abs(m) / sem if sem > 0 else float("inf")
            signos = sum(1 for s in slopes if s > 0)
            print(f"         target[{i}] Class={t.get('Class','?')}: "
                  f"slope_mean={m:+.4f}  slope_std={sd:.4f}  SNR={snr:.2f}  "
                  f"(umbral={snr_threshold})  signos +/-: {signos}/{R-signos}  "
                  f"[{'PASA' if snr >= snr_threshold else 'ABSTIENE'}]")
            print(f"           slopes: {[f'{s:+.3f}' for s in slopes]}")

    results: Dict[float, List[float]] = {}
    for epsilon_in in epsilons_in:
        client = HttpAttackClient(base_url, epsilon_in=epsilon_in, epsilon_out=epsilon_out_fixed)
        print(f"  [barrido] epsilon_in={epsilon_in}, epsilon_out={epsilon_out_fixed}...", flush=True)
        errors = []
        for target in targets:
            known = {k: v for k, v in target.items() if k != secret_name}
            p_target = denoise(client, target, K)
            estimate = invert_feature(
                client, known, secret_name, probe_lo, probe_hi, K, p_target,
                R=R, snr_threshold=snr_threshold,
            )
            if estimate is None:
                estimate = public_mean
            else:
                estimate = min(hi, max(lo, estimate))
            errors.append(estimate - target[secret_name])
        results[epsilon_in] = errors
    curve = build_curve(results, tol=tol_abs)
    for row in curve:
        row["baseline_mae"] = baseline_mae
        row["beats_baseline"] = row["mae"] < baseline_mae
    return curve


def run_experiment_out(
    base_url: str,
    pool: List[dict],
    targets: List[dict],
    secret_name: str = "V4",
    K: int = 200,
    epsilons_out=DEFAULT_EPSILONS_OUT,
    tol_frac: float = 0.25,
    epsilon_in_fixed: float = 10000.0,
    R: int = 8,
    snr_threshold: float = 5.0,
    probe_lo_pct: float = 25.0,
    probe_hi_pct: float = 75.0,
    diag: bool = True,
) -> List[dict]:
    """Barrido PURO de epsilon_out (Experimento 2, salida aislada).

    Espejo de run_experiment con los ejes intercambiados: el input shield queda
    casi transparente (epsilon_in_fixed alto) y se barre epsilon_out. Diferencia
    conceptual clave frente al Exp 1: aquí el sesgo de recorte del output shield
    ES parte de la defensa medida, no un artefacto, así que epsilon_out NO se
    fuerza alto -- se barre hasta valores bajos a propósito.
    """
    probe_lo, probe_hi = public_probe_range(
        pool, secret_name, lo_pct=probe_lo_pct, hi_pct=probe_hi_pct
    )
    tol_abs = tol_frac * secret_scale(pool, secret_name)
    lo, hi = PUBLIC_BOUNDS.get(secret_name, (probe_lo, probe_hi))
    public_mean = float(np.mean([row[secret_name] for row in pool]))
    baseline_errors = [public_mean - t[secret_name] for t in targets]
    baseline_mae = float(np.mean(np.abs(baseline_errors))) if baseline_errors else 0.0

    print(f"  [probes] {secret_name} en p{probe_lo_pct:.0f}/p{probe_hi_pct:.0f} = "
          f"[{probe_lo:.4f}, {probe_hi:.4f}]  (span={probe_hi - probe_lo:.4f})  R={R}")

    # Diagnóstico anclado en el extremo de MÁXIMA señal (output transparente):
    # confirma que el ataque recupera cuando ninguna capa protege. Si aquí
    # ABSTIENE, el problema es el instrumento (epsilon_in_fixed no es bastante
    # transparente, span, etc.), no la defensa -> bisecar antes de reportar.
    if diag and targets:
        diag_eps_out = max(epsilons_out)
        diag_client = HttpAttackClient(
            base_url, epsilon_in=epsilon_in_fixed, epsilon_out=diag_eps_out
        )
        print(f"  [diag] ε_in={epsilon_in_fixed:g}, ε_out={diag_eps_out:g}, R={R}, K={K}")
        for i in [0, len(targets)//2, len(targets)-1]:
            t = targets[i]
            known = {k: v for k, v in t.items() if k != secret_name}
            slopes = []
            for _ in range(R):
                p_lo = denoise(diag_client, {**known, secret_name: probe_lo}, K)
                p_hi = denoise(diag_client, {**known, secret_name: probe_hi}, K)
                slopes.append((_logit(p_hi) - _logit(p_lo)) / (probe_hi - probe_lo))
            m = sum(slopes) / R
            sd = (sum((s - m) ** 2 for s in slopes) / (R - 1)) ** 0.5
            sem = sd / (R ** 0.5)
            snr = abs(m) / sem if sem > 0 else float("inf")
            signos = sum(1 for s in slopes if s > 0)
            print(f"         target[{i}] Class={t.get('Class','?')}: "
                  f"slope_mean={m:+.4f}  slope_std={sd:.4f}  SNR={snr:.2f}  "
                  f"(umbral={snr_threshold})  signos +/-: {signos}/{R-signos}  "
                  f"[{'PASA' if snr >= snr_threshold else 'ABSTIENE'}]")
            print(f"           slopes: {[f'{s:+.3f}' for s in slopes]}")

    results: Dict[float, List[float]] = {}
    for epsilon_out in epsilons_out:
        client = HttpAttackClient(base_url, epsilon_in=epsilon_in_fixed, epsilon_out=epsilon_out)
        print(f"  [barrido] epsilon_in={epsilon_in_fixed} (fijo), epsilon_out={epsilon_out}...", flush=True)
        errors = []
        for target in targets:
            known = {k: v for k, v in target.items() if k != secret_name}
            p_target = denoise(client, target, K)
            estimate = invert_feature(
                client, known, secret_name, probe_lo, probe_hi, K, p_target,
                R=R, snr_threshold=snr_threshold,
            )
            if estimate is None:
                estimate = public_mean
            else:
                estimate = min(hi, max(lo, estimate))
            errors.append(estimate - target[secret_name])
        results[epsilon_out] = errors
    # OJO: build_curve etiqueta la columna clave como 'epsilon'. En este barrido
    # esa columna contiene epsilon_OUT, no epsilon_in. El fixed_label del stdout
    # y el nombre del output-dir (exp2_*) son lo que distingue el eje al analizar.
    curve = build_curve(results, tol=tol_abs)
    for row in curve:
        row["baseline_mae"] = baseline_mae
        row["beats_baseline"] = row["mae"] < baseline_mae
    return curve


def run_floor_experiment(
    base_url: str,
    pool: List[dict],
    targets: List[dict],
    secret_name: str = "V4",
    K: int = 20,
    floor_runs: int = 20,
    floor_epsilon_in: float = 500.0,
    floor_epsilon_out: float = 10000.0,
    floor_targets: int = None,
    R: int = 8,
    snr_threshold: float = 2.0,
    probe_lo_pct: float = 25.0,
    probe_hi_pct: float = 75.0,
) -> Tuple[dict, List[dict]]:
    """Measure the irreducible input-shield bias at one fixed DP point."""
    if floor_runs < 1:
        raise ValueError("floor_runs debe ser >= 1")
    if floor_targets is not None and floor_targets < 1:
        raise ValueError("floor_targets debe ser >= 1 cuando se especifica")

    selected_targets = targets if floor_targets is None else targets[:floor_targets]
    if not selected_targets:
        raise ValueError("No hay targets disponibles para medir el piso")

    probe_lo, probe_hi = public_probe_range(
        pool, secret_name, lo_pct=probe_lo_pct, hi_pct=probe_hi_pct
    )
    lo, hi = PUBLIC_BOUNDS.get(secret_name, (probe_lo, probe_hi))
    public_mean = float(np.mean([row[secret_name] for row in pool]))
    client = HttpAttackClient(
        base_url, epsilon_in=floor_epsilon_in, epsilon_out=floor_epsilon_out
    )
    estimates_by_target: List[List[float]] = []
    target_details: List[dict] = []

    print(f"  [floor] {secret_name} en p{probe_lo_pct:.0f}/p{probe_hi_pct:.0f} = "
          f"[{probe_lo:.4f}, {probe_hi:.4f}]  floor_runs={floor_runs} R={R}")
    print(f"  [floor] epsilon_in={floor_epsilon_in}, epsilon_out={floor_epsilon_out}, "
          f"targets={len(selected_targets)}, K={K}", flush=True)

    for index, target in enumerate(selected_targets):
        known = {key: value for key, value in target.items() if key != secret_name}
        estimates = []
        for _ in range(floor_runs):
            p_target = denoise(client, target, K)
            estimate = invert_feature(
                client, known, secret_name, probe_lo, probe_hi, K, p_target,
                R=R, snr_threshold=snr_threshold,
            )
            if estimate is not None:
                estimates.append(min(hi, max(lo, estimate)))

        true_value = target[secret_name]
        estimate_mean = sum(estimates) / len(estimates) if estimates else None
        variance = (
            sum((estimate - estimate_mean) ** 2 for estimate in estimates) /
            (len(estimates) - 1)
            if len(estimates) >= 2
            else None
        )
        target_details.append({
            "target_index": index,
            "true_value": true_value,
            "estimate_mean": estimate_mean,
            "bias": estimate_mean - true_value if estimate_mean is not None else None,
            "var": variance,
            "n_estimadas": len(estimates),
            "n_abstenidas": floor_runs - len(estimates),
            "public_mean": public_mean,
        })
        estimates_by_target.append(estimates)

    result = build_bias_variance_curve(
        estimates_by_target,
        [target[secret_name] for target in selected_targets],
        floor_runs,
        floor_epsilon_in,
        floor_epsilon_out,
        K,
    )
    return result, target_details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon-out", type=float, default=50.0,
                        help="Epsilon fijo del output shield durante el barrido. "
                         "Debe ser alto (>=50) para evitar sesgo de clipping "
                         "en la inversión logit. Default=50.0.")
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--dataset", type=Path,
                        default=Path(__file__).parent.parent / "modelo-python" / "dataset" / "creditcard.csv")
    parser.add_argument("--secret", default="V4")
    parser.add_argument("--limit", type=int, default=220,
                        help="Total de filas a cargar (pool + targets).")
    parser.add_argument("--public-size", type=int, default=200,
                        help="Filas reservadas al pool público para los probes.")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--epsilons-in", type=float, nargs="+", default=list(DEFAULT_EPSILONS),
                        help="Lista de epsilon_in a barrer (eje de la curva), separados por espacio.")
    parser.add_argument("--axis", choices=["in", "out"], default="in",
                        help="Eje a barrer. 'in' (default): Exp 1/3, barre epsilon_in "
                             "con epsilon_out fijo. 'out': Exp 2, barre epsilon_out con "
                             "epsilon_in fijo alto (input transparente).")
    parser.add_argument("--epsilons-out", type=float, nargs="+",
                        default=list(DEFAULT_EPSILONS_OUT),
                        help="Lista de epsilon_out a barrer. Solo con --axis out.")
    parser.add_argument("--epsilon-in-fixed", type=float, default=10000.0,
                        help="Epsilon_in fijo (alto = input transparente) durante el "
                             "barrido de epsilon_out. Solo con --axis out. Default=10000.")
    parser.add_argument("--no-diag", dest="diag", action="store_false",
                        help="Desactiva el bloque [diag] (ahorra ~3*2*K*R requests/corrida). "
                             "Aplica al barrido --axis out.")
    parser.set_defaults(diag=True)
    parser.add_argument("--tol-frac", type=float, default=0.25,
                        help="Tolerancia de éxito como fracción de la std pública del secreto.")

    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla RNG para reproducibilidad. Si no se pasa, "
                             "cada corrida es aleatoria.")

    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--R", type=int, default=8,
                        help="Repeticiones del sondeo para el guard estadístico "
                             "(Camino 1). Más alto = estimación de pendiente más "
                             "estable, pero R× más requests. Default=8.")
    parser.add_argument("--snr-threshold", type=float, default=2.0,
                        help="Umbral señal/ruido del guard: |slope|/std(slope) "
                             "debe superarlo para no abstenerse. Default=2.0.")
    parser.add_argument("--probe-lo-pct", type=float, default=25.0,
                        help="Percentil inferior del span de sondeo (Camino 2). "
                             "Menos extremo (25 vs 5) = span más estrecho, más "
                             "lineal. Default=25.0.")
    parser.add_argument("--probe-hi-pct", type=float, default=75.0,
                        help="Percentil superior del span de sondeo (Camino 2). "
                             "Default=75.0.")
    parser.add_argument("--floor", action="store_true",
                        help="Mide el piso de proteccion sesgo-varianza en un punto fijo y omite los barridos. "
                            "Costo aprox.: floor_runs * n_targets * (2*K*R + 3*K) requests; "
                            "use --floor-targets, menos --floor-runs o mayor K para acotarlo.")
    parser.add_argument("--floor-runs", type=int, default=20,
                        help="Corridas independientes por target para medir el piso. Default=20.")
    parser.add_argument("--floor-epsilon-in", type=float, default=500.0,
                        help="Epsilon_in fijo del experimento de piso. Default=500.0.")
    parser.add_argument("--floor-epsilon-out", type=float, default=10000.0,
                        help="Epsilon_out transparente para aislar el input shield. Default=10000.0.")
    parser.add_argument("--floor-targets", type=int, default=None,
                        help="Numero de primeros targets para el piso; por defecto usa todos.")
    args = parser.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"[seed] usando semilla fija = {args.seed}")
    else:
        print("[seed] semilla aleatoria (no reproducible)")

    transactions = load_transactions(args.dataset, args.limit)
    pool, targets = split_public_targets(transactions, args.public_size)
    if args.floor:
        floor_result, target_details = run_floor_experiment(
            args.base_url, pool, targets, args.secret, args.K,
            floor_runs=args.floor_runs,
            floor_epsilon_in=args.floor_epsilon_in,
            floor_epsilon_out=args.floor_epsilon_out,
            floor_targets=args.floor_targets,
            R=args.R,
            snr_threshold=args.snr_threshold,
            probe_lo_pct=args.probe_lo_pct,
            probe_hi_pct=args.probe_hi_pct,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "floor_curve.json").write_text(
            json.dumps(floor_result, indent=2), encoding="utf-8"
        )
        with (args.output_dir / "floor_curve.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(floor_result.keys()))
            writer.writeheader()
            writer.writerow(floor_result)
        (args.output_dir / "floor_targets.json").write_text(
            json.dumps(target_details, indent=2), encoding="utf-8"
        )
        print(f"piso_sesgo_abs = {floor_result['piso_sesgo_abs']}")
        print(f"varianza_media = {floor_result['varianza_media']}")
        print(f"tasa_abstencion = {floor_result['tasa_abstencion']}")
        return

    if args.axis == "out":
        curve = run_experiment_out(
            args.base_url, pool, targets, args.secret, args.K,
            epsilons_out=args.epsilons_out, tol_frac=args.tol_frac,
            epsilon_in_fixed=args.epsilon_in_fixed,
            R=args.R, snr_threshold=args.snr_threshold,
            probe_lo_pct=args.probe_lo_pct, probe_hi_pct=args.probe_hi_pct,
            diag=args.diag,
        )
        sweep_label = "epsilon_out"
        fixed_label = f"epsilon_in_fijo={args.epsilon_in_fixed}"
    else:
        curve = run_experiment(
            args.base_url, pool, targets, args.secret, args.K,
            epsilons_in=args.epsilons_in, tol_frac=args.tol_frac,
            epsilon_out_fixed=args.epsilon_out,
            R=args.R, snr_threshold=args.snr_threshold,
            probe_lo_pct=args.probe_lo_pct, probe_hi_pct=args.probe_hi_pct,
        )
        sweep_label = "epsilon_in"
        fixed_label = f"epsilon_out_fijo={args.epsilon_out}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inversion_curve.json").write_text(
        json.dumps(curve, indent=2), encoding="utf-8"
    )
    fieldnames = list(curve[0].keys()) if curve else [
        "epsilon", "mae", "rmse", "mean_error", "n", "success_rate", "tol_abs"
    ]
    with (args.output_dir / "inversion_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve)

    print(f"pool={len(pool)}  targets={len(targets)}  secret={args.secret}  K={args.K}  {fixed_label}")
    if curve:
        print(f"linea_base (adivinar la media) MAE = {curve[0]['baseline_mae']:.4f}")
        print("  -> el ataque solo 'gana' en las filas donde MAE < linea_base\n")
    print(f"{sweep_label}\tMAE_ataque\t¿gana?")
    for row in curve:
        verdict = "SI" if row["beats_baseline"] else "no"
        print(f"{row['epsilon']:>8.0f}\t{row['mae']:.4f}\t\t{verdict}")

if __name__ == "__main__":
    main()