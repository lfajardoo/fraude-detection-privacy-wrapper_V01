"""Harness investigador para el ataque de inversión inter-request."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from attacks.api_client import HttpAttackClient
from attacks.inversion_attack import denoise, invert_feature
from attacks.metrics_attack import build_curve


DEFAULT_EPSILONS = (0.1, 0.5, 1.0, 3.0, 10.0)


def load_transactions(dataset_path: Path, limit: int) -> List[dict]:
    with dataset_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))[:limit]
    public_fields = ["Time"] + [f"V{i}" for i in range(1, 15)] + ["Amount"]
    return [{field: float(row[field]) for field in public_fields} for row in rows]


def public_probe_range(transactions: List[dict], secret_name: str) -> tuple[float, float]:
    values = sorted(row[secret_name] for row in transactions)
    if not values:
        raise ValueError("No hay transacciones para calcular probes públicos")
    return values[0], values[-1]


def run_experiment(
    base_url: str,
    transactions: List[dict],
    secret_name: str = "V4",
    K: int = 20,
    epsilons=DEFAULT_EPSILONS,
    tolerance: float = 0.5,
) -> List[dict]:
    probe_lo, probe_hi = public_probe_range(transactions, secret_name)
    results: Dict[float, List[float]] = {}
    for epsilon in epsilons:
        client = HttpAttackClient(base_url, epsilon_in=epsilon)
        errors = []
        for target in transactions:
            known = {key: value for key, value in target.items() if key != secret_name}
            p_target = denoise(client, target, K)
            estimate = invert_feature(
                client, known, secret_name, probe_lo, probe_hi, K, p_target
            )
            if estimate is not None:
                errors.append(estimate - target[secret_name])
        results[epsilon] = errors
    return build_curve(results, tol=tolerance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent.parent / "modelo-python" / "dataset" / "creditcard.csv")
    parser.add_argument("--secret", default="V4")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    curve = run_experiment(args.base_url, load_transactions(args.dataset, args.limit), args.secret, args.K, tolerance=args.tolerance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inversion_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
    with (args.output_dir / "inversion_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=curve[0].keys() if curve else ["epsilon_in", "mae", "rmse", "n", "success_rate", "floor"])
        writer.writeheader()
        writer.writerows(curve)
    print("epsilon_in\tMAE")
    for row in curve:
        print(f"{row['epsilon_in']:.3g}\t{row['mae']:.6f}")


if __name__ == "__main__":
    main()