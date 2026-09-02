# privacy-api/sanity_check.py
"""Verifica, sin ruido y sin promediado, que mover una feature mueve la salida."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from attacks.api_client import HttpAttackClient
from attacks.inversion_attack import denoise
from evaluator_attack import load_transactions, split_public_targets

BASE_URL = "http://localhost:8001"
EPS = 1e9  # ruido despreciable en ambas capas

def main():
    dataset = Path(__file__).parent.parent / "modelo-python" / "dataset" / "creditcard.csv"
    pool, targets = split_public_targets(load_transactions(dataset, 130), 100)
    client = HttpAttackClient(BASE_URL, epsilon_in=EPS, epsilon_out=EPS)

    t = targets[0]
    known = {k: v for k, v in t.items() if k != "V4"}

    print("=== Barrido de V4 (una sola request por punto, sin ruido) ===")
    prev = None
    for v4 in [-5, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
        p = denoise(client, {**known, "V4": v4}, 1)
        delta = "" if prev is None else f"  Δ={p-prev:+.6f}"
        print(f"  V4={v4:>3}  p={p:.6f}{delta}")
        prev = p

    print("\n=== Sensibilidad de cada feature (±2 alrededor de su valor real) ===")
    base = denoise(client, t, 1)
    print(f"  p base = {base:.6f}")
    sens = []
    for f in known:
        p_hi = denoise(client, {**t, f: t[f] + 2}, 1)
        p_lo = denoise(client, {**t, f: t[f] - 2}, 1)
        sens.append((abs(p_hi - p_lo), f, p_lo, p_hi))
    for d, f, lo, hi in sorted(sens, reverse=True)[:8]:
        print(f"  {f:<8} p(-2)={lo:.6f}  p(+2)={hi:.6f}  |Δ|={d:.6f}")

if __name__ == "__main__":
    main()