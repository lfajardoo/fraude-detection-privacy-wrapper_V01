# Archivo: ataque de inversión (experimentos 1–3)

## Qué es

Resultados (`results/`: `run_1…5`, `exp2_*`, `exp3_*`) y documentación
([EXPLICACION_RESULTADOS.md](./EXPLICACION_RESULTADOS.md)) de los experimentos 1–3, que
validaban el privacy wrapper con un **ataque de inversión** sobre la feature V4
(≈710 líneas de código y ≈10 h de cómputo por experimento).

## Dónde está el código

El código del ataque (`privacy-api/attacks/`, `evaluator_attack.py`, `test_attacks.py`,
`sanity_check.py`) ya no está en `main`. Se conserva en el tag **`v0-ataque-inversion`**
(commit `0542d54`):

```bash
git checkout v0-ataque-inversion
```

## Estado

Son un **caso de estudio ilustrativo**, no la validación principal. La validación vigente es
la de tres niveles descrita en [VALIDACION.md](../../VALIDACION.md).
