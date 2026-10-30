"""Hace importables las capas del wrapper y el paquete `validacion`.

Las capas usan imports planos (`from config import ...`), por lo que
privacy-api/ debe estar en sys.path. No se modifica ningún archivo existente.
"""

import sys
from pathlib import Path

PRIVACY_API_DIR = Path(__file__).resolve().parents[1]
if str(PRIVACY_API_DIR) not in sys.path:
    sys.path.insert(0, str(PRIVACY_API_DIR))
