"""Clientes que exponen al atacante únicamente una probabilidad publicada."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import requests


class AttackClient(ABC):
    """Interfaz ciega del oráculo de predicción."""

    @abstractmethod
    def query(self, features: dict) -> float:
        """Devuelve una única probabilidad ruidosa."""
        raise NotImplementedError


class HttpAttackClient(AttackClient):
    """Cliente HTTP cuyo contrato descarta todos los campos no publicados."""

    def __init__(self, base_url: str, epsilon_in: float, epsilon_out: float, timeout: int = 30):
        self._base_url = base_url.rstrip("/")
        self._epsilon_in = epsilon_in
        self._epsilon_out = epsilon_out
        self._timeout = timeout

    def query(self, features: dict) -> float:
        body: Dict[str, Any] = dict(features)
        body.update({"noise_repetitions": 1, "epsilon_in": self._epsilon_in, "epsilon_out": self._epsilon_out})
        response = requests.post(
            f"{self._base_url}/predict?mode=legacy",
            json=body,
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return float(payload["output_shield"]["samples"][0]["fraud_probability"])

    def get_epsilon_out_from_response(self, features: dict) -> float:
        """Extrae el epsilon_out efectivo de la metadata de la respuesta."""
        body: Dict[str, Any] = dict(features)
        body.update({"noise_repetitions": 1, "epsilon_in": self._epsilon_in, "epsilon_out": self._epsilon_out})
        response = requests.post(
            f"{self._base_url}/predict?mode=legacy",
            json=body,
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return float(payload["output_shield"]["epsilon_out"])