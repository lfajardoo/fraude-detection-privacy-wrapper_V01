"""Ataques de laboratorio contra la interfaz publicada de predicción."""

from .api_client import AttackClient, HttpAttackClient
from .inversion_attack import denoise, invert_feature

__all__ = ["AttackClient", "HttpAttackClient", "denoise", "invert_feature"]