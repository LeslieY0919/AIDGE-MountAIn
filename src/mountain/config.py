"""Lecture des fichiers de ``configs/``.

Les constantes qui étaient dispersées en tête de chaque script (taille
d'entrée, seuils, MEAN/STD, hyperparamètres) vivent désormais dans
``configs/*.yaml``. Un seul endroit à éditer, et les valeurs deviennent
lisibles sans ouvrir de code Python.

Chaque accesseur met son résultat en cache : appeler ``detector()`` dix fois
ne relit pas le fichier dix fois.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import yaml

from . import paths


def _load_yaml(path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def detector() -> dict[str, Any]:
    """Configuration du détecteur : résolution, seuils, post-traitement."""
    return _load_yaml(paths.DETECTOR_CONFIG)


@lru_cache(maxsize=None)
def classifier() -> dict[str, Any]:
    """Configuration du classifieur : architecture, normalisation, entraînement."""
    return _load_yaml(paths.CLASSIFIER_CONFIG)


@lru_cache(maxsize=None)
def quantization() -> dict[str, Any]:
    """Configuration de la quantification post-entraînement (PTQ int8)."""
    return _load_yaml(paths.QUANTIZATION_CONFIG)


@lru_cache(maxsize=None)
def class_names() -> list[str]:
    """Noms des espèces, dans l'ordre des logits du classifieur."""
    with open(paths.CLASS_NAMES, encoding="utf-8") as f:
        return json.load(f)


def num_classes() -> int:
    return len(class_names())
