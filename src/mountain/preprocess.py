"""Pré-traitement d'image, partagé entre toutes les étapes.

Ce module existe parce que le même bloc « resize + normalisation + NCHW »
était recopié à l'identique dans huit scripts, avec le risque qu'une copie
dérive des autres. Le pré-traitement de la calibration PTQ *doit* être
rigoureusement identique à celui de l'entraînement, sinon la quantification
s'étalonne sur une distribution qui n'est pas celle du modèle.

Deux conventions coexistent dans le projet, et c'est volontaire :

- **classifieur** : normalisation ImageNet (``mean``/``std``), tenseur NCHW ;
- **détecteur**   : simple division par 255, entrées dans [0, 1] et NCHW,
  contrainte de la chaîne d'export AIDGE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import config

#: Normalisation ImageNet utilisée par le classifieur (cf. configs/classifier.yaml).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_nchw(arr: np.ndarray) -> np.ndarray:
    """HWC -> NCHW contigu, avec la dimension de batch ajoutée."""
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...])


def classifier_input(
    img: str | Path | Image.Image,
    size: int | None = None,
) -> np.ndarray:
    """Image -> tenseur ``(1, 3, size, size)`` normalisé ImageNet.

    ``size`` vaut par défaut ``input_size`` de ``configs/classifier.yaml``.
    C'est le pré-traitement de référence : entraînement, calibration PTQ et
    vérification doivent tous passer par ici.
    """
    if size is None:
        size = config.classifier()["input_size"]
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = (np.array(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return to_nchw(arr)


def detector_input(
    img: str | Path | Image.Image,
    size: int | None = None,
) -> np.ndarray:
    """Image -> tenseur ``(1, 3, size, size)`` dans [0, 1].

    Pas de normalisation ImageNet ici : la chaîne d'export AIDGE exige un ONNX
    dont les entrées sont dans [0, 1] et en NCHW.
    """
    if size is None:
        size = config.detector()["input_size"]
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return to_nchw(arr)
