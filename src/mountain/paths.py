"""Chemins du projet — source unique de vérité.

Tout script doit passer par ce module plutôt que d'écrire un chemin relatif.
C'est ce qui permet de lancer n'importe quel script depuis n'importe quel
répertoire, et de déplacer un artefact sans aller corriger quinze fichiers.

Convention des trois familles de répertoires :

- ``models/``   artefacts de modèles (``.pt``, ``.onnx``) — non versionnés, à
                partager via GitHub Releases ;
- ``exports/``  sorties de ``aidge_export_cpp`` — intégralement régénérables,
                jamais versionnées ;
- ``artifacts/`` résultats de run (images annotées, tampons ``.bin``) — jetables.

Seuls ``configs/``, ``data/samples/``, ``src/``, ``scripts/``, ``cpp/``,
``tests/``, ``tools/`` et ``docs/`` sont versionnés.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- racine


def _find_root() -> Path:
    """Racine du dépôt : ``$MOUNTAIN_ROOT`` sinon deux niveaux au-dessus d'ici.

    ``src/mountain/paths.py`` → ``src/mountain`` → ``src`` → racine.
    """
    env = os.environ.get("MOUNTAIN_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


ROOT = _find_root()

# ------------------------------------------------------- dossiers globaux

CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
MODELS = ROOT / "models"
EXPORTS = ROOT / "exports"
ARTIFACTS = ROOT / "artifacts"
CPP = ROOT / "cpp"
SCRIPTS = ROOT / "scripts"
DOCS = ROOT / "docs"

# ------------------------------------------------------------ données

#: Images d'exemple versionnées, utilisées par les scripts de vérification.
SAMPLES = DATA / "samples"
#: Datasets de classification construits par ``scripts/classifier/build_dataset.py``.
DATASETS = DATA / "datasets"
#: Dataset des 5 espèces (ex-``dataset_wildlife/``), arborescence ``train/`` + ``val/``.
WILDLIFE_DATASET = DATASETS / "wildlife"

#: Image de référence 1 Mpx utilisée par presque tous les scripts de contrôle.
SAMPLE_IMAGE = SAMPLES / "test_wildlife_1mp.jpg"

# ------------------------------------------------------------ modèles

DETECTOR_DIR = MODELS / "detector"
CLASSIFIER_DIR = MODELS / "classifier"

#: Poids PyTorch MegaDetector v6 (YOLOv10-c), récupérés par ``fetch_weights.py``.
DETECTOR_PT = DETECTOR_DIR / "MDV6-yolov10-c.pt"
#: Export ONNX brut du détecteur, post-traitement YOLO inclus.
DETECTOR_ONNX = DETECTOR_DIR / "MDV6-yolov10-c.onnx"
#: Détecteur coupé avant le post-traitement DFL/Sigmoid (2 sorties Concat).
DETECTOR_ONNX_CUT = DETECTOR_DIR / "MDV6-yolov10-c_cut.onnx"
#: Version coupée *et* constant-foldée — c'est l'entrée de la quantification.
DETECTOR_ONNX_CUT_FOLDED = DETECTOR_DIR / "MDV6-yolov10-c_cut_folded.onnx"
#: Détecteur fine-tuné 640x640 produit par ``scripts/detector/finetune.py``.
DETECTOR_ONNX_TRAINED = DETECTOR_DIR / "detector_v6_trained_640.onnx"

#: Meilleur checkpoint PyTorch du classifieur d'espèces.
CLASSIFIER_PT = CLASSIFIER_DIR / "best_species_classifier.pt"
#: Classifieur exporté en ONNX fp32 (cf. REQ-027).
CLASSIFIER_ONNX = CLASSIFIER_DIR / "species_classifier.onnx"
#: Classifieur constant-foldé par ONNX Runtime, prêt pour Aidge.
CLASSIFIER_ONNX_FOLDED = CLASSIFIER_DIR / "species_classifier_folded.onnx"

# -------------------------------------------------- exports C++ générés

#: Export C++ fp32 du détecteur (``aidge_export_cpp``).
EXPORT_DETECTOR_FP32 = EXPORTS / "detector_fp32"
#: Export C++ int8 du détecteur, après PTQ.
EXPORT_DETECTOR_INT8 = EXPORTS / "detector_int8"
#: Export C++ fp32 du classifieur.
EXPORT_CLASSIFIER_FP32 = EXPORTS / "classifier_fp32"
#: Export C++ int8 du classifieur, après PTQ.
EXPORT_CLASSIFIER_INT8 = EXPORTS / "classifier_int8"

#: Pipeline C++ autonome assemblé par ``scripts/pipeline/build_standalone_cpp.py``.
#: Entièrement généré — le script efface ce dossier avant de le reconstruire.
#: C'est pour cela qu'il vit sous ``exports/`` et non sous ``cpp/``.
EXPORT_PIPELINE = EXPORTS / "wildlife_pipeline"
#: Sous-dossiers recopiés depuis les exports int8 des deux modèles.
PIPELINE_DETECTOR_DNN = EXPORT_PIPELINE / "detector_dnn"
PIPELINE_CLASSIFIER_DNN = EXPORT_PIPELINE / "classifier_dnn"

# ------------------------------------------------------------ sources C++

#: Harnais C++ du détecteur seul (pré-traitement, décodage DFL, NMS).
CPP_DETECTOR_HARNESS = CPP / "detector_standalone" / "main.cpp"
#: Instantané de référence de la sortie du générateur de pipeline.
CPP_PIPELINE_REFERENCE = CPP / "wildlife_pipeline"
#: Décodeur JPEG en en-tête unique, source canonique.
STB_HEADER = CPP / "third_party" / "stb" / "stb_image.h"

# ------------------------------------------------------- configuration

#: Noms des classes du classifieur, dans l'ordre des logits.
CLASS_NAMES = CONFIGS / "classes.json"
DETECTOR_CONFIG = CONFIGS / "detector.yaml"
CLASSIFIER_CONFIG = CONFIGS / "classifier.yaml"
QUANTIZATION_CONFIG = CONFIGS / "quantization.yaml"
#: Ports série, débits et référence du composant de la carte cible.
BOARD_CONFIG = CONFIGS / "board.yaml"


# ---------------------------------------------------------------- utils


def artifact(name: str) -> Path:
    """Chemin d'un fichier de sortie jetable, en garantissant le dossier.

    À utiliser pour toute image annotée ou tout tampon ``.bin`` intermédiaire :
    ``artifacts/`` est entièrement ignoré par git.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS / name


def ensure_dirs() -> None:
    """Crée les dossiers de sortie absents. Idempotent."""
    for d in (
        DATASETS,
        DETECTOR_DIR,
        CLASSIFIER_DIR,
        EXPORTS,
        ARTIFACTS,
    ):
        d.mkdir(parents=True, exist_ok=True)
