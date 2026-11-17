# Points d'entrée du projet, dans l'ordre de la chaîne.
#
# `make help` liste les cibles. Chaque cible n'est qu'un appel au script
# correspondant de scripts/ : la logique vit dans les scripts, pas ici.

PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help install detector-weights dataset train-classifier \
        export-detector export-classifier quantize build-cpp \
        run-reference check clean-exports clean-artifacts

help:  ## Affiche cette aide
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sed -e 's/:.*## /\t/' \
	  | awk -F'\t' '{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install:  ## Installe le paquet mountain et ses dépendances en mode éditable
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pip install --no-deps aidge-quantization

# --- 1. préparation des modèles et des données ---------------------------

detector-weights:  ## Récupère les poids MegaDetector v6 et les exporte en ONNX
	$(PYTHON) scripts/detector/fetch_weights.py

dataset:  ## Construit le dataset 5 espèces (nécessite le détecteur ONNX)
	$(PYTHON) scripts/classifier/build_dataset.py

train-classifier:  ## Entraîne le classifieur d'espèces et exporte son ONNX
	$(PYTHON) scripts/classifier/train.py

# --- 2. export C++ fp32 (contrôle de faisabilité) ------------------------

export-detector:  ## Exporte le détecteur en C++ fp32 sous exports/detector_fp32/
	$(PYTHON) scripts/detector/export_cpp.py

export-classifier:  ## Exporte le classifieur en C++ fp32 sous exports/classifier_fp32/
	$(PYTHON) scripts/classifier/export_cpp.py

# --- 3. quantification int8 et assemblage --------------------------------

quantize:  ## PTQ int8 des deux modèles + export C++ sous exports/*_int8/
	$(PYTHON) scripts/quantize/pipeline_int8.py

build-cpp:  ## Assemble et compile le pipeline C++ sous exports/wildlife_pipeline/
	$(PYTHON) scripts/pipeline/build_standalone_cpp.py

# --- 4. vérification -----------------------------------------------------

run-reference:  ## Exécute la référence Python sur l'image d'exemple
	$(PYTHON) scripts/pipeline/run_reference.py

check:  ## Lance les contrôles de parité Python <-> C++
	$(PYTHON) -m pytest

# --- ménage --------------------------------------------------------------

clean-exports:  ## Supprime les exports C++ générés
	rm -rf exports/

clean-artifacts:  ## Supprime les résultats de run
	rm -rf artifacts/
