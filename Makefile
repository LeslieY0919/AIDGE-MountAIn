# Points d'entrée du projet, dans l'ordre de la chaîne.
#
# `make help` liste les cibles. Chaque cible n'est qu'un appel au script
# correspondant de scripts/ : la logique vit dans les scripts, pas ici.

PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help install detector-weights dataset train-classifier \
        export-detector export-classifier quantize build-cpp \
        install-board run-reference check test board-list board-check connect \
        clean-exports clean-artifacts

help:  ## Affiche cette aide
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sed -e 's/:.*## /\t/' \
	  | awk -F'\t' '{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install:  ## Installe la chaîne complète (entraînement, export, quantification)
	$(PYTHON) -m pip install -e ".[ml,dev]"
	$(PYTHON) -m pip install --no-deps aidge-quantization

install-board:  ## Installe le minimum pour parler à la carte (sans torch ni AIDGE)
	$(PYTHON) -m pip install -e ".[board]"

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

# --- carte cible ---------------------------------------------------------

board-list:  ## Liste les ports série candidats de la carte Alif
	$(PYTHON) scripts/board/connect.py --list

board-check:  ## Vérifie que SETOOLS voit bien un E7
	$(PYTHON) scripts/board/connect.py --setools

connect:  ## Ouvre la console série de la carte (115200 bauds, journalisée)
	$(PYTHON) scripts/board/connect.py --log

# --- 4. vérification -----------------------------------------------------

run-reference:  ## Exécute la référence Python sur l'image d'exemple
	$(PYTHON) scripts/pipeline/run_reference.py

# Les contrôles de tests/ sont préfixés check_ et non test_ : ce sont des
# scripts exécutables qui ont besoin d'artefacts construits. pytest ne les
# collecte donc pas, et on les enchaîne explicitement ici. L'ordre est celui du
# diagnostic : un échec en bas se comprend en regardant ce qui a passé en haut.
CHECKS = \
  tests/check_aidge_onnx_smoke.py \
  tests/check_detector_export.py \
  tests/check_detector_aidge_parity.py \
  tests/check_cpp_detector_fp32.py \
  tests/check_cpp_detector_int8.py \
  tests/check_detector_int8_decode.py \
  tests/check_cpp_pipeline_full.py

check:  ## Lance les contrôles de parité Python <-> C++, dans l'ordre du diagnostic
	@for c in $(CHECKS); do \
	  echo "=== $$c ==="; \
	  $(PYTHON) $$c || { echo "ÉCHEC : $$c"; exit 1; }; \
	done
	@echo "Tous les contrôles ont passé."

test:  ## Lance les tests unitaires pytest (test_*.py ; aucun pour l'instant)
	$(PYTHON) -m pytest

# --- ménage --------------------------------------------------------------

clean-exports:  ## Supprime les exports C++ générés
	rm -rf exports/

clean-artifacts:  ## Supprime les résultats de run
	rm -rf artifacts/
