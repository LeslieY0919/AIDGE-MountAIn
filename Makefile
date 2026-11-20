# Points d'entrée du projet, dans l'ordre de la chaîne.
#
# `make help` liste les cibles. Chaque cible n'est qu'un appel au script
# correspondant de scripts/ : la logique vit dans les scripts, pas ici.

# Interpréteur Python. AIDGE exige 3.11 ou 3.12 — pas 3.13, pas moins de 3.11.
# On ne peut pas se contenter de `python` : sur macOS il n'existe souvent pas,
# et `python3` est fréquemment un 3.9 livré par le système. On cherche donc le
# premier interpréteur compatible, en laissant gagner un venv actif.
# `make PYTHON=/chemin/vers/python <cible>` passe outre.
PYTHON ?= $(shell \
  if [ -n "$$VIRTUAL_ENV" ] && [ -x "$$VIRTUAL_ENV/bin/python" ]; then \
    echo "$$VIRTUAL_ENV/bin/python"; \
  else \
    for c in python3.12 python3.11 python3 python; do \
      command -v $$c >/dev/null 2>&1 || continue; \
      if $$c -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info < (3,13) else 1)' \
         >/dev/null 2>&1; then command -v $$c; break; fi; \
    done; \
  fi)

# Toute cible qui exécute du Python en dépend : mieux vaut un message clair
# qu'un `ERROR: Package 'mountain' requires a different Python`.
.PHONY: require-python
require-python:
	@if [ -z "$(PYTHON)" ]; then \
	  echo "Aucun Python 3.11 ou 3.12 trouvé sur cette machine."; \
	  echo; \
	  echo "Interpréteurs visibles :"; \
	  for c in python python3 python3.11 python3.12 python3.13; do \
	    command -v $$c >/dev/null 2>&1 && \
	      printf '  %-12s %s\n' "$$c" "$$($$c -V 2>&1)"; \
	  done; \
	  echo; \
	  echo "AIDGE ne supporte ni 3.13 ni les versions antérieures à 3.11."; \
	  echo "Installer Python 3.12, puis : make venv"; \
	  echo "Ou forcer : make PYTHON=/chemin/vers/python3.12 <cible>"; \
	  exit 1; \
	fi


.DEFAULT_GOAL := help
PIP_FLAGS ?=

.PHONY: help venv install detector-weights dataset train-classifier \
        export-detector export-classifier quantize build-cpp \
        install-board run-reference check test board-list board-check connect \
        board-firmware \
        clean-exports clean-artifacts

help:  ## Affiche cette aide
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sed -e 's/:.*## /\t/' \
	  | awk -F'\t' '{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# Homebrew et les Python de distribution se déclarent « externally managed »
# (PEP 668) et refusent toute installation hors environnement virtuel. Sans ce
# garde-fou, `make install` renvoie un pavé d'erreur pip au lieu de dire quoi
# faire. On ne bloque que si l'interpréteur refuse vraiment : un Python non
# managé s'installe très bien tel quel.
.PHONY: require-env
require-env: require-python
	@if [ -z "$$VIRTUAL_ENV" ] && [ -z "$$CONDA_PREFIX" ] && \
	   $(PYTHON) -c 'import os,sysconfig,sys; sys.exit(0 if os.path.exists(\
	     os.path.join(sysconfig.get_paths()["stdlib"], "EXTERNALLY-MANAGED")) else 1)' \
	     >/dev/null 2>&1; then \
	  echo "$$($(PYTHON) -V 2>&1) refuse d'installer hors environnement virtuel (PEP 668)."; \
	  echo; \
	  if [ -d .venv ]; then \
	    echo "Un .venv existe déjà mais n'est pas activé :"; \
	    echo "  source .venv/bin/activate"; \
	  else \
	    echo "  make venv"; \
	    echo "  source .venv/bin/activate"; \
	  fi; \
	  echo "  make $(MAKECMDGOALS)"; \
	  echo; \
	  echo "Pour passer outre malgré tout : make PIP_FLAGS=--break-system-packages $(MAKECMDGOALS)"; \
	  exit 1; \
	fi

# AIDGE ne publie des roues que pour Linux x86_64 et Windows amd64 : ni macOS,
# ni ARM, et aucun sdist de repli. Sans ce garde-fou, `make install` part dans
# une résolution pip de plusieurs minutes pour finir sur un
# « Could not find a version that satisfies the requirement aidge-core », qui
# ressemble à un problème de version alors que c'est un problème de plateforme.
.PHONY: require-platform
require-platform: require-python
	@if ! $(PYTHON) -c 'import platform, sys; \
	  s, m = platform.system(), platform.machine(); \
	  sys.exit(0 if (s == "Linux" and m == "x86_64") \
	           or (s == "Windows" and m in ("AMD64", "x86_64")) else 1)' \
	  >/dev/null 2>&1; then \
	  echo "AIDGE ne fournit pas de paquet pour cette plateforme :"; \
	  echo "  $$($(PYTHON) -c 'import platform; print(platform.system(), platform.machine())')"; \
	  echo; \
	  echo "Les roues publiées couvrent uniquement Linux x86_64 et Windows amd64."; \
	  echo "Il n'existe ni roue macOS ni sdist : aucune version de Python n'y change rien."; \
	  echo; \
	  echo "Cette machine peut en revanche parler à la carte :"; \
	  echo "  make install-board"; \
	  echo; \
	  echo "La chaîne d'entraînement et d'export doit tourner sur une machine"; \
	  echo "Linux x86_64 (ou Windows amd64). Voir la section Prérequis du README."; \
	  echo; \
	  echo "Pour tenter quand même : make FORCE_PLATFORM=1 install"; \
	  test -n "$(FORCE_PLATFORM)" || exit 1; \
	fi

venv: require-python  ## Crée .venv avec un interpréteur compatible
	$(PYTHON) -m venv .venv
	@echo
	@echo "Créé avec $$($(PYTHON) -V 2>&1). Activer puis installer :"
	@echo "  source .venv/bin/activate"
	@echo "  make install         # chaîne complète"
	@echo "  make install-board   # ou le minimum pour la carte"

install: require-platform require-env  ## Installe la chaîne complète (entraînement, export, quantification)
	$(PYTHON) -m pip install $(PIP_FLAGS) -e ".[ml,dev]"
	$(PYTHON) -m pip install $(PIP_FLAGS) --no-deps aidge-quantization

install-board: require-env  ## Installe le minimum pour parler à la carte (sans torch ni AIDGE)
	$(PYTHON) -m pip install $(PIP_FLAGS) -e ".[board]"

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

# Compile sur l'ordinateur uniquement : rien n'est écrit sur la carte. Le flash
# (app-write-mram) reste une étape manuelle, voir AGENTS.md.
board-firmware:  ## Compile la démo img_class du kit Alif et son paquet ATOC (sans flasher)
	scripts/board/build_img_class.sh

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
