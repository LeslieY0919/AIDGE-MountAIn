#!/usr/bin/env bash
# Exécute Aidge dans Docker sur une machine Linux x86_64 distante, jointe en SSH (alias SSH « dev2 » par défaut,
# modifiable avec DEV2_HOST). Sert sur les machines sans wheel Aidge (macOS). Les fichiers vont dans
# ~/wildlife_moniter sur la machine distante, avec les mêmes chemins que le dépôt local ; c'est aussi le dossier
# de travail du conteneur. Le paquet mountain y est importé par PYTHONPATH=src, sans installation.
#
# Usage :
#   scripts/npu/aidge_docker.sh build          # envoie docker/aidge/ et construit l'image en arrière-plan
#   scripts/npu/aidge_docker.sh status         # fin du journal de construction et présence de l'image
#   scripts/npu/aidge_docker.sh push PATH...   # envoie des fichiers ou dossiers du dépôt
#   scripts/npu/aidge_docker.sh run CMD...     # envoie le code (scripts/npu, src, configs) puis exécute CMD
#   scripts/npu/aidge_docker.sh fetch PATH...  # rapatrie des résultats
# PIP_INDEX_URL permet de passer par un miroir pendant la construction.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HOST="${DEV2_HOST:-dev2}"
REMOTE_DIR="wildlife_moniter"   # relatif au ~ de la machine distante
IMAGE="wildlife-aidge:0.10.1"
# Code dont le conteneur a besoin : synchronisé avant chaque exécution
CODE=(scripts/npu/ src/mountain/ configs/)

push() { (cd "$ROOT" && rsync -az -R "$@" "$HOST:$REMOTE_DIR/"); }

cmd="${1:-}"
shift || true
case "$cmd" in
  build)
    push docker/aidge/
    # Réseau lent et SSH parfois instable : construction en arrière-plan avec nohup
    args="${PIP_INDEX_URL:+--build-arg PIP_INDEX_URL=$PIP_INDEX_URL}"
    ssh "$HOST" "cd ~/$REMOTE_DIR/docker/aidge && nohup docker build --progress=plain $args -t $IMAGE . > build.log 2>&1 < /dev/null &"
    echo "$HOST : construction de $IMAGE lancée, suivi : scripts/npu/aidge_docker.sh status"
    ;;
  status)
    ssh "$HOST" "tail -n 15 ~/$REMOTE_DIR/docker/aidge/build.log; docker image ls $IMAGE"
    ;;
  push)
    push "$@"
    ;;
  run)
    push "${CODE[@]}"
    # Lancé avec l'utilisateur distant : les fichiers créés n'appartiennent pas à root
    ssh "$HOST" "cd ~/$REMOTE_DIR && docker run --rm --user \$(id -u):\$(id -g) -e HOME=/tmp -e PYTHONPATH=/work/src -e PYTHONDONTWRITEBYTECODE=1 -v \"\$PWD\":/work -w /work $IMAGE $(printf '%q ' "$@")"
    ;;
  fetch)
    for p in "$@"; do
      mkdir -p "$ROOT/$(dirname "$p")"
      rsync -a "$HOST:$REMOTE_DIR/$p" "$ROOT/$(dirname "$p")/"
    done
    ;;
  *)
    sed -n '2,13p' "$0"
    exit 1
    ;;
esac
