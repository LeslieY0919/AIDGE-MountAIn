#!/usr/bin/env bash
# Exécute Aidge dans Docker sur une machine Linux x86_64 distante, jointe en SSH (hôte « dev2 » par défaut,
# modifiable avec DEV2_HOST). Les fichiers vont dans ~/wildlife_moniter sur cette machine, avec les mêmes chemins
# que le dépôt local ; c'est aussi le dossier de travail du conteneur.
#
# Usage :
#   scripts/dev2_aidge.sh build          # synchronise aidge/ et construit l'image en arrière-plan
#   scripts/dev2_aidge.sh status         # fin du journal de construction et présence de l'image
#   scripts/dev2_aidge.sh push PATH...   # envoie des fichiers ou dossiers du dépôt
#   scripts/dev2_aidge.sh run CMD...     # synchronise aidge/ puis exécute CMD dans le conteneur
#   scripts/dev2_aidge.sh fetch PATH...  # rapatrie des résultats
# PIP_INDEX_URL permet de passer par un miroir pendant la construction.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${DEV2_HOST:-dev2}"
REMOTE_DIR="wildlife_moniter"   # relatif au ~ de la machine distante
IMAGE="wildlife-aidge:0.10.1"

push() { (cd "$ROOT" && rsync -a -R "$@" "$HOST:$REMOTE_DIR/"); }

cmd="${1:-}"
shift || true
case "$cmd" in
  build)
    push aidge/
    # Réseau lent et SSH parfois instable : construction en arrière-plan avec nohup
    args="${PIP_INDEX_URL:+--build-arg PIP_INDEX_URL=$PIP_INDEX_URL}"
    ssh "$HOST" "cd ~/$REMOTE_DIR/aidge && nohup docker build --progress=plain $args -t $IMAGE . > build.log 2>&1 < /dev/null &"
    echo "$HOST : construction de $IMAGE lancée, suivi : scripts/dev2_aidge.sh status"
    ;;
  status)
    ssh "$HOST" "tail -n 15 ~/$REMOTE_DIR/aidge/build.log; docker image ls $IMAGE"
    ;;
  push)
    push "$@"
    ;;
  run)
    push aidge/
    # Lancé avec l'utilisateur distant : les fichiers créés n'appartiennent pas à root
    ssh "$HOST" "cd ~/$REMOTE_DIR && docker run --rm --user \$(id -u):\$(id -g) -e HOME=/tmp -v \"\$PWD\":/work -w /work $IMAGE $(printf '%q ' "$@")"
    ;;
  fetch)
    for p in "$@"; do
      mkdir -p "$ROOT/$(dirname "$p")"
      rsync -a "$HOST:$REMOTE_DIR/$p" "$ROOT/$(dirname "$p")/"
    done
    ;;
  *)
    sed -n '2,12p' "$0"
    exit 1
    ;;
esac
