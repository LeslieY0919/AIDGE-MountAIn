#!/usr/bin/env bash
# Compile alif_img_class (cœur M55-HP de l'AppKit-e7) et génère le paquet ATOC dans SETOOLS.
# Tout se passe sur l'ordinateur, rien n'est écrit sur la carte ; le flash se fait avec app-write-mram,
# après accord de l'utilisateur.
#
# Variables d'environnement optionnelles :
#   MODEL_PATH   .tflite compilé par Vela (ethos-u55-256) ; MobileNet-v2 par défaut
#   LABELS_FILE  fichier d'étiquettes, une classe par ligne
#   IMAGE_SIZE   côté de l'image d'entrée du modèle, 224 par défaut
#   SHOW_INF_TIME=1  affiche sur l'écran le temps de prétraitement + inférence + post-traitement
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$PROJECT_ROOT/scripts/env.sh"

BUILD_DIR="$PROJECT_ROOT/build/mlek_img_class_hp"
# CMake 3.30.5 de l'environnement virtuel du kit
CMAKE="$MLEK_ROOT/resources_downloaded/env/bin/cmake"

cmake_args=(
    -DTARGET_PLATFORM=alif
    -DTARGET_SUBSYSTEM=RTSS-HP
    -DTARGET_BOARD=AppKit-e7
    -DUSE_CASE_BUILD=alif_img_class
    -DCMAKE_TOOLCHAIN_FILE=scripts/cmake/toolchains/bare-metal-gcc.cmake
    -DCMAKE_BUILD_TYPE=Release
    -DMLEK_LOG_LEVEL=MLEK_LOG_LEVEL_INFO
)
if [ -n "${MODEL_PATH:-}" ]; then cmake_args+=("-Dalif_img_class_MODEL_PATH=$MODEL_PATH"); fi
if [ -n "${LABELS_FILE:-}" ]; then cmake_args+=("-Dalif_img_class_LABELS_TXT_FILE=$LABELS_FILE"); fi
if [ -n "${IMAGE_SIZE:-}" ]; then cmake_args+=("-Dalif_img_class_IMAGE_SIZE=$IMAGE_SIZE"); fi
if [ "${SHOW_INF_TIME:-0}" = "1" ]; then cmake_args+=("-Dalif_img_class_SHOW_INF_TIME=ON"); else cmake_args+=("-Dalif_img_class_SHOW_INF_TIME=OFF"); fi

"$CMAKE" -S "$MLEK_ROOT" -B "$BUILD_DIR" -G "Unix Makefiles" "${cmake_args[@]}"
make -C "$BUILD_DIR" mlek_alif_img_class -j"$(sysctl -n hw.ncpu)"

# Copie l'image et la configuration dans SETOOLS, puis génère le paquet ATOC (build/AppTocPackage.bin)
cp "$BUILD_DIR/bin/sectors/alif_img_class/mram.bin" "$ALIF_SETOOLS_ROOT/build/images/mlek_alif_img_class.bin"
cp "$MLEK_ROOT/alif_img_class_hp.json" "$ALIF_SETOOLS_ROOT/build/config/alif_img_class_hp.json"
(cd "$ALIF_SETOOLS_ROOT" && ./app-gen-toc -f build/config/alif_img_class_hp.json)
