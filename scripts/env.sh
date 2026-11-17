# Variables d'environnement pour la carte Alif E7. Utilisation : source scripts/env.sh
# Les valeurs par défaut correspondent à une installation sous macOS dans ~/opt ; chaque machine peut les
# redéfinir dans scripts/env.local.sh (non versionné), chargé en premier.

# Racine du dépôt (fonctionne avec bash et zsh)
WILDLIFE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)"
export WILDLIFE_ROOT
[ -f "$WILDLIFE_ROOT/scripts/env.local.sh" ] && source "$WILDLIFE_ROOT/scripts/env.local.sh"

# Emplacement des outils (dossier utilisateur, installés sans sudo)
export JLINK_ROOT="${JLINK_ROOT:-$HOME/opt/SEGGER/JLink_V984}"
export ALIF_SETOOLS_ROOT="${ALIF_SETOOLS_ROOT:-$HOME/opt/alif/app-release-exec-macos}"
export ARM_GCC_ROOT="${ARM_GCC_ROOT:-$HOME/opt/arm/arm-gnu-toolchain-13.3.rel1-darwin-arm64-arm-none-eabi}"

# Kit d'évaluation ML d'Alif (sources, environnement virtuel Python, Vela et CMake en version figée)
export MLEK_ROOT="${MLEK_ROOT:-$WILDLIFE_ROOT/external/alif_ml-embedded-evaluation-kit}"

# Port série SEUART de la carte (VCOM de la J-Link, via PRG USB) ; par défaut, le premier usbmodem trouvé
if [ -z "${ALIF_SEUART_PORT:-}" ]; then
    ALIF_SEUART_PORT="$(ls /dev/cu.usbmodem* 2>/dev/null | head -n 1)"
fi
export ALIF_SEUART_PORT

# Le Python 3.10 de python.org n'a pas de certificats CA configurés (erreurs SSL au téléchargement des modèles) ;
# on utilise ceux de macOS
[ -f /etc/ssl/cert.pem ] && export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/cert.pem}"

# Le guide Alif exige GNU make 4.x au lieu du make 3.81 fourni avec macOS
export PATH="$ARM_GCC_ROOT/bin:$JLINK_ROOT:/opt/homebrew/opt/make/libexec/gnubin:$PATH"

# CMake 4 ne gère plus cmake_minimum_required < 3.5 ; nécessaire pour les anciens sous-modules
export CMAKE_POLICY_VERSION_MINIMUM=3.5
