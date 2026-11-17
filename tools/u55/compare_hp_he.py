#!/usr/bin/env python3
# Compile avec Vela des modèles int8 existants pour le HP (M55-HP + U55-256 à 400 MHz) et le HE (M55-HE + U55-128 à 160 MHz),
# puis compare vitesse et mémoire. Les modèles viennent de profile_yolov10_u55.py et profile_classifier_u55.py.
# S'exécute uniquement sur l'ordinateur, rien n'est écrit sur la carte.
# Usage : .venv-yolo/bin/python tools/u55/compare_hp_he.py artifacts/u55_profile/yolov10n_relu_nopsa_256/*_full_integer_quant.tflite ...
import argparse
import json
from pathlib import Path

from u55_profile_common import OUT_DIR, TARGETS, run_vela


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tflite", nargs="+", type=Path)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=TARGETS)
    args = ap.parse_args()

    results = []
    for tflite in args.tflite:
        for target in args.targets:
            result = {"model": tflite.parent.name, "target": target, **run_vela(tflite.resolve(), target)}
            print(json.dumps(result), flush=True)
            results.append(result)
    (OUT_DIR / "hp_he_results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
