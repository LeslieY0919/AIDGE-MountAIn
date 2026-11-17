#!/usr/bin/env python3
# Évaluation de la vitesse et de la mémoire des classifieurs d'espèces candidats (2e étape) sur Ethos-U55-256 (M55-HP de
# l'Alif E7) : Keras (pré-entraîné sur ImageNet) → TFLite int8 → Vela. La tête compte provisoirement 20 espèces, uniquement
# pour mesurer vitesse et mémoire. S'exécute uniquement sur l'ordinateur, rien n'est écrit sur la carte.
# Usage : .venv-yolo/bin/python u55/profile_classifier_u55.py
import os

# tf_keras (Keras 2), le mieux pris en charge par TFLiteConverter
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json

import cv2
import numpy as np
import tensorflow as tf

from u55_profile_common import OUT_DIR, calibration_images, run_vela

NUM_SPECIES = 20
# (nom, constructeur, alpha, côté d'entrée)
CANDIDATES = [
    ("mobilenet_v2_035", tf.keras.applications.MobileNetV2, 0.35, 128),
    ("mobilenet_v2_050", tf.keras.applications.MobileNetV2, 0.5, 128),
    ("mobilenet_v2_050", tf.keras.applications.MobileNetV2, 0.5, 160),
    ("mobilenet_v2_075", tf.keras.applications.MobileNetV2, 0.75, 160),
    ("mobilenet_v2_100", tf.keras.applications.MobileNetV2, 1.0, 160),
    ("mobilenet_v3_small_100", tf.keras.applications.MobileNetV3Small, 1.0, 160),
]


def build(ctor, alpha, size):
    extra = {"include_preprocessing": False} if ctor is tf.keras.applications.MobileNetV3Small else {}
    base = ctor(input_shape=(size, size, 3), alpha=alpha, include_top=False, weights="imagenet", pooling="avg", **extra)
    return tf.keras.Model(base.input, tf.keras.layers.Dense(NUM_SPECIES)(base.output))


def crops(images, size, per_image=4, seed=0):
    """Découpe des carrés aléatoires dans les images coco8, redimensionnés à la taille d'entrée et normalisés dans [-1, 1]
    (plage d'entrée de MobileNet)."""
    rng = np.random.default_rng(seed)
    out = []
    for img in images:
        h, w = img.shape[:2]
        for _ in range(per_image):
            s = int(rng.integers(min(h, w) // 3, min(h, w) + 1))
            y, x = int(rng.integers(0, h - s + 1)), int(rng.integers(0, w - s + 1))
            out.append(cv2.resize(img[y:y + s, x:x + s], (size, size)).astype(np.float32) / 127.5 - 1)
    return out


def to_int8_tflite(model, samples, path):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([s[None]] for s in samples)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    path.write_bytes(conv.convert())


def main():
    # Tous les fichiers intermédiaires vont dans build/ml_profile
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(OUT_DIR)
    images = calibration_images()
    results = []
    for name, ctor, alpha, size in CANDIDATES:
        tag = f"{name}_{size}"
        out_dir = OUT_DIR / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        model = build(ctor, alpha, size)
        tflite = out_dir / f"{tag}_int8.tflite"
        to_int8_tflite(model, crops(images, size), tflite)
        result = {"model": tag, "params_m": round(model.count_params() / 1e6, 2), **run_vela(tflite)}
        print(json.dumps(result), flush=True)
        results.append(result)
    (OUT_DIR / "classifier_u55_results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
