#!/usr/bin/env python3
# Évaluation de la vitesse et de la mémoire de YOLOv10n sur Ethos-U55-256 (M55-HP de l'Alif E7) :
# PyTorch → ONNX → TFLite int8 (onnx2tf) → Vela. S'exécute uniquement sur l'ordinateur, rien n'est écrit sur la carte.
# Usage : .venv-yolo/bin/python tools/u55/profile_yolov10_u55.py [--imgsz 256 320]
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mountain import paths

from u55_profile_common import OUT_DIR, calibration_images, letterbox, run_vela

WEIGHTS = paths.PRETRAINED_DIR / "yolov10n.pt"

# Variantes : d'origine ; tous les SiLU remplacés par ReLU (ReLU se fusionne dans la convolution, SiLU demande une table
# et une multiplication en plus) ; puis sans le bloc d'attention PSA.
# Les modèles modifiés ne sont pas réentraînés : ils servent à comparer vitesse et mémoire, pas la précision.
VARIANTS = ("silu_psa", "relu_psa", "relu_nopsa")


class RawOutputs(nn.Module):
    """Retire TopK et le décodage DFL : ne sort que les distributions de boîtes et les logits de classes bruts de la branche
    one-to-one pour chaque échelle (6 tenseurs). Chaque sortie a sa propre quantification int8 ; le seuillage, le DFL et
    le décodage des boîtes se font sur le M55, uniquement pour les ancres au-dessus du seuil."""

    def __init__(self, det):
        super().__init__()
        self.layers, self.save, self.head = det.model, det.save, det.model[-1]

    def forward(self, x):
        y = []
        for m in self.layers[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        outs = []
        for i, j in enumerate(self.head.f):
            outs += [self.head.one2one_cv2[i](y[j]), self.head.one2one_cv3[i](y[j])]
        return tuple(outs)


def build(variant):
    from ultralytics import YOLO
    from ultralytics.nn.modules import PSA

    det = YOLO(str(WEIGHTS)).model.float().eval()
    # Indiquer end2end avant la fusion, sinon fuse() supprime la branche one-to-one (sans NMS)
    det.model[-1].end2end = True
    det = det.fuse(verbose=False)
    if variant.startswith("relu"):
        for parent in list(det.modules()):
            for name, child in parent.named_children():
                if isinstance(child, nn.SiLU):
                    setattr(parent, name, nn.ReLU())
    if variant.endswith("nopsa"):
        for k, m in enumerate(det.model):
            if isinstance(m, PSA):
                skip = nn.Identity()
                skip.f, skip.i = m.f, m.i
                det.model[k] = skip
    return RawOutputs(det).eval()


def export_onnx(model, size, path):
    import onnx
    import onnxslim

    names = [f"{kind}_p{stride}" for stride in (8, 16, 32) for kind in ("box", "cls")]
    with torch.no_grad():
        torch.onnx.export(model, torch.zeros(1, 3, size, size), str(path), opset_version=17,
                          input_names=["images"], output_names=names, dynamo=False)
    onnx.save(onnxslim.slim(onnx.load(str(path))), str(path))


def to_int8_tflite(onnx_path, calib, out_dir):
    """Entièrement int8 (entrées et sorties comprises), quantification par canal ; les données de calibration sont en RGB
    0–255 et normalisées par /255 comme dans le modèle."""
    import onnx2tf
    from ultralytics.utils.downloads import attempt_download_asset

    # onnx2tf cherche ces données d'exemple dans le dossier courant pour vérifier les sorties ; son propre lien ne marche plus,
    # on utilise la copie d'Ultralytics (comme le fait l'export d'Ultralytics)
    sample = Path("calibration_image_sample_data_20x128x128x3_float32.npy")
    if not sample.exists():
        attempt_download_asset(f"{sample}.zip", unzip=True, delete=True)

    calib_file = out_dir / "calib.npy"
    np.save(calib_file, calib)
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(out_dir),
        not_use_onnxsim=True,
        verbosity="error",
        output_integer_quantized_tflite=True,
        quant_type="per-channel",
        input_quant_dtype="int8",
        output_quant_dtype="int8",
        custom_input_op_name_np_data_path=[["images", str(calib_file), [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]],
        output_signaturedefs=True,
        copy_onnx_input_output_names_to_tflite=True,
    )
    return out_dir / f"{onnx_path.stem}_full_integer_quant.tflite"


def prune(out_dir):
    """onnx2tf génère beaucoup de fichiers inutiles (SavedModel, float16, quantification dynamique, activations int16, données de
    calibration…) : on ne garde que l'ONNX, le modèle entièrement int8 et les résultats de Vela."""
    for p in out_dir.iterdir():
        if p.name.startswith("vela") or p.name.endswith((".onnx", "_full_integer_quant.tflite")):
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, nargs="+", default=[256, 320])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    args = ap.parse_args()

    # Tous les fichiers intermédiaires vont dans artifacts/u55_profile
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(OUT_DIR)
    images = calibration_images()
    results = []
    for variant in args.variants:
        model = build(variant)
        for size in args.imgsz:
            name = f"yolov10n_{variant}_{size}"
            out_dir = OUT_DIR / name
            out_dir.mkdir(parents=True, exist_ok=True)
            onnx_path = out_dir / f"{name}.onnx"
            export_onnx(model, size, onnx_path)
            calib = np.stack([letterbox(img, size) for img in images]).astype(np.float32)
            tflite = to_int8_tflite(onnx_path, calib, out_dir)
            result = {"model": name, **run_vela(tflite)}
            prune(out_dir)
            print(json.dumps(result), flush=True)
            results.append(result)
    # Fusion par nom de modèle avec les résultats existants : les exécutions successives ne s'écrasent pas
    path = OUT_DIR / "yolov10n_u55_results.json"
    done = {r["model"] for r in results}
    old = json.loads(path.read_text()) if path.exists() else []
    path.write_text(json.dumps([r for r in old if r["model"] not in done] + results, indent=1))


if __name__ == "__main__":
    main()
