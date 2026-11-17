#!/usr/bin/env python3
# Exporte un ONNX (une entrée NCHW dans [0, 1]) par la chaîne du SDK du porteur, puis résume le résultat.
#
# L'export est exactement celui du SDK : alif_aidge_sdk.export.export_model (PTQ int8 d'Aidge → aidge_tosa →
# Vela, puis model_io.json). Le modèle exporté suit donc la convention d'entrée du runtime du SDK : int8 NHWC,
# code = round(pixel / 255 * 127), zéro 0. Ce script ajoute seulement un résumé, <out>/result.json : estimation de
# Vela (temps, SRAM, MRAM, MAC) et corrélation entre le modèle flottant et le graphe int8, sortie par sortie.
#
# Outils demandés : ceux de CLAUDE.md §3 (fork aidge-tosa-fork, aidge_export_arm_ethos, alif_aidge_sdk). Pour un
# modèle qui garde des îlots flottants (Softmax du PSA de YOLOv10), aidge_export_arm_ethos doit venir de la branche
# feat/ethos-be-lowering du fork du porteur : voir notes/2026-11-17-aidge-vers-npu.md.
#
# L'image docker/aidge/ installe l'aidge_tosa officiel : y placer les dépôts du porteur en tête du PYTHONPATH.
#   F=/work/external/organisateur
#   scripts/npu/aidge_docker.sh run sh -c "PYTHONPATH=$F/aidge_export_ethos:$F/aidge-tosa-fork:$F/alif_aidge_sdk:/work/src \
#       python scripts/npu/sdk_export_report.py --onnx models/yolov10n.onnx --size 320 --calib data/samples --verify"
#
# Sans onnxruntime (absent de l'image), les sorties flottantes de référence, que le SDK utilise pour ajuster les
# échelles de sortie de model_io.json, sont calculées par Aidge en flottant sur le même ONNX.
import argparse
import csv
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image

import aidge_core  # à importer avant aidge_quantization (importé par le SDK)
import aidge_tosa
from alif_aidge_sdk import export as sdk
from aidge_export_arm_ethos.export import export as ethos_export  # la fonction : le paquet la réexporte

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def float_outputs_aidge(onnx_path, array):
    """Sorties du modèle flottant calculées par Aidge, à la place d'onnxruntime."""
    graph = sdk.load_model(onnx_path, list(array.shape))
    scheduler = aidge_core.SequentialScheduler(graph)
    scheduler.forward(data=[sdk.to_tensor(np.asarray(array, dtype=np.float32))])
    return [np.array(node.get_operator().get_output(idx)) for node, idx in graph.get_ordered_outputs()]


def calibration_arrays(dirs, size, count):
    """Images des dossiers (récursivement) : carré central, redimensionné, [0, 1], NCHW."""
    files = sorted(p for d in dirs for p in Path(d).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)[:count]
    if not files:
        raise SystemExit(f"aucune image dans {dirs}")
    arrays = []
    for path in files:
        img = Image.open(path).convert("RGB")
        side = min(img.size)
        left, top = (img.width - side) // 2, (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((size, size), Image.BILINEAR)
        arrays.append((np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None])
    return arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", type=Path, required=True, help="ONNX flottant, entrée NCHW dans [0, 1]")
    ap.add_argument("--size", type=int, required=True, help="côté de l'entrée carrée")
    ap.add_argument("--calib", nargs="+", required=True, help="dossiers d'images de calibration")
    ap.add_argument("--max-images", type=int, default=64)
    ap.add_argument("--target", choices=sorted(sdk.TARGETS), default="hp")
    ap.add_argument("--out", type=Path, help="défaut : exports/npu/sdk_<modèle>_<cible>")
    ap.add_argument("--verify", action="store_true", help="compare le TOSA au graphe Aidge (modèle de référence TOSA)")
    args = ap.parse_args()
    out = args.out or Path("exports/npu") / f"sdk_{args.onnx.stem}_{args.target}"
    out.mkdir(parents=True, exist_ok=True)
    aidge_core.Log.set_console_level(aidge_core.Level.Error)

    ethos_be = "ethos_be" in inspect.signature(ethos_export).parameters
    print(f"aidge_tosa : {aidge_tosa.__file__}", flush=True)
    print(f"aidge_export_arm_ethos : {inspect.getsourcefile(ethos_export)} (ethos_be : {ethos_be})", flush=True)
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        sdk.run_float_onnx = lambda onnx_path, float_input, input_name=None: float_outputs_aidge(onnx_path, float_input)
        print("onnxruntime absent : sorties flottantes de référence calculées par Aidge", flush=True)

    arrays = calibration_arrays(args.calib, args.size, args.max_images)
    print(f"{len(arrays)} images de calibration", flush=True)
    _, aidge_outputs, float_outputs, io = sdk.export_model(
        args.onnx, out, [1, 3, args.size, args.size], arrays, nb_samples=len(arrays),
        verify_tosa=args.verify, target=args.target)

    # Accord flottant / int8 (déquantifié avec l'échelle ajustée par le SDK) sur l'image de contrôle du SDK
    outputs = []
    for desc, ref, codes in zip(io["outputs"], float_outputs, aidge_outputs):
        ref = np.asarray(ref, dtype=np.float64).ravel()
        deq = np.asarray(codes, dtype=np.float64).ravel() * desc["dequantize_scale"]
        corr = float(np.corrcoef(ref, deq)[0, 1]) if ref.std() > 0 and deq.std() > 0 else None
        outputs.append({"shape_nchw": desc["shape_nchw"], "dequantize_scale": desc["dequantize_scale"],
                        "corr_float_int8": None if corr is None else round(corr, 4)})

    with open(sorted((out / "output").glob("*_summary_*.csv"))[0]) as f:
        summary = next(csv.DictReader(f))
    result = {
        "onnx": str(args.onnx), "size": args.size, "target": args.target, "images": len(arrays),
        "aidge_tosa": aidge_tosa.__file__, "ethos_be": ethos_be,
        "ms": round(float(summary["inference_time"]) * 1e3, 2),
        "sram_kib": round(float(summary["sram_memory_used"]), 1),
        "mram_kib": round(float(summary["off_chip_flash_memory_used"]), 1),
        "gmac": round(float(summary["nn_macs"]) / 1e9, 3),
        "input": io["input"], "outputs": outputs,
    }
    if (out / "verify.json").exists():
        result["verify"] = json.loads((out / "verify.json").read_text())
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "input"}), flush=True)


if __name__ == "__main__":
    main()
