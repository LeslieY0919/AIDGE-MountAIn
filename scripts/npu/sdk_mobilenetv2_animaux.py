#!/usr/bin/env python3
# MobileNetV2 de l'exemple du SDK du porteur, réduit aux 398 classes animales d'ImageNet : démonstration sur la
# carte en attendant notre classifieur.
#
# Dans ImageNet, les indices 0 à 397 sont des animaux (0 tench … 397 puffer ; 398 abacus est le premier objet). Le
# script ne garde que ces 398 sorties de la dernière convolution 1x1 : même réseau, mêmes poids, aucun
# réentraînement. Il exporte ensuite par alif_aidge_sdk.export.export_model, avec les réglages par défaut de
# examples/mobilenetv2/model/export.py (PTQ int8, écrêtage MSE, top-5) et le nom des 398 classes dans
# model_io.json. Il affiche enfin le top-3 flottant et int8 de chaque image de calibration : c'est la référence à
# comparer avec l'écran de la carte.
#
# Le pré-traitement et les noms de classes viennent de l'exemple : examples/mobilenetv2 doit être dans le
# PYTHONPATH (module model). L'entrée est l'ONNX préparé par l'exemple (python -m model.prepare_onnx).
#   S=external/organisateur/alif_aidge_sdk
#   PYTHONPATH=$S/examples/mobilenetv2:$PYTHONPATH python scripts/npu/sdk_mobilenetv2_animaux.py \
#       --onnx $S/examples/mobilenetv2/models/mobilenetv2_224.onnx --calib data/samples datasets/coco8/images --verify
# Suite (génération des sources, compilation, flash) : docs/GUIDE_CARTE.md.
#
# Sans onnxruntime (absent de l'image docker/aidge/), les sorties flottantes de référence sont calculées par Aidge
# sur le même ONNX, comme dans sdk_export_report.py.
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

import aidge_core  # à importer avant aidge_quantization (importé par le SDK)
from alif_aidge_sdk import export as sdk
from model import preprocess  # examples/mobilenetv2 du SDK

NB_ANIMAUX = 398
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def float_outputs_aidge(onnx_path, float_input, input_name=None):
    """Sorties du modèle flottant calculées par Aidge, à la place d'onnxruntime."""
    array = np.asarray(float_input, dtype=np.float32)
    graph = sdk.load_model(onnx_path, list(array.shape))
    aidge_core.SequentialScheduler(graph).forward(data=[sdk.to_tensor(array)])
    return [np.array(node.get_operator().get_output(idx)) for node, idx in graph.get_ordered_outputs()]


def restreindre(src, dst):
    """Copie de l'ONNX dont la convolution de sortie ne garde que les NB_ANIMAUX premières classes."""
    model = onnx.load(str(src))
    graph = model.graph
    sortie = graph.output[0].name
    derniere = next(n for n in graph.node if sortie in n.output)
    if derniere.op_type != "Conv":
        raise SystemExit(f"attendu une convolution 1x1 en sortie, trouvé {derniere.op_type} ({derniere.name})")
    inits = {i.name: i for i in graph.initializer}
    for nom in derniere.input[1:]:  # poids [1000, 1280, 1, 1], puis biais [1000] s'il existe
        tableau = numpy_helper.to_array(inits[nom])
        if tableau.shape[0] != 1000:
            raise SystemExit(f"{nom} : {tableau.shape[0]} sorties, 1000 attendues (ONNX préparé par l'exemple ?)")
        inits[nom].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(tableau[:NB_ANIMAUX]), nom))
        print(f"{derniere.name} : {nom} {list(tableau.shape)} -> {list(tableau[:NB_ANIMAUX].shape)}", flush=True)
    graph.output[0].type.tensor_type.shape.dim[1].dim_value = NB_ANIMAUX
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def top3(logits, classes):
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return ", ".join(f"{classes[i]} {100 * p[i]:.0f}%" for i in np.argsort(p)[::-1][:3])


def main():
    ap = argparse.ArgumentParser(description="MobileNetV2 du SDK réduit aux 398 classes animales d'ImageNet")
    ap.add_argument("--onnx", type=Path, required=True, help="ONNX préparé par l'exemple (python -m model.prepare_onnx)")
    ap.add_argument("--calib", nargs="+", required=True, help="dossiers d'images de calibration (parcourus récursivement)")
    ap.add_argument("--target", choices=sorted(sdk.TARGETS), default="hp")
    ap.add_argument("--out", type=Path, help="défaut : exports/npu/sdk_mobilenetv2_animaux_<cible>")
    ap.add_argument("--verify", action="store_true", help="compare le TOSA au graphe Aidge (modèle de référence TOSA)")
    args = ap.parse_args()
    out = args.out or Path("exports/npu") / f"sdk_mobilenetv2_animaux_{args.target}"
    out.mkdir(parents=True, exist_ok=True)
    aidge_core.Log.set_console_level(aidge_core.Level.Error)
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        sdk.run_float_onnx = float_outputs_aidge
        print("onnxruntime absent : sorties flottantes de référence calculées par Aidge", flush=True)

    onnx_animaux = out / "mobilenetv2_224_animaux.onnx"
    restreindre(args.onnx, onnx_animaux)

    labels_path = Path(preprocess.__file__).resolve().parent / "imagenet_labels.txt"
    labels = [l.strip() for l in labels_path.read_text().splitlines() if l.strip()]
    if len(labels) != 1000:
        raise SystemExit(f"{labels_path} : {len(labels)} classes, 1000 attendues")
    classes = [l.split(",")[0][:24] for l in labels[:NB_ANIMAUX]]  # même format que l'exemple (ligne de l'écran)

    images = sorted(p for d in args.calib for p in Path(d).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"aucune image dans {args.calib}")
    arrays = [preprocess.to_input_tensor(preprocess.to_square_rgb(preprocess.load_rgb(p))) for p in images]
    print(f"{len(arrays)} images de calibration", flush=True)
    model, _, _, io = sdk.export_model(
        onnx_animaux, out, [1, 3, preprocess.IMG_SIZE, preprocess.IMG_SIZE], arrays, nb_samples=len(arrays),
        verify_tosa=args.verify, check_array=arrays[0], params={"classes": classes, "top_k": 5},
        model_name="MobileNetV2 (mobilenetv2-7), 398 classes animales d'ImageNet, normalisation repliée, aidge",
        target=args.target)

    # Top-3 flottant et int8 (graphe Aidge, déquantifié avec l'échelle de model_io.json) sur chaque image
    scale = io["outputs"][0]["dequantize_scale"]
    for path, array in zip(images, arrays):
        flottant = np.asarray(sdk.run_float_onnx(onnx_animaux, array)[0], dtype=np.float64).reshape(-1)
        int8 = sdk.run_aidge(model, sdk.quantize_input(array))[0].astype(np.float64).reshape(-1) * scale
        print(f"{path.name}\n    float {top3(flottant, classes)}\n    int8  {top3(int8, classes)}", flush=True)


if __name__ == "__main__":
    main()
