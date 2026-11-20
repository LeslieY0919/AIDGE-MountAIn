#!/usr/bin/env python3
# Exporte un classifieur ONNX vers l'Ethos-U55 avec Aidge : quantification INT8 (PTQ) → TOSA (aidge_tosa) → Vela.
# Le résultat, <out>/output/model_vela.tflite, se passe ensuite à scripts/board/build_img_class.sh (MODEL_PATH) pour
# être exécuté sur la carte par le kit d'évaluation ML d'Alif.
#
# Le modèle exporté prend une image int8 avec q = pixel - 128 : c'est la conversion que fait déjà le kit d'Alif,
# il n'y a donc aucune normalisation à faire sur la carte. Ce n'est PAS la convention du SDK du porteur
# (code = round(pixel / 255 * 127)) : pour le firmware du SDK, exporter avec scripts/npu/sdk_export_report.py.
#
# Par défaut l'entrée est en NCHW. Avec l'aidge_tosa officiel (celui de l'image docker/aidge/), c'est la seule
# disposition qui passe sur un MobileNetV2, au prix de deux Transpose autour de chaque convolution. Avec le fork
# du porteur, les Transpose disparaissent, mais l'export NHWC de ce script plante. Voir
# notes/2026-11-17-aidge-vers-npu.md.
#
# Demande aidge_tosa, aidge_export_arm_ethos et Vela : s'exécute dans l'environnement décrit par
# docker/aidge/Dockerfile. Sur une machine sans wheel Aidge (macOS), passer par scripts/npu/aidge_docker.sh :
#   scripts/npu/aidge_docker.sh push data/samples/ external/alif_ml-embedded-evaluation-kit/scripts/vela/ensemble_vela.ini
#   scripts/npu/aidge_docker.sh run python scripts/npu/export_classifier.py \
#       --onnx models/classifier/species_classifier.onnx --size 128
#   scripts/npu/aidge_docker.sh fetch exports/npu/species_classifier/output/model_vela.tflite
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from PIL import Image

import aidge_core  # à importer avant aidge_quantization
import aidge_backend_cpu  # noqa: F401  (enregistre le backend "cpu")
import aidge_onnx
import aidge_quantization
from aidge_export_arm_ethos.export import ethos_export_config, export

from mountain import paths
from mountain.preprocess import IMAGENET_MEAN as MEAN, IMAGENET_STD as STD


def load_images(dirs, size, count):
    """Images des dossiers donnés, recadrées au carré central et redimensionnées : tableaux uint8 [H, W, 3]."""
    files = sorted(p for d in dirs for p in Path(d).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise SystemExit(f"Aucune image de calibration dans {dirs}")
    images = []
    for path in files[:count]:
        img = Image.open(path).convert("RGB")
        side = min(img.size)
        left, top = (img.width - side) // 2, (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.BILINEAR)
        images.append(np.asarray(img, dtype=np.uint8))
    return images


def to_tensor(hwc, dtype=np.float32):
    t = aidge_core.Tensor(np.ascontiguousarray(hwc.transpose(2, 0, 1)[None].astype(dtype)))
    t.set_data_format(aidge_core.dformat.nchw)
    return t


def fold_input_normalization(src, dst):
    """Replie la normalisation ImageNet dans la première convolution.
    Le modèle d'origine attend n = (pixel / 255 - moyenne) / écart-type ; le modèle replié attend
    x = (pixel - 128) / 127, dans [-1, 1], l'intervalle que suppose la PTQ d'Aidge (q = 127 x = pixel - 128).
    Seul le bord de l'image diffère : le remplissage vaut pixel = 128 au lieu de la moyenne."""
    model = onnx.load(str(src))
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    data_input = next(i.name for i in graph.input if i.name not in inits)
    conv = next((n for n in graph.node if n.input and n.input[0] == data_input), None)
    if conv is None or conv.op_type != "Conv":
        raise SystemExit("La première couche n'est pas une convolution : normalisation non repliable")

    a = (127.0 / (255.0 * STD)).astype(np.float32)
    b = ((128.0 / 255.0 - MEAN) / STD).astype(np.float32)
    w = numpy_helper.to_array(inits[conv.input[1]]).astype(np.float32)  # [sorties, 3, kh, kw]
    bias = (numpy_helper.to_array(inits[conv.input[2]]).astype(np.float32)
            if len(conv.input) > 2 and conv.input[2] else np.zeros(w.shape[0], np.float32))
    bias = bias + (w * b[None, :, None, None]).sum(axis=(1, 2, 3))
    w = w * a[None, :, None, None]

    names = (f"{conv.input[1]}_fold", f"{conv.name or 'conv0'}_bias_fold")
    graph.initializer.extend([numpy_helper.from_array(w, names[0]), numpy_helper.from_array(bias, names[1])])
    del conv.input[1:]
    conv.input.extend(names)
    onnx.save(model, str(dst))


def load_graph(onnx_path, shape):
    graph = aidge_onnx.load_onnx(str(onnx_path))
    aidge_core.fuse_batchnorm(graph)
    aidge_core.remove_flatten(graph)
    graph.set_datatype(aidge_core.dtype.float32)
    graph.set_backend("cpu")
    graph.forward_dims(dims=[shape], allow_data_dependency=True)
    return graph


def top1(graph, tensors):
    """Classe gagnante du graphe pour chaque tenseur d'entrée."""
    scheduler = aidge_core.SequentialScheduler(graph)
    result = []
    for t in tensors:
        scheduler.forward(data=[t.clone()])
        node, idx = graph.get_ordered_outputs()[0]
        result.append(int(np.array(node.get_operator().get_output(idx)).reshape(-1).argmax()))
    return result


def vela_summary(output_dir):
    """Lit le résumé CSV écrit par Vela : temps d'inférence estimé et occupation mémoire."""
    files = sorted(Path(output_dir).glob("*_summary_*.csv"))
    if not files:
        return {}
    with open(files[0]) as f:
        row = next(csv.DictReader(f))
    return {
        "ms": round(float(row["inference_time"]) * 1e3, 2),
        "sram_kib": round(float(row["sram_memory_used"]), 1),
        "mram_kib": round(float(row["off_chip_flash_memory_used"]), 1),
        "gmac": round(float(row["nn_macs"]) / 1e9, 3),
    }


def tflite_signature(path):
    """Entrées et sorties du .tflite produit par Vela (nom, dimensions), pour vérifier ce que la carte devra fournir."""
    from ethosu.vela.tflite.Model import Model

    sg = Model.GetRootAsModel(Path(path).read_bytes(), 0).Subgraphs(0)

    def tensor(i):
        t = sg.Tensors(i)
        return f"{t.Name().decode()}{[int(d) for d in t.ShapeAsNumpy()]}"

    return {
        "inputs": [tensor(sg.Inputs(i)) for i in range(sg.InputsLength())],
        "outputs": [tensor(sg.Outputs(i)) for i in range(sg.OutputsLength())],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--size", type=int, default=224, help="côté de l'image d'entrée")
    ap.add_argument("--calib", nargs="+", default=[str(paths.SAMPLES)], help="dossiers d'images de calibration")
    ap.add_argument("--nb-calib", type=int, default=32)
    ap.add_argument("--input-norm", choices=("imagenet", "pm1"), default="imagenet",
                    help="normalisation avec laquelle le modèle a été entraîné : moyenne/écart-type ImageNet (repliée "
                         "dans la première convolution) ou (pixel - 127.5) / 127.5")
    ap.add_argument("--input-format", choices=("nchw", "nhwc"), default="nchw",
                    help="disposition de l'entrée du modèle exporté ; avec l'aidge_tosa officiel, nhwc évite les "
                         "Transpose mais échoue dès que le réseau contient des convolutions en profondeur avec "
                         "remplissage ; avec le fork du porteur, nhwc plante")
    ap.add_argument("--out", type=Path, default=None, help="par défaut exports/npu/<nom du modèle>")
    ap.add_argument("--accelerator", default="ethos-u55-256")
    ap.add_argument("--system-config", default="RTSS_HP_SRAM_MRAM")
    ap.add_argument("--vela-ini", type=Path, default=paths.VELA_INI,
                    help="même configuration Vela que tools/u55/ : poids en MRAM, activations en SRAM")
    args = ap.parse_args()

    out = args.out or paths.EXPORT_NPU / args.onnx.stem
    out.mkdir(parents=True, exist_ok=True)
    shape = [1, 3, args.size, args.size]
    images = load_images(args.calib, args.size, args.nb_calib)

    # 1. Référence flottante : le modèle d'origine, avec sa propre normalisation
    if args.input_norm == "imagenet":
        reference_inputs = [to_tensor((img / 255.0 - MEAN) / STD) for img in images]
        onnx_path = out / f"{args.onnx.stem}_folded.onnx"
        fold_input_normalization(args.onnx, onnx_path)
    else:
        reference_inputs = [to_tensor((img - 127.5) / 127.5) for img in images]
        onnx_path = args.onnx
    reference = top1(load_graph(args.onnx, shape), reference_inputs)

    # 2. Quantification INT8 après entraînement, sur le modèle qui prend x = (pixel - 128) / 127
    model = load_graph(onnx_path, shape)
    aidge_quantization.auto_assign_node_precision(
        graphview=model,
        act_precision=aidge_core.dtype.int8,
        weight_precision=aidge_core.dtype.int8,
        bias_precision=aidge_core.dtype.int32,
    )
    aidge_quantization.quantize_network(
        network=model,
        calibration_set=[to_tensor((img.astype(np.float32) - 128.0) / 127.0) for img in images],
        clipping_mode=aidge_quantization.Clipping.MSE,
        no_quant=False,
        optimize_signs=False,
        single_shift=False,
        fold_graph=True,
        fake_quantize=False,
        verbose=False,
    )
    model.forward_dims(allow_data_dependency=True)
    model.forward_dtype(dtype=aidge_core.dtype.int8)

    # 3. Vérification : le graphe int8, nourri avec q = pixel - 128, donne-t-il la même classe que le flottant ?
    model.set_backend("cpu")
    quantized = top1(model, [to_tensor(img.astype(np.int16) - 128, np.int8) for img in images])
    agreement = sum(r == q for r, q in zip(reference, quantized))

    # 4. Export TOSA puis compilation Vela. Le tenseur d'entrée ne sert qu'à décrire la signature (dimensions, type,
    # format). Avec l'aidge_tosa officiel : en NCHW, le graphe reste en NCHW et chaque convolution est encadrée de
    # deux Transpose (environ 4 fois plus lent d'après Vela sur un petit réseau) ; en NHWC, aidge_tosa convertit tout
    # le graphe et l'entrée du modèle devient une image entrelacée (ce qu'attend le kit d'Alif), mais le remplissage
    # des convolutions en profondeur est alors appliqué sur les mauvais axes et Vela refuse le fichier (constaté sur
    # MobileNetV2). Avec le fork du porteur, les Transpose disparaissent en NCHW, et l'export NHWC plante.
    if args.input_format == "nhwc":
        example = aidge_core.Tensor(np.zeros([1, args.size, args.size, 3], dtype=np.int8))
        example.set_data_format(aidge_core.dformat.nhwc)
    else:
        example = aidge_core.Tensor(np.zeros(shape, dtype=np.int8))
        example.set_data_format(aidge_core.dformat.nchw)
    config = ethos_export_config(
        accelerator=args.accelerator,
        memory_mode="Shared_Sram",
        system_config=args.system_config,
        config_ini=str(args.vela_ini),
        extra_flags=["--optimise", "Performance", "--arena-cache-size=2097152"],
    )
    export(model, str(out), config, example)

    result = {
        "model": args.onnx.stem, "size": args.size, "input_norm": args.input_norm, "input_format": args.input_format,
        "top1_float_vs_int8": f"{agreement}/{len(images)}", "float_top1": reference, "int8_top1": quantized,
        **vela_summary(out / "output"), **tflite_signature(out / "output" / "model_vela.tflite"),
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
    print(f"Modèle Vela : {out / 'output' / 'model_vela.tflite'}")


if __name__ == "__main__":
    main()
