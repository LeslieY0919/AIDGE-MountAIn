#!/usr/bin/env python3
# Implémentation de référence du décodage CPU des sorties brutes de la branche one-to-one de YOLOv10 (le code C du M55 suivra
# celle-ci), et comparaison entre le TFLite int8 et le modèle flottant.
# Usage : .venv-yolo/bin/python tools/u55/yolov10_decode.py [--variant silu_psa] [--imgsz 256]
import argparse

import numpy as np

STRIDES = (8, 16, 32)
REG_MAX = 16


def decode(outputs, conf=0.25, max_det=20):
    """outputs : {"box_p8": [H,W,64], "cls_p8": [H,W,nc], ...}, en flottant, NHWC, sans dimension batch.
    Le seuillage se fait d'abord sur les logits de classe ; le DFL et le décodage des boîtes ne portent que sur les ancres
    retenues ; la branche one-to-one n'a pas besoin de NMS. Renvoie [x1, y1, x2, y2, score, classe] en pixels de l'entrée."""
    logit_thr = np.log(conf / (1 - conf))  # sigmoid(x) > conf  ⇔  x > logit(conf)
    dets = []
    for s in STRIDES:
        box, cls = outputs[f"box_p{s}"], outputs[f"cls_p{s}"]
        best = cls.max(-1)
        for y, x in zip(*np.nonzero(best > logit_thr)):
            d = box[y, x].reshape(4, REG_MAX)  # ordre des canaux : côté × 16 intervalles
            p = np.exp(d - d.max(-1, keepdims=True))
            ltrb = (p / p.sum(-1, keepdims=True)) @ np.arange(REG_MAX)  # espérance DFL
            cx, cy = x + 0.5, y + 0.5
            k = int(cls[y, x].argmax())
            dets.append([(cx - ltrb[0]) * s, (cy - ltrb[1]) * s, (cx + ltrb[2]) * s, (cy + ltrb[3]) * s,
                         1 / (1 + np.exp(-cls[y, x, k])), k])
    dets.sort(key=lambda r: -r[4])
    return np.array(dets[:max_det]).reshape(-1, 6)


def run_tflite(path, image, names):
    """Exécute le modèle int8 avec l'interpréteur TFLite sur l'ordinateur (le même fichier qu'avant Vela) et déquantifie les
    sorties. Dans la signature, output_0…output_5 suivent l'ordre des sorties de l'export ONNX (names)."""
    from ai_edge_litert.interpreter import Interpreter

    runner = Interpreter(model_path=str(path)).get_signature_runner()
    s, z = runner.get_input_details()["images"]["quantization"]
    res = runner(images=np.clip(np.round(image[None] / 255.0 / s + z), -128, 127).astype(np.int8))
    details = runner.get_output_details()
    out = {}
    for key, val in res.items():
        s, z = details[key]["quantization"]
        out[names[int(key.rsplit("_", 1)[1])]] = (val[0].astype(np.float32) - z) * s
    return out


def main():
    import torch

    from profile_yolov10_u55 import OUT_DIR, build
    from u55_profile_common import calibration_images, letterbox

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="silu_psa")
    ap.add_argument("--imgsz", type=int, default=256)
    args = ap.parse_args()

    names = [f"{k}_p{s}" for s in STRIDES for k in ("box", "cls")]
    model = build(args.variant)
    tflite = OUT_DIR / f"yolov10n_{args.variant}_{args.imgsz}" / f"yolov10n_{args.variant}_{args.imgsz}_full_integer_quant.tflite"
    for i, img in enumerate(calibration_images()):
        im = letterbox(img, args.imgsz)
        with torch.no_grad():
            ref = model(torch.from_numpy(im).permute(2, 0, 1)[None].float() / 255)
        ref = {n: t[0].permute(1, 2, 0).numpy() for n, t in zip(names, ref)}
        f, q = decode(ref), decode(run_tflite(tflite, im.astype(np.float32), names))
        fmt = lambda d: ", ".join(f"c{int(r[5])}:{r[4]:.2f}" for r in d[:4]) or "-"
        print(f"img{i}: float [{fmt(f)}] | int8 [{fmt(q)}]")


if __name__ == "__main__":
    main()
