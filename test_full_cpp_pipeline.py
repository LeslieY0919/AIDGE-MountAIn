import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

CLS_ONNX = Path("species_classifier.onnx")
CLASS_NAMES_PATH = Path("class_names.json")

DET_CPP_DIR = Path("export_cpp_detector_int8/build")
CLS_CPP_DIR = Path("export_cpp_classifier_int8/build")
CPP_BIN = "./run_export"

CLS_IMG_SIZE = 128
CONF_THRESH = 0.25

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DET_LINE_RE = re.compile(
    r"^\s*\[(\w+)\]\s+conf=([0-9.]+)\s+bbox=\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]"
)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def run_cpp_detector(image_path: Path, conf_thresh: float = 0.25):
    proc = subprocess.run(
        [CPP_BIN, str(image_path.resolve()), str(conf_thresh)],
        cwd=DET_CPP_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    detections = []
    for line in proc.stdout.splitlines():
        m = DET_LINE_RE.match(line)
        if m:
            label = m.group(1)
            score = float(m.group(2))
            box = (float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)))
            detections.append((box, score, label))
    return detections, proc.stdout.strip()


def run_cpp_classifier(tensor_4d: np.ndarray) -> np.ndarray:
    in_bin = CLS_CPP_DIR / "cls_input.bin"
    out_bin = CLS_CPP_DIR / "cls_logits.bin"
    np.ascontiguousarray(tensor_4d, dtype=np.float32).tofile(in_bin)
    subprocess.run([CPP_BIN], cwd=CLS_CPP_DIR, check=True)
    return np.fromfile(out_bin, dtype=np.float32)


def main():
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_wildlife_1mp.jpg")
    if not img_path.exists():
        print(f"Erreur : image introuvable ({img_path})")
        sys.exit(1)

    with open(CLASS_NAMES_PATH, "r") as f:
        species_names = json.load(f)

    orig_img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    # 1. Étage 1 : Détecteur 100% C++ (prétraitement + inférence Aidge + décodage DFL/NMS)
    detections, det_stdout = run_cpp_detector(img_path, CONF_THRESH)
    print("=== Sortie Détecteur C++ (Étage 1) ===")
    print(det_stdout)

    # 2. Calibration automatique de l'ordre des axes du classifieur C++ (NHWC vs NCHW)
    cls_sess = ort.InferenceSession(str(CLS_ONNX), providers=["CPUExecutionProvider"])
    cls_in_name = cls_sess.get_inputs()[0].name

    rng = np.random.default_rng(42)
    probe_nchw = rng.standard_normal((1, 3, CLS_IMG_SIZE, CLS_IMG_SIZE), dtype=np.float32)
    probe_nhwc = np.ascontiguousarray(np.transpose(probe_nchw, (0, 2, 3, 1)))

    ref_probe_logits = cls_sess.run(None, {cls_in_name: probe_nchw})[0][0]
    diff_nhwc = float(np.abs(ref_probe_logits - run_cpp_classifier(probe_nhwc)).max())
    diff_nchw = float(np.abs(ref_probe_logits - run_cpp_classifier(probe_nchw)).max())

    if diff_nhwc < diff_nchw:
        cls_layout, cls_cal_diff = "NHWC", diff_nhwc
    else:
        cls_layout, cls_cal_diff = "NCHW", diff_nchw

    print(f"\n=== Calibration Classifieur C++ (Étage 2) ===")
    print(f"Format mémoire détecté : {cls_layout} | Écart max logits ONNX vs C++ : {cls_cal_diff:.2e}")

    # 3. Enchaînement Détecteur C++ -> Recadrage HD -> Classifieur C++
    print(f"\n=== Résultat Pipeline Complet 100% C++ ({len(detections)} détection(s)) ===")
    annotated = orig_img.copy()
    draw = ImageDraw.Draw(annotated)

    for idx, (box, det_score, det_label) in enumerate(detections):
        x1, y1, x2, y2 = box
        ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
        ix2, iy2 = min(orig_w, max(ix1 + 1, int(round(x2)))), min(orig_h, max(iy1 + 1, int(round(y2))))

        if det_label == "animal":
            crop = orig_img.crop((ix1, iy1, ix2, iy2)).resize((CLS_IMG_SIZE, CLS_IMG_SIZE), Image.Resampling.BILINEAR)
            arr = (np.asarray(crop, dtype=np.float32) / 255.0 - MEAN) / STD
            x_nchw = np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...])
            x_nhwc = np.ascontiguousarray(arr[None, ...])

            onnx_logits = cls_sess.run(None, {cls_in_name: x_nchw})[0][0]
            cpp_logits = run_cpp_classifier(x_nhwc if cls_layout == "NHWC" else x_nchw)

            logit_diff = float(np.abs(onnx_logits - cpp_logits).max())
            cpp_probs = softmax(cpp_logits)
            onnx_probs = softmax(onnx_logits)

            cpp_id = int(np.argmax(cpp_probs))
            onnx_id = int(np.argmax(onnx_probs))
            cpp_species = species_names[cpp_id]
            onnx_species = species_names[onnx_id]

            print(
                f"#{idx} [animal det={det_score*100:.1f}%] Box=({ix1},{iy1},{ix2},{iy2}) -> "
                f"C++: {cpp_species.upper()} ({cpp_probs[cpp_id]*100:.1f}%) | "
                f"ONNX: {onnx_species.upper()} ({onnx_probs[onnx_id]*100:.1f}%) | "
                f"max_logit_diff={logit_diff:.2e}"
            )
            tag = f"{cpp_species} ({cpp_probs[cpp_id]*100:.1f}%)"
            draw.rectangle([ix1, iy1, ix2, iy2], outline=(50, 255, 80), width=4)
            draw.text((ix1 + 6, max(0, iy1 - 18)), tag, fill=(50, 255, 80))
        else:
            print(f"#{idx} [{det_label} det={det_score*100:.1f}%] Box=({ix1},{iy1},{ix2},{iy2}) -> Non-animal")
            tag = f"{det_label} ({det_score*100:.1f}%)"
            draw.rectangle([ix1, iy1, ix2, iy2], outline=(255, 180, 0), width=3)
            draw.text((ix1 + 6, max(0, iy1 - 18)), tag, fill=(255, 180, 0))

    out_path = Path("full_cpp_pipeline_result.jpg")
    annotated.save(out_path)
    print(f"\nImage annotée enregistrée : {out_path.resolve()}")


if __name__ == "__main__":
    main()
