import subprocess
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

RAW_ONNX = Path("detector_v6_trained_640.onnx")
CUT_ONNX = Path("MDV6-yolov10-c_cut_folded.onnx")

CPP_BUILD_DIR = Path("export_cpp_detector/build")
CPP_BIN = "./run_export"
IMG_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45
CLASS_NAMES = ["animal", "person", "vehicle"]


def preprocess_letterbox(image_path: Path, size: int = 640):
    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    scale = min(size / orig_w, size / orig_h)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    resized = orig_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(arr, (2, 0, 1))[None, ...]  # [1, 3, 640, 640] NCHW
    return np.ascontiguousarray(tensor, dtype=np.float32), orig_img, (scale, pad_x, pad_y)


def make_anchors(img_size: int = 640, strides=(8, 16, 32)):
    anchors_list = []
    strides_list = []
    for s in strides:
        grid = img_size // s
        iy, ix = np.meshgrid(np.arange(grid, dtype=np.float32), np.arange(grid, dtype=np.float32), indexing="ij")
        pts = np.stack([(ix + 0.5).ravel(), (iy + 0.5).ravel()], axis=0)  # [2, grid*grid]
        anchors_list.append(pts)
        strides_list.append(np.full((1, grid * grid), s, dtype=np.float32))
    return np.concatenate(anchors_list, axis=1)[None, ...], np.concatenate(strides_list, axis=1)[None, ...]


def postprocess_yolov10_head(raw_concat: np.ndarray, img_size: int = 640) -> np.ndarray:
    """Reproduit les 11 derniers noeuds ONNX : dist2bbox(lt, rb) * stride + sigmoid(cls)."""
    anchors, stride_mat = make_anchors(img_size)
    lt = raw_concat[:, 0:2, :]
    rb = raw_concat[:, 2:4, :]
    x1y1 = anchors - lt
    x2y2 = anchors + rb
    xyxy = np.concatenate([x1y1, x2y2], axis=1) * stride_mat
    cls_probs = 1.0 / (1.0 + np.exp(-np.clip(raw_concat[:, 4:, :], -60.0, 60.0)))
    return np.concatenate([xyxy, cls_probs], axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def decode_yolov10(final_out: np.ndarray, meta, orig_size):
    scale, pad_x, pad_y = meta
    orig_w, orig_h = orig_size
    pred = final_out[0].T  # [8400, 7]

    boxes_xyxy = pred[:, :4]
    cls_scores = pred[:, 4:]
    class_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(len(pred)), class_ids]

    mask = scores >= CONF_THRESH
    boxes_xyxy, scores, class_ids = boxes_xyxy[mask], scores[mask], class_ids[mask]
    if len(boxes_xyxy) == 0:
        return []

    x1 = np.clip((boxes_xyxy[:, 0] - pad_x) / scale, 0, orig_w)
    y1 = np.clip((boxes_xyxy[:, 1] - pad_y) / scale, 0, orig_h)
    x2 = np.clip((boxes_xyxy[:, 2] - pad_x) / scale, 0, orig_w)
    y2 = np.clip((boxes_xyxy[:, 3] - pad_y) / scale, 0, orig_h)

    valid = (x2 > x1) & (y2 > y1)
    boxes = np.stack([x1[valid], y1[valid], x2[valid], y2[valid]], axis=1)
    scores = scores[valid]
    class_ids = class_ids[valid]
    if len(boxes) == 0:
        return []

    keep = nms(boxes, scores, IOU_THRESH)
    return [(boxes[i], float(scores[i]), int(class_ids[i])) for i in keep]


def main():
    if len(sys.argv) < 2:
        print("Usage : python test_cpp_detector.py <image.jpg>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    input_tensor, orig_img, meta = preprocess_letterbox(img_path, IMG_SIZE)

    # 1. Référence ONNX sur le modèle coupé (périmètre exact de l'export C++)
    cut_sess = ort.InferenceSession(str(CUT_ONNX), providers=["CPUExecutionProvider"])
    cut_onnx_out = cut_sess.run(None, {cut_sess.get_inputs()[0].name: input_tensor})[0]
    num_elements = int(cut_onnx_out.size)

    # 2. Exécution du binaire C++ Aidge
    input_bin = CPP_BUILD_DIR / "input.bin"
    output_bin = CPP_BUILD_DIR / "output.bin"
    input_tensor.tofile(input_bin)

    subprocess.run([CPP_BIN, str(num_elements)], cwd=CPP_BUILD_DIR, check=True)
    cpp_raw = np.fromfile(output_bin, dtype=np.float32)

    # 3. Vérifier l'alignement mémoire (NCHW vs NHWC) face à MDV6-yolov10-c_cut_folded.onnx
    cand_nchw = cpp_raw.reshape(cut_onnx_out.shape)
    cand_nhwc = cpp_raw.reshape(1, 8400, 7).transpose(0, 2, 1)

    diff_nchw = np.abs(cut_onnx_out - cand_nchw).max()
    diff_nhwc = np.abs(cut_onnx_out - cand_nhwc).max()
    print(f"[1. Validation graphe coupé] Écart max NCHW : {diff_nchw:.6f} | Écart max NHWC : {diff_nhwc:.6f}")

    if diff_nhwc < diff_nchw:
        print("   -> Format sortie C++ détecté : NHWC (1, 8400, 7) transposé en (1, 7, 8400)")
        cpp_aligned = cand_nhwc
    else:
        print("   -> Format sortie C++ détecté : NCHW (1, 7, 8400)")
        cpp_aligned = cand_nchw

    # 4. Vérifier que le post-traitement NumPy reproduit exactement les 11 noeuds ONNX
    raw_sess = ort.InferenceSession(str(RAW_ONNX), providers=["CPUExecutionProvider"])
    final_raw_out = raw_sess.run(None, {raw_sess.get_inputs()[0].name: input_tensor})[0]
    post_diff = np.abs(postprocess_yolov10_head(cut_onnx_out, IMG_SIZE) - final_raw_out).max()
    print(f"[2. Validation post-traitement NumPy] Écart max vs ONNX complet : {post_diff:.6f}")

    # 5. Appliquer le post-traitement sur la sortie C++ et comparer au modèle complet
    final_cpp_out = postprocess_yolov10_head(cpp_aligned, IMG_SIZE)
    print(f"[3. Validation bout-en-bout] Écart max final ONNX vs C++ : {np.abs(final_raw_out - final_cpp_out).max():.6f}")

    # 6. Décoder et tracer les boîtes
    detections = decode_yolov10(final_cpp_out, meta, orig_img.size)
    print(f"\nDétections retenues (seuil >= {CONF_THRESH}) : {len(detections)}")

    draw = ImageDraw.Draw(orig_img)
    for box, score, cls_id in detections:
        x1, y1, x2, y2 = box
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"classe_{cls_id}"
        label = f"{name} ({score:.2f})"
        print(f"  - {label} : [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
        draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=3)
        draw.text((x1 + 4, max(0, y1 - 14)), label, fill=(255, 255, 0))

    out_img_path = Path("detection_cpp_result.jpg")
    orig_img.save(out_img_path)
    print(f"\nImage annotée enregistrée dans : {out_img_path.resolve()}")


if __name__ == "__main__":
    main()
