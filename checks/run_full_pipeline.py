import json
import numpy as np
from PIL import Image, ImageDraw
import aidge_core
import aidge_backend_cpu
import aidge_onnx

def load_aidge_model(onnx_path, input_shape):
    graph = aidge_onnx.load_onnx(onnx_path)
    for node in graph.get_nodes():
        if node.type() == "Producer":
            op = node.get_operator()
            if hasattr(op, "set_constant"):
                op.set_constant(True)
            elif hasattr(op, "attr") and hasattr(op.attr, "constant"):
                op.attr.constant = True
    aidge_core.constant_folding(graph)
    graph.forward_dims([input_shape])
    graph.set_datatype(aidge_core.dtype.float32)
    graph.set_backend("cpu")
    scheduler = aidge_core.SequentialScheduler(graph)
    out_node = list(graph.get_output_nodes())[0]
    return graph, scheduler, out_node

# 1. Chargement des deux modèles dans Aidge + noms des classes
with open("class_names.json", "r") as f:
    class_names = json.load(f)

_, det_sched, det_out = load_aidge_model("MDV6-yolov10-c.onnx", [1, 3, 320, 320])
_, cls_sched, cls_out = load_aidge_model("species_classifier.onnx", [1, 3, 128, 128])

orig_img = Image.open("samples/test_wildlife_1mp.jpg").convert("RGB")
W, H = orig_img.size

# 2. Étage 1 : Détection MDV6 (passe globale + 4 tuiles 1 MP)
def detect_region(crop_img, ox, oy, sx, sy, is_global=False):
    resized = crop_img.resize((320, 320), Image.BILINEAR)
    x = np.ascontiguousarray(np.transpose(np.array(resized, dtype=np.float32) / 255.0, (2, 0, 1))[None, ...])
    det_sched.forward(data=[aidge_core.Tensor(x)])
    preds = np.array(det_out.get_operator().get_output(0))[0].T
    boxes = []
    for row in preds[preds[:, 4] > 0.35]:
        cx, cy, bw, bh, score = row[:5]
        boxes.append([
            max(0, ox + (cx - bw / 2) * sx),
            max(0, oy + (cy - bh / 2) * sy),
            min(W, ox + (cx + bw / 2) * sx),
            min(H, oy + (cy + bh / 2) * sy),
            float(score),
            is_global,
        ])
    return boxes

all_boxes = detect_region(orig_img, 0, 0, W / 320.0, H / 320.0, is_global=True)
overlap = 0.15
tw, th = int(W * (0.5 + overlap / 2)), int(H * (0.5 + overlap / 2))
for ox, oy in [(0, 0), (W - tw, 0), (0, H - th), (W - tw, H - th)]:
    all_boxes.extend(detect_region(orig_img.crop((ox, oy, ox + tw, oy + th)), ox, oy, tw / 320.0, th / 320.0))

all_boxes.sort(key=lambda b: (b[5], b[4]), reverse=True)
final_boxes = []
while all_boxes:
    best = all_boxes.pop(0)
    final_boxes.append(best)
    kept = []
    for b in all_boxes:
        ix1, iy1 = max(best[0], b[0]), max(best[1], b[1])
        ix2, iy2 = min(best[2], b[2]), min(best[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        iou = inter / ((best[2]-best[0])*(best[3]-best[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)
        containment = inter / ((b[2]-b[0])*(b[3]-b[1]) + 1e-6)
        if iou < 0.45 and not (best[5] and not b[5] and containment > 0.65):
            kept.append(b)
    all_boxes = kept

# 3. Étage 2 : Recadrage haute résolution 1 MP + Classification dans Aidge
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

out_img = orig_img.copy()
draw = ImageDraw.Draw(out_img)

print(f"\n--- Résultat Pipeline 2 Étages Aidge ({len(final_boxes)} détection) ---")
for idx, (x1, y1, x2, y2, det_score, _) in enumerate(final_boxes):
    crop = orig_img.crop((int(x1), int(y1), int(x2), int(y2))).resize((128, 128), Image.BILINEAR)
    arr = (np.array(crop, dtype=np.float32) / 255.0 - MEAN) / STD
    x_cls = np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...])

    cls_sched.forward(data=[aidge_core.Tensor(x_cls)])
    logits = np.array(cls_out.get_operator().get_output(0))[0]
    probs = np.exp(logits - np.max(logits))
    probs /= np.sum(probs)

    cls_id = int(np.argmax(probs))
    species = class_names[cls_id]
    cls_conf = float(probs[cls_id])

    print(f"#{idx} -> Détection: {det_score*100:.1f}% | Espèce: {species.upper()} ({cls_conf*100:.1f}%) | Box: ({int(x1)},{int(y1)},{int(x2)},{int(y2)})")
    draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
    draw.text((x1 + 6, max(0, y1 - 18)), f"{species} ({cls_conf*100:.1f}%)", fill="lime")

out_img.save("result_pipeline.jpg")
print("Image annotée sauvegardée : result_pipeline.jpg")
