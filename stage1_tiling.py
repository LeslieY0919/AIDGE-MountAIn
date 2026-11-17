import numpy as np
from PIL import Image, ImageDraw
import aidge_core
import aidge_backend_cpu
import aidge_onnx

orig_img = Image.open("test_wildlife_1mp.jpg").convert("RGB")
W, H = orig_img.size

graph = aidge_onnx.load_onnx("MDV6-yolov10-c.onnx")
for node in graph.get_nodes():
    if node.type() == "Producer":
        op = node.get_operator()
        if hasattr(op, "set_constant"):
            op.set_constant(True)
        elif hasattr(op, "attr") and hasattr(op.attr, "constant"):
            op.attr.constant = True

aidge_core.constant_folding(graph)
graph.forward_dims([[1, 3, 320, 320]])
graph.set_datatype(aidge_core.dtype.float32)
graph.set_backend("cpu")
scheduler = aidge_core.SequentialScheduler(graph)
out_node = list(graph.get_output_nodes())[0]

def run_detector(crop_img, ox, oy, scale_x, scale_y, is_global=False):
    resized = crop_img.resize((320, 320), Image.BILINEAR)
    x = np.transpose(np.array(resized, dtype=np.float32) / 255.0, (2, 0, 1))[None, ...]
    scheduler.forward(data=[aidge_core.Tensor(x)])
    preds = np.array(out_node.get_operator().get_output(0))[0].T
    boxes = []
    for row in preds[preds[:, 4] > 0.35]:
        cx, cy, bw, bh, score = row[:5]
        x1 = max(0, ox + (cx - bw / 2) * scale_x)
        y1 = max(0, oy + (cy - bh / 2) * scale_y)
        x2 = min(W, ox + (cx + bw / 2) * scale_x)
        y2 = min(H, oy + (cy + bh / 2) * scale_y)
        boxes.append([x1, y1, x2, y2, float(score), is_global])
    return boxes

# 1. Passe globale (grands animaux) + 4 tuiles (petits animaux en 1 MP)
all_boxes = run_detector(orig_img, 0, 0, W / 320.0, H / 320.0, is_global=True)

overlap = 0.15
tile_w, tile_h = int(W * (0.5 + overlap / 2)), int(H * (0.5 + overlap / 2))
for ox, oy in [(0, 0), (W - tile_w, 0), (0, H - tile_h), (W - tile_w, H - tile_h)]:
    tile = orig_img.crop((ox, oy, ox + tile_w, oy + tile_h))
    all_boxes.extend(run_detector(tile, ox, oy, tile_w / 320.0, tile_h / 320.0, is_global=False))

# 2. NMS + suppression des sous-fragments inclus dans une détection globale
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
        area_a = (best[2] - best[0]) * (best[3] - best[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        iou = inter / (area_a + area_b - inter + 1e-6)
        containment = inter / (area_b + 1e-6)
        if iou < 0.45 and not (best[5] and not b[5] and containment > 0.65):
            kept.append(b)
    all_boxes = kept

print(f"Animaux détectés après fusion : {len(final_boxes)}")
debug_img = orig_img.copy()
draw = ImageDraw.Draw(debug_img)
for idx, (x1, y1, x2, y2, score, _) in enumerate(final_boxes):
    print(f" -> Détection #{idx} : box=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}) | confiance={score:.2f}")
    orig_img.crop((int(x1), int(y1), int(x2), int(y2))).save(f"crop_animal_{idx}.jpg")
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

debug_img.save("debug_detections.jpg")
