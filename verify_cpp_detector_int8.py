import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

BUILD_DIR = Path("export_cpp_detector_int8/build")
CLASS_NAMES = ["animal", "person", "vehicle"]


def decode_top_box(boxes_arr, scores_arr, orig_w, orig_h):
    # boxes_arr: (1, 64, 8400), scores_arr: (1, 3, 8400)
    boxes = boxes_arr[0]
    scores = 1.0 / (1.0 + np.exp(-np.clip(scores_arr[0], -60.0, 60.0)))
    best_cls = np.argmax(scores, axis=0)
    best_sc = scores[best_cls, np.arange(8400)]
    i = int(np.argmax(best_sc))

    dist = []
    for side in range(4):
        logits = boxes[side * 16 : (side + 1) * 16, i]
        e = np.exp(logits - np.max(logits))
        dist.append(float(np.sum((e / np.sum(e)) * np.arange(16))))

    if i < 6400:
        gx, gy, stride = (i % 80) + 0.5, (i // 80) + 0.5, 8.0
    elif i < 8000:
        idx = i - 6400
        gx, gy, stride = (idx % 40) + 0.5, (idx // 40) + 0.5, 16.0
    else:
        idx = i - 8000
        gx, gy, stride = (idx % 20) + 0.5, (idx // 20) + 0.5, 32.0

    x1 = int(round(np.clip((gx - dist[0]) * stride, 0, 640) * orig_w / 640.0))
    y1 = int(round(np.clip((gy - dist[1]) * stride, 0, 640) * orig_h / 640.0))
    x2 = int(round(np.clip((gx + dist[2]) * stride, 0, 640) * orig_w / 640.0))
    y2 = int(round(np.clip((gy + dist[3]) * stride, 0, 640) * orig_h / 640.0))
    return int(best_cls[i]), float(best_sc[i]), [x1, y1, x2, y2], i


def main():
    img_path = Path("test_wildlife_1mp.jpg")
    orig_img = Image.open(img_path).convert("RGB")
    W, H = orig_img.size

    # Meme pretraitement exact que quantize_pipeline_int8.py (calib_det[0])
    resized = orig_img.resize((640, 640), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    in_nchw = np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...], dtype=np.float32)

    in_bin = BUILD_DIR / "input_f32.bin"
    in_nchw.tofile(in_bin)

    subprocess.run(["./run_export", "input_f32.bin"], cwd=BUILD_DIR, check=True)

    boxes_i8 = np.fromfile(BUILD_DIR / "out_boxes_int8.bin", dtype=np.int8).reshape(1, 64, 8400)
    scores_i8 = np.fromfile(BUILD_DIR / "out_scores_int8.bin", dtype=np.int8).reshape(1, 3, 8400)

    # Lecture brute (scale=1.0) avant injection des scales exacts de forward.cpp
    cid, sc, box, anchor_idx = decode_top_box(boxes_i8.astype(np.float32), scores_i8.astype(np.float32), W, H)
    print(f"Sortie C++ brute (scale=1.0) : class={CLASS_NAMES[cid]} ({cid}) | anchor={anchor_idx} | bbox={box}")
    print(f"  Logits scores_i8 sur l'ancre gagnante : {scores_i8[0, :, anchor_idx].tolist()}")
    print(f"  Min/Max boxes_i8 : [{boxes_i8.min()}, {boxes_i8.max()}] | Min/Max scores_i8 : [{scores_i8.min()}, {scores_i8.max()}]")

    draw = ImageDraw.Draw(orig_img)
    draw.rectangle(box, outline=(0, 255, 0), width=4)
    orig_img.save("detection_cpp_int8_result.jpg")


if __name__ == "__main__":
    main()
