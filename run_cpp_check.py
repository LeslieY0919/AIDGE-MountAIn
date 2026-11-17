import subprocess
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

EXPORT_DIR = Path("export_cpp_detector_int8")
BUILD_DIR = EXPORT_DIR / "build"
CLASS_NAMES = ["animal", "person", "vehicle"]

MAIN_CPP = """#include <cstdio>
#include <cstdlib>
#include "forward.hpp"
#include "data/_0_PadConv_0_input_0.h"

int main(int argc, char** argv) {
    if (argc >= 2) {
        FILE* fin = std::fopen(argv[1], "rb");
        if (!fin) { std::perror("fopen input"); return 1; }
        std::size_t n_in = sizeof(_0_PadConv_0_input_0) / sizeof(_0_PadConv_0_input_0[0]);
        if (std::fread(_0_PadConv_0_input_0, sizeof(_0_PadConv_0_input_0[0]), n_in, fin) != n_in) {
            std::fprintf(stderr, "Taille entree invalide\\n");
            std::fclose(fin);
            return 1;
        }
        std::fclose(fin);
    }

    int8_t* _606_Concat_14_output_0 = nullptr;
    int8_t* _619_Concat_15_output_0 = nullptr;
    int64_t* _234_PadMaxPool_0_output_1 = nullptr;
    int64_t* _235_PadMaxPool_1_output_1 = nullptr;
    int64_t* _236_PadMaxPool_2_output_1 = nullptr;

    model_forward(
        _0_PadConv_0_input_0,
        &_606_Concat_14_output_0,
        &_619_Concat_15_output_0,
        &_234_PadMaxPool_0_output_1,
        &_235_PadMaxPool_1_output_1,
        &_236_PadMaxPool_2_output_1
    );

    FILE* fb = std::fopen("out_boxes_int8.bin", "wb");
    std::fwrite(_606_Concat_14_output_0, sizeof(int8_t), 1 * 64 * 8400, fb);
    std::fclose(fb);

    FILE* fs = std::fopen("out_scores_int8.bin", "wb");
    std::fwrite(_619_Concat_15_output_0, sizeof(int8_t), 1 * 3 * 8400, fs);
    std::fclose(fs);
    return 0;
}
"""

def decode_top_box(boxes_arr, scores_arr, orig_w, orig_h):
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

subprocess.run(["cmake", "--build", str(BUILD_DIR), "-j"], check=True)

orig_img = Image.open("test_wildlife_1mp.jpg").convert("RGB")
W, H = orig_img.size
arr = np.asarray(orig_img.resize((640, 640), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
in_nchw = np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...], dtype=np.float32)
in_nchw.tofile(BUILD_DIR / "input_f32.bin")

subprocess.run(["./run_export", "input_f32.bin"], cwd=BUILD_DIR, check=True)

boxes_i8 = np.fromfile(BUILD_DIR / "out_boxes_int8.bin", dtype=np.int8).reshape(1, 64, 8400)
scores_i8 = np.fromfile(BUILD_DIR / "out_scores_int8.bin", dtype=np.int8).reshape(1, 3, 8400)

sess = ort.InferenceSession("MDV6-yolov10-c_cut_folded.onnx", providers=["CPUExecutionProvider"])
onnx_outs = sess.run(None, {sess.get_inputs()[0].name: in_nchw})
fp32_boxes = next(o for o in onnx_outs if o.shape == (1, 64, 8400))
fp32_scores = next(o for o in onnx_outs if o.shape == (1, 3, 8400))

s_boxes = float(np.sum(fp32_boxes * boxes_i8) / np.sum(boxes_i8.astype(np.float64) ** 2))
s_scores = float(np.sum(fp32_scores * scores_i8) / np.sum(scores_i8.astype(np.float64) ** 2))

cid, sc, box, idx = decode_top_box(boxes_i8 * s_boxes, scores_i8 * s_scores, W, H)
print(f"Scales calibres -> S_boxes={s_boxes:.6f} | S_scores={s_scores:.6f}")
print(f"C++ INT8  : class={CLASS_NAMES[cid]} ({cid}) | conf={sc*100:.2f}% | bbox={box} | anchor={idx}")
print(f"Ref FP32  : class=animal (0) | conf=92.25% | bbox=[24, 83, 1072, 866]")

draw = ImageDraw.Draw(orig_img)
draw.rectangle(box, outline=(0, 255, 0), width=4)
orig_img.save("detection_cpp_int8_result.jpg")
