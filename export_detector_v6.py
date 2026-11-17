import faulthandler
faulthandler.enable()

import subprocess
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np
import aidge_core
import aidge_onnx
import aidge_backend_cpu
import aidge_export_cpp

RAW_MODEL = "detector_v6_trained_640.onnx"
CUT_MODEL = "MDV6-yolov10-c_cut.onnx"
FOLDED_CUT_MODEL = "MDV6-yolov10-c_cut_folded.onnx"
EXPORT_DIR = "export_cpp_detector"

# 1. Extraction des 2 branches de sortie avant le post-traitement DFL/Sigmoid
onnx.utils.extract_model(
    RAW_MODEL,
    CUT_MODEL,
    input_names=["images"],
    output_names=["/model.23/Concat_output_0", "/model.23/Concat_1_output_0"],
)

# 2. Pliage des constantes statiques via ONNX Runtime
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
opts.optimized_model_filepath = FOLDED_CUT_MODEL
ort.InferenceSession(CUT_MODEL, opts, providers=["CPUExecutionProvider"])

# 3. Chargement Aidge et promotion des scalaires 0D
graph = aidge_onnx.load_onnx(FOLDED_CUT_MODEL)
aidge_core.remove_flatten(graph)
graph.set_datatype(aidge_core.dtype.float32)
graph.set_backend("cpu")

in_shape = [1, 3, 640, 640]
dummy_input = aidge_core.Tensor(np.zeros(in_shape, dtype=np.float32))
dummy_input.set_datatype(aidge_core.dtype.float32)
dummy_input.set_data_format(aidge_core.dformat.nchw)

graph.forward_dims([in_shape], allow_data_dependency=True)

for node in graph.get_nodes():
    if node.type() == "Producer":
        t = node.get_operator().get_output(0)
        if t is not None and len(t.dims) == 0:
            target_rank = max(
                (
                    len(child.get_operator().get_input(i).dims)
                    for child in node.get_children()
                    for i in range(child.get_nb_inputs())
                    if child.get_operator().get_input(i) is not None
                ),
                default=0,
            )
            if target_rank > 0:
                new_t = aidge_core.Tensor(np.array(t).reshape((1,) * target_rank))
                new_t.set_datatype(t.dtype)
                new_t.set_data_format(
                    aidge_core.dformat.nchw if target_rank == 4 else aidge_core.dformat.default
                )
                node.get_operator().set_output(0, new_t)

graph.forward_dims([in_shape], allow_data_dependency=True)

# 4. Conserver uniquement les 2 sorties Concat
clean_outputs = [
    (node, out_idx)
    for node, out_idx in graph.get_ordered_outputs()
    if out_idx == 0 and "Concat" in node.name()
]
if len(clean_outputs) == 2:
    graph.set_ordered_outputs(clean_outputs)

# 5. Export C++ (sans compilation parallèle interne pour préserver la RAM)
config = aidge_export_cpp.ExportConfig(
    export_folder=EXPORT_DIR,
    dtype="float32",
    cuda=False,
    compile=False,
)

aidge_core.export_utils.export(
    graph=graph,
    config=config,
    input_tensors=[dummy_input],
)

# 6. Écrire main.cpp (entrée NHWC + 6 arguments forward.hpp) et compiler en Release -O3 -j1
main_cpp = '#include <algorithm>\n#include <cmath>\n#include <cstdint>\n#include <cstdlib>\n#include <fstream>\n#include <iomanip>\n#include <iostream>\n#include <string>\n#include <vector>\n#include "forward.hpp"\n\n#define STB_IMAGE_IMPLEMENTATION\n#include "stb_image.h"\n\nstruct Detection {\n    int class_id;\n    float score;\n    float x1, y1, x2, y2;\n};\n\nstatic float compute_iou(const Detection& a, const Detection& b)\n{\n    float xx1 = std::max(a.x1, b.x1);\n    float yy1 = std::max(a.y1, b.y1);\n    float xx2 = std::min(a.x2, b.x2);\n    float yy2 = std::min(a.y2, b.y2);\n    float w = std::max(0.0f, xx2 - xx1);\n    float h = std::max(0.0f, yy2 - yy1);\n    float inter = w * h;\n    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);\n    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);\n    float denom = area_a + area_b - inter;\n    return (denom > 0.0f) ? (inter / denom) : 0.0f;\n}\n\nstatic bool load_and_preprocess(const std::string& path, std::vector<float>& out_nhwc, int& orig_w, int& orig_h)\n{\n    constexpr int DST_W = 640;\n    constexpr int DST_H = 640;\n    out_nhwc.resize(1 * DST_H * DST_W * 3);\n\n    if (path.size() >= 4 && path.substr(path.size() - 4) == ".bin") {\n        std::ifstream in_file(path, std::ios::binary);\n        if (!in_file) return false;\n        in_file.read(reinterpret_cast<char*>(out_nhwc.data()), out_nhwc.size() * sizeof(float));\n        orig_w = DST_W;\n        orig_h = DST_H;\n        return static_cast<bool>(in_file);\n    }\n\n    int channels = 0;\n    unsigned char* img = stbi_load(path.c_str(), &orig_w, &orig_h, &channels, 3);\n    if (!img) return false;\n\n    const float scale_x = static_cast<float>(orig_w) / static_cast<float>(DST_W);\n    const float scale_y = static_cast<float>(orig_h) / static_cast<float>(DST_H);\n\n    for (int y = 0; y < DST_H; ++y) {\n        float fy = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;\n        int y0 = std::max(0, std::min(orig_h - 1, static_cast<int>(std::floor(fy))));\n        int y1 = std::max(0, std::min(orig_h - 1, y0 + 1));\n        float wy = std::max(0.0f, std::min(1.0f, fy - static_cast<float>(y0)));\n\n        for (int x = 0; x < DST_W; ++x) {\n            float fx = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;\n            int x0 = std::max(0, std::min(orig_w - 1, static_cast<int>(std::floor(fx))));\n            int x1 = std::max(0, std::min(orig_w - 1, x0 + 1));\n            float wx = std::max(0.0f, std::min(1.0f, fx - static_cast<float>(x0)));\n\n            for (int c = 0; c < 3; ++c) {\n                float v00 = static_cast<float>(img[(y0 * orig_w + x0) * 3 + c]);\n                float v01 = static_cast<float>(img[(y0 * orig_w + x1) * 3 + c]);\n                float v10 = static_cast<float>(img[(y1 * orig_w + x0) * 3 + c]);\n                float v11 = static_cast<float>(img[(y1 * orig_w + x1) * 3 + c]);\n                float val = (1.0f - wy) * ((1.0f - wx) * v00 + wx * v01)\n                          + wy * ((1.0f - wx) * v10 + wx * v11);\n                out_nhwc[(y * DST_W + x) * 3 + c] = val / 255.0f;\n            }\n        }\n    }\n\n    stbi_image_free(img);\n    return true;\n}\n\nint main(int argc, char** argv)\n{\n    const std::string input_path = (argc > 1) ? argv[1] : "test_wildlife_1mp.jpg";\n    const float conf_thresh = (argc > 2) ? std::strtof(argv[2], nullptr) : 0.25f;\n    const float iou_thresh = 0.70f;\n    const char* LABELS[3] = {"animal", "person", "vehicle"};\n    constexpr int NUM_ANCHORS = 8400;\n\n    std::vector<float> input_data;\n    int orig_w = 640, orig_h = 640;\n    if (!load_and_preprocess(input_path, input_data, orig_w, orig_h)) {\n        std::cerr << "Erreur : impossible de charger " << input_path << std::endl;\n        return 1;\n    }\n\n    float* boxes_ptr = nullptr;\n    float* scores_ptr = nullptr;\n    int64_t* idx0 = nullptr;\n    int64_t* idx1 = nullptr;\n    int64_t* idx2 = nullptr;\n\n    model_forward(input_data.data(), &boxes_ptr, &scores_ptr, &idx0, &idx1, &idx2);\n\n    std::vector<Detection> candidates;\n    for (int i = 0; i < NUM_ANCHORS; ++i) {\n        int best_c = 0;\n        float best_logit = scores_ptr[i];\n        for (int c = 1; c < 3; ++c) {\n            float l = scores_ptr[c * NUM_ANCHORS + i];\n            if (l > best_logit) {\n                best_logit = l;\n                best_c = c;\n            }\n        }\n        float best_score = 1.0f / (1.0f + std::exp(-best_logit));\n        if (best_score < conf_thresh) continue;\n\n        float dist[4];\n        for (int side = 0; side < 4; ++side) {\n            float max_v = boxes_ptr[(side * 16) * NUM_ANCHORS + i];\n            for (int b = 1; b < 16; ++b) {\n                float v = boxes_ptr[(side * 16 + b) * NUM_ANCHORS + i];\n                if (v > max_v) max_v = v;\n            }\n            float sum_e = 0.0f;\n            float acc = 0.0f;\n            for (int b = 0; b < 16; ++b) {\n                float e = std::exp(boxes_ptr[(side * 16 + b) * NUM_ANCHORS + i] - max_v);\n                sum_e += e;\n                acc += e * static_cast<float>(b);\n            }\n            dist[side] = acc / sum_e;\n        }\n\n        float gx = 0.0f, gy = 0.0f, stride = 8.0f;\n        if (i < 6400) {\n            gx = static_cast<float>(i % 80) + 0.5f;\n            gy = static_cast<float>(i / 80) + 0.5f;\n            stride = 8.0f;\n        } else if (i < 8000) {\n            int idx = i - 6400;\n            gx = static_cast<float>(idx % 40) + 0.5f;\n            gy = static_cast<float>(idx / 40) + 0.5f;\n            stride = 16.0f;\n        } else {\n            int idx = i - 8000;\n            gx = static_cast<float>(idx % 20) + 0.5f;\n            gy = static_cast<float>(idx / 20) + 0.5f;\n            stride = 32.0f;\n        }\n\n        float x1 = std::max(0.0f, std::min(640.0f, (gx - dist[0]) * stride)) * (static_cast<float>(orig_w) / 640.0f);\n        float y1 = std::max(0.0f, std::min(640.0f, (gy - dist[1]) * stride)) * (static_cast<float>(orig_h) / 640.0f);\n        float x2 = std::max(0.0f, std::min(640.0f, (gx + dist[2]) * stride)) * (static_cast<float>(orig_w) / 640.0f);\n        float y2 = std::max(0.0f, std::min(640.0f, (gy + dist[3]) * stride)) * (static_cast<float>(orig_h) / 640.0f);\n\n        candidates.push_back({best_c, best_score, x1, y1, x2, y2});\n    }\n\n    std::sort(candidates.begin(), candidates.end(),\n              [](const Detection& a, const Detection& b) { return a.score > b.score; });\n\n    std::vector<Detection> final_dets;\n    for (const auto& cand : candidates) {\n        bool suppressed = false;\n        for (const auto& kept : final_dets) {\n            if (cand.class_id == kept.class_id && compute_iou(cand, kept) > iou_thresh) {\n                suppressed = true;\n                break;\n            }\n        }\n        if (!suppressed) final_dets.push_back(cand);\n    }\n\n    std::cout << std::fixed << std::setprecision(4);\n    std::cout << "Image: " << input_path << " (" << orig_w << "x" << orig_h << ") -> Detections (" << final_dets.size() << "):" << std::endl;\n    for (const auto& d : final_dets) {\n        std::cout << "  [" << LABELS[d.class_id] << "] conf=" << d.score\n                  << " bbox=[" << d.x1 << ", " << d.y1 << ", " << d.x2 << ", " << d.y2 << "]"\n                  << std::endl;\n    }\n    return 0;\n}\n'
(Path(EXPORT_DIR) / "main.cpp").write_text(main_cpp)
subprocess.run(["cmake", "-B", f"{EXPORT_DIR}/build", "-S", EXPORT_DIR, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_FLAGS=-O3"], check=True)
subprocess.run(["cmake", "--build", f"{EXPORT_DIR}/build", "-j1"], check=True)
print("Export et compilation Release terminés.")
