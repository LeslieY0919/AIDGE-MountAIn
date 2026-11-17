import json
import re
import shutil
import subprocess
from pathlib import Path

from mountain import paths

DET_DIR = paths.EXPORT_DETECTOR_INT8
CLS_DIR = paths.EXPORT_CLASSIFIER_INT8
OUT_DIR = paths.EXPORT_PIPELINE


def parse_forward_hpp(hpp_path: Path):
    txt = hpp_path.read_text()
    m = re.search(r"void\s+model_forward\s*\(([^)]+)\)", txt, re.S)
    params = [p.strip() for p in m.group(1).split(",")]
    in_type = "int8_t" if "int8_t" in params[0] else "float"
    outs = []
    for p in params[1:]:
        if "int8_t" in p:
            otype = "int8_t"
        elif "int64_t" in p:
            otype = "int64_t"
        else:
            otype = "float"
        oname = p.split()[-1].lstrip("*")
        outs.append((otype, oname))
    return in_type, outs


def main():
    class_names = json.loads(paths.CLASS_NAMES.read_text())
    num_classes = len(class_names)

    # Lecture des scales de sortie du detecteur INT8 depuis son main.cpp valide
    det_main_txt = (DET_DIR / "main.cpp").read_text()
    s_boxes = float(re.search(r"S_BOXES\s*=\s*([0-9.eE+-]+)f", det_main_txt).group(1))
    s_scores = float(re.search(r"S_SCORES\s*=\s*([0-9.eE+-]+)f", det_main_txt).group(1))

    det_in_type, det_outs = parse_forward_hpp(DET_DIR / "dnn/include/forward.hpp")
    cls_in_type, cls_outs = parse_forward_hpp(CLS_DIR / "dnn/include/forward.hpp")

    shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.mkdir(parents=True)
    shutil.copytree(DET_DIR / "dnn", OUT_DIR / "detector_dnn")
    shutil.copytree(CLS_DIR / "dnn", OUT_DIR / "classifier_dnn")
    shutil.copy2(paths.STB_HEADER, OUT_DIR / "stb_image.h")

    # Signatures extern "C" sans conflit de header guards
    det_sig_args = [f"const {det_in_type}* input"] + [f"{t}** out_{i}" for i, (t, _) in enumerate(det_outs)]
    cls_sig_args = [f"const {cls_in_type}* input"] + [f"{t}** out_{i}" for i, (t, _) in enumerate(cls_outs)]

    det_call_decls = "\n".join(f"    {t}* det_out_{i} = nullptr;" for i, (t, _) in enumerate(det_outs))
    det_call_args = ", ".join(["det_in_ptr"] + [f"&det_out_{i}" for i in range(len(det_outs))])

    cls_call_decls = []
    cls_call_args = ["cls_input.data()"]
    cls_logits_var = "cls_out_0"
    for i, (t, _) in enumerate(cls_outs):
        cls_call_decls.append(f"            {t}* cls_out_{i} = nullptr;")
        cls_call_args.append(f"&cls_out_{i}")
        if t == "float":
            cls_logits_var = f"cls_out_{i}"

    species_c_array = ", ".join(f'"{c}"' for c in class_names)

    if det_in_type == "int8_t":
        det_quant_block = """    std::vector<int8_t> det_quant_in(det_input.size());
    for (size_t i = 0; i < det_input.size(); ++i) {
        int q = static_cast<int>(std::round(det_input[i] * 127.0f));
        det_quant_in[i] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
    }
    const int8_t* det_in_ptr = det_quant_in.data();"""
    else:
        det_quant_block = "    const float* det_in_ptr = det_input.data();"

    main_cpp = f"""#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

extern "C" {{
void detector_forward({", ".join(det_sig_args)});
void classifier_forward({", ".join(cls_sig_args)});
}}

struct Detection {{
    int class_id;
    float score;
    float x1, y1, x2, y2;
}};

static float compute_iou(const Detection& a, const Detection& b) {{
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float denom = area_a + area_b - inter;
    return (denom > 0.0f) ? (inter / denom) : 0.0f;
}}

static void preprocess_detector(const unsigned char* img, int orig_w, int orig_h, std::vector<float>& out_nhwc) {{
    constexpr int DST_W = 640;
    constexpr int DST_H = 640;
    out_nhwc.resize(1 * DST_H * DST_W * 3);

    const float scale_x = static_cast<float>(orig_w) / static_cast<float>(DST_W);
    const float scale_y = static_cast<float>(orig_h) / static_cast<float>(DST_H);

    for (int y = 0; y < DST_H; ++y) {{
        float fy = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        int y0 = std::max(0, std::min(orig_h - 1, static_cast<int>(std::floor(fy))));
        int y1 = std::max(0, std::min(orig_h - 1, y0 + 1));
        float wy = std::max(0.0f, std::min(1.0f, fy - static_cast<float>(y0)));

        for (int x = 0; x < DST_W; ++x) {{
            float fx = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            int x0 = std::max(0, std::min(orig_w - 1, static_cast<int>(std::floor(fx))));
            int x1 = std::max(0, std::min(orig_w - 1, x0 + 1));
            float wx = std::max(0.0f, std::min(1.0f, fx - static_cast<float>(x0)));

            for (int c = 0; c < 3; ++c) {{
                float v00 = static_cast<float>(img[(y0 * orig_w + x0) * 3 + c]);
                float v01 = static_cast<float>(img[(y0 * orig_w + x1) * 3 + c]);
                float v10 = static_cast<float>(img[(y1 * orig_w + x0) * 3 + c]);
                float v11 = static_cast<float>(img[(y1 * orig_w + x1) * 3 + c]);
                float val = (1.0f - wy) * ((1.0f - wx) * v00 + wx * v01)
                          + wy * ((1.0f - wx) * v10 + wx * v11);
                out_nhwc[(y * DST_W + x) * 3 + c] = val / 255.0f;
            }}
        }}
    }}
}}

static void preprocess_classifier_crop(const unsigned char* img, int orig_w, int orig_h,
                                       const Detection& det, std::vector<float>& out_nhwc) {{
    constexpr int CLS_W = 128;
    constexpr int CLS_H = 128;
    constexpr float MEAN[3] = {{0.485f, 0.456f, 0.406f}};
    constexpr float STD[3]  = {{0.229f, 0.224f, 0.225f}};
    out_nhwc.resize(1 * CLS_H * CLS_W * 3);

    int cx1 = std::max(0, static_cast<int>(std::floor(det.x1)));
    int cy1 = std::max(0, static_cast<int>(std::floor(det.y1)));
    int cx2 = std::min(orig_w, static_cast<int>(std::ceil(det.x2)));
    int cy2 = std::min(orig_h, static_cast<int>(std::ceil(det.y2)));
    int crop_w = std::max(1, cx2 - cx1);
    int crop_h = std::max(1, cy2 - cy1);

    const float scale_x = static_cast<float>(crop_w) / static_cast<float>(CLS_W);
    const float scale_y = static_cast<float>(crop_h) / static_cast<float>(CLS_H);

    for (int y = 0; y < CLS_H; ++y) {{
        float fy = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        int y0 = std::max(0, std::min(crop_h - 1, static_cast<int>(std::floor(fy))));
        int y1 = std::max(0, std::min(crop_h - 1, y0 + 1));
        float wy = std::max(0.0f, std::min(1.0f, fy - static_cast<float>(y0)));

        for (int x = 0; x < CLS_W; ++x) {{
            float fx = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            int x0 = std::max(0, std::min(crop_w - 1, static_cast<int>(std::floor(fx))));
            int x1 = std::max(0, std::min(crop_w - 1, x0 + 1));
            float wx = std::max(0.0f, std::min(1.0f, fx - static_cast<float>(x0)));

            for (int c = 0; c < 3; ++c) {{
                float v00 = static_cast<float>(img[((cy1 + y0) * orig_w + (cx1 + x0)) * 3 + c]);
                float v01 = static_cast<float>(img[((cy1 + y0) * orig_w + (cx1 + x1)) * 3 + c]);
                float v10 = static_cast<float>(img[((cy1 + y1) * orig_w + (cx1 + x0)) * 3 + c]);
                float v11 = static_cast<float>(img[((cy1 + y1) * orig_w + (cx1 + x1)) * 3 + c]);
                float val = (1.0f - wy) * ((1.0f - wx) * v00 + wx * v01)
                          + wy * ((1.0f - wx) * v10 + wx * v11);
                out_nhwc[(y * CLS_W + x) * 3 + c] = ((val / 255.0f) - MEAN[c]) / STD[c];
            }}
        }}
    }}
}}

int main(int argc, char** argv) {{
    if (argc < 2) {{
        std::cerr << "Usage: " << argv[0] << " <image_path> [conf_thresh]" << std::endl;
        return 1;
    }}

    const std::string img_path = argv[1];
    const float conf_thresh = (argc > 2) ? std::strtof(argv[2], nullptr) : 0.25f;
    const float iou_thresh = 0.70f;

    const char* DET_LABELS[3] = {{"animal", "person", "vehicle"}};
    const char* SPECIES_LABELS[{num_classes}] = {{{species_c_array}}};
    constexpr int NUM_ANCHORS = 8400;
    constexpr float S_BOXES = {s_boxes:.10f}f;
    constexpr float S_SCORES = {s_scores:.10f}f;

    int orig_w = 0, orig_h = 0, channels = 0;
    unsigned char* img = stbi_load(img_path.c_str(), &orig_w, &orig_h, &channels, 3);
    if (!img) {{
        std::cerr << "Erreur chargement image : " << img_path << std::endl;
        return 1;
    }}

    auto t0 = std::chrono::steady_clock::now();

    // 1. Etage Detection (YOLOv10 INT8)
    std::vector<float> det_input;
    preprocess_detector(img, orig_w, orig_h, det_input);
{det_quant_block}

{det_call_decls}
    detector_forward({det_call_args});

    std::vector<Detection> candidates;
    for (int i = 0; i < NUM_ANCHORS; ++i) {{
        int best_c = 0;
        float best_logit = static_cast<float>(det_out_1[i]) * S_SCORES;
        for (int c = 1; c < 3; ++c) {{
            float l = static_cast<float>(det_out_1[c * NUM_ANCHORS + i]) * S_SCORES;
            if (l > best_logit) {{
                best_logit = l;
                best_c = c;
            }}
        }}
        float best_score = 1.0f / (1.0f + std::exp(-best_logit));
        if (best_score < conf_thresh) continue;

        float dist[4];
        for (int side = 0; side < 4; ++side) {{
            float max_v = static_cast<float>(det_out_0[(side * 16) * NUM_ANCHORS + i]) * S_BOXES;
            for (int b = 1; b < 16; ++b) {{
                float v = static_cast<float>(det_out_0[(side * 16 + b) * NUM_ANCHORS + i]) * S_BOXES;
                if (v > max_v) max_v = v;
            }}
            float sum_e = 0.0f, acc = 0.0f;
            for (int b = 0; b < 16; ++b) {{
                float v = static_cast<float>(det_out_0[(side * 16 + b) * NUM_ANCHORS + i]) * S_BOXES;
                float e = std::exp(v - max_v);
                sum_e += e;
                acc += e * static_cast<float>(b);
            }}
            dist[side] = acc / sum_e;
        }}

        float gx = 0.0f, gy = 0.0f, stride = 8.0f;
        if (i < 6400) {{
            gx = static_cast<float>(i % 80) + 0.5f;
            gy = static_cast<float>(i / 80) + 0.5f;
            stride = 8.0f;
        }} else if (i < 8000) {{
            int idx = i - 6400;
            gx = static_cast<float>(idx % 40) + 0.5f;
            gy = static_cast<float>(idx / 40) + 0.5f;
            stride = 16.0f;
        }} else {{
            int idx = i - 8000;
            gx = static_cast<float>(idx % 20) + 0.5f;
            gy = static_cast<float>(idx / 20) + 0.5f;
            stride = 32.0f;
        }}

        float x1 = std::max(0.0f, std::min(640.0f, (gx - dist[0]) * stride)) * (static_cast<float>(orig_w) / 640.0f);
        float y1 = std::max(0.0f, std::min(640.0f, (gy - dist[1]) * stride)) * (static_cast<float>(orig_h) / 640.0f);
        float x2 = std::max(0.0f, std::min(640.0f, (gx + dist[2]) * stride)) * (static_cast<float>(orig_w) / 640.0f);
        float y2 = std::max(0.0f, std::min(640.0f, (gy + dist[3]) * stride)) * (static_cast<float>(orig_h) / 640.0f);
        candidates.push_back({{best_c, best_score, x1, y1, x2, y2}});
    }}

    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) {{ return a.score > b.score; }});

    std::vector<Detection> final_dets;
    for (const auto& cand : candidates) {{
        bool suppressed = false;
        for (const auto& kept : final_dets) {{
            if (cand.class_id == kept.class_id && compute_iou(cand, kept) > iou_thresh) {{
                suppressed = true;
                break;
            }}
        }}
        if (!suppressed) final_dets.push_back(cand);
    }}

    // 2. Etage Classification (ResNet-34 C++) sur chaque animal detecte
    std::vector<float> cls_input;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Image: " << img_path << " (" << orig_w << "x" << orig_h << ") -> "
              << final_dets.size() << " detection(s):" << std::endl;

    for (size_t k = 0; k < final_dets.size(); ++k) {{
        const auto& d = final_dets[k];
        std::cout << "  #" << k << " [" << DET_LABELS[d.class_id] << " " << (d.score * 100.0f) << "%]"
                  << " bbox=[" << static_cast<int>(d.x1) << ", " << static_cast<int>(d.y1)
                  << ", " << static_cast<int>(d.x2) << ", " << static_cast<int>(d.y2) << "]";

        if (d.class_id == 0) {{
            preprocess_classifier_crop(img, orig_w, orig_h, d, cls_input);
{chr(10).join(cls_call_decls)}
            classifier_forward({", ".join(cls_call_args)});

            float max_l = {cls_logits_var}[0];
            for (int c = 1; c < {num_classes}; ++c) {{
                if ({cls_logits_var}[c] > max_l) max_l = {cls_logits_var}[c];
            }}
            float sum_exp = 0.0f;
            float probs[{num_classes}];
            for (int c = 0; c < {num_classes}; ++c) {{
                probs[c] = std::exp({cls_logits_var}[c] - max_l);
                sum_exp += probs[c];
            }}
            int best_sp = 0;
            for (int c = 0; c < {num_classes}; ++c) {{
                probs[c] /= sum_exp;
                if (probs[c] > probs[best_sp]) best_sp = c;
            }}
            std::cout << " -> ESPECE: " << SPECIES_LABELS[best_sp]
                      << " (" << (probs[best_sp] * 100.0f) << "%)";
        }}
        std::cout << std::endl;
    }}

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Temps total pipeline C++ (preprocess + YOLOv10 + ResNet34) : " << ms << " ms" << std::endl;

    stbi_image_free(img);
    return 0;
}}
"""
    (OUT_DIR / "main.cpp").write_text(main_cpp)

    (OUT_DIR / "detector.map").write_text("{\n  global:\n    detector_forward;\n  local:\n    *;\n};\n")
    (OUT_DIR / "classifier.map").write_text("{\n  global:\n    classifier_forward;\n  local:\n    *;\n};\n")

    cmake_txt = """cmake_minimum_required(VERSION 3.16)
project(wildlife_pipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH "$ORIGIN")

file(GLOB_RECURSE DETECTOR_SOURCES "detector_dnn/src/*.cpp")
add_library(detector_dnn SHARED ${DETECTOR_SOURCES})
target_include_directories(detector_dnn PRIVATE detector_dnn detector_dnn/include)
target_compile_definitions(detector_dnn PRIVATE model_forward=detector_forward)
target_compile_options(detector_dnn PRIVATE -O3 -ffast-math -march=native)
target_link_options(detector_dnn PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/detector.map")

file(GLOB_RECURSE CLASSIFIER_SOURCES "classifier_dnn/src/*.cpp")
add_library(classifier_dnn SHARED ${CLASSIFIER_SOURCES})
target_include_directories(classifier_dnn PRIVATE classifier_dnn classifier_dnn/include)
target_compile_definitions(classifier_dnn PRIVATE model_forward=classifier_forward)
target_compile_options(classifier_dnn PRIVATE -O3 -ffast-math -march=native)
target_link_options(classifier_dnn PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/classifier.map")

add_executable(wildlife_pipeline main.cpp)
target_compile_options(wildlife_pipeline PRIVATE -O3 -ffast-math -march=native)
target_link_libraries(wildlife_pipeline PRIVATE detector_dnn classifier_dnn)
"""
    (OUT_DIR / "CMakeLists.txt").write_text(cmake_txt)

    subprocess.run(["cmake", "-B", f"{OUT_DIR}/build", "-S", str(OUT_DIR), "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", f"{OUT_DIR}/build", "-j2"], check=True)
    print(f"\nExecutable autonome pret : {OUT_DIR}/build/wildlife_pipeline")


if __name__ == "__main__":
    main()
