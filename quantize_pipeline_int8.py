import os
import re
import json
import shutil
import inspect
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

import aidge_core
import aidge_backend_cpu
import aidge_onnx
import aidge_quantization
import aidge_export_cpp
from aidge_core.export_utils import ExportNodeCpp, get_node_from_metaop
import importlib
export_mod = importlib.import_module("aidge_core.export_utils.export")
from aidge_export_cpp import CPP_ROOT, ExportLibCpp

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLS_NORM_DIV = 3.0


def apply_aidge_export_fixes():
    """Enregistre les correctifs Dequantizer (int32) et Quantizer (float32)."""
    @ExportLibCpp.register_metaop(
        "Dequantizer",
        "Cast->Sub#->Mul#; Sub#<1-Producer; Mul#<1-Producer",
        aidge_core.ImplSpec(
            [aidge_core.IOSpec(aidge_core.dtype.any, aidge_core.dformat.any)],
            [aidge_core.IOSpec(aidge_core.dtype.float32, aidge_core.dformat.any)],
        ),
    )
    class FixedDequantizer(ExportNodeCpp):
        def __init__(self, node, mem_info, instance_count):
            super().__init__(node, mem_info, instance_count)
            self.attributes["activation"] = "Linear"
            self.attributes["rescaling"] = "FloatingPointScaling"
            mul_node = get_node_from_metaop(node, "Mul")
            self.attributes["coef_value"] = float(
                np.array(mul_node[0].get_operator().get_input(1)).reshape(-1)[0]
            )
            self.attributes["accumulation_type"] = "float"
            self.config_template = str(
                CPP_ROOT / "templates" / "configuration" / "activation_config.jinja"
            )
            self.forward_template = str(
                CPP_ROOT / "templates" / "kernel_forward" / "activation_forward.jinja"
            )
            self.include_list = []
            self.add_kernel_to_copy(
                CPP_ROOT / "kernels" / "activation.hpp", "include/kernels/cpp"
            )

    from aidge_export_cpp.operators import Quantizer as QMod

    orig_q_init = QMod.Quantizer.__init__

    def patched_q_init(self, node, mem_info, instance_count):
        orig_q_init(self, node, mem_info, instance_count)
        in_t = node.get_operator().get_input(0)
        if in_t is not None and in_t.dtype in (
            aidge_core.dtype.float32,
            aidge_core.dtype.float64,
        ):
            mul_nodes = get_node_from_metaop(node, "Mul")
            if mul_nodes:
                scale_val = float(
                    np.array(mul_nodes[0].get_operator().get_input(1)).reshape(-1)[0]
                )
                self.attributes["coef_value"] = scale_val
                self.attributes["rescaling"] = "FloatingPointScaling"
                self.attributes["accumulation_type"] = "float"
                self.attributes["shift_value"] = 0

    QMod.Quantizer.__init__ = patched_q_init


def patch_exported_headers(export_dir):
    """Applique les correctifs C++ sur activation_utils.hpp et transpose.hpp."""
    patch_cache = Path(".aidge_patches")
    patch_cache.mkdir(exist_ok=True)
    ref_root = Path("export_tmp_cmp_valid/dnn/include")
    dst_root = Path(export_dir) / "dnn" / "include"

    for rel in [
        Path("utils/cpp/activation_utils.hpp"),
        Path("utils/cpp/rescaling_utils.hpp"),
    ]:
        if (ref_root / rel).exists():
            shutil.copy2(ref_root / rel, patch_cache / rel.name)
        if (patch_cache / rel.name).exists() and (dst_root / rel).exists():
            shutil.copy2(patch_cache / rel.name, dst_root / rel)

    t_path = dst_root / "kernels/cpp/transpose.hpp"
    if t_path.exists():
        txt = t_path.read_text()
        old = "        outputs[output_index] = inputs[inOffset + idx];"
        if old in txt:
            new = """        const size_t last_dim = IN_DIMS[NB_DIMS - 1];
        const size_t in_stride_elts = (INPUT_MEM_STRIDE > 0) ? (INPUT_MEM_STRIDE / sizeof(T)) : last_dim;
        const size_t mapped_idx = (NB_DIMS >= 2 && last_dim > 0)
            ? ((idx / last_dim) * in_stride_elts + (idx % last_dim))
            : idx;
        outputs[output_index] = inputs[inOffset + mapped_idx];"""
            t_path.write_text(txt.replace(old, new, 1))


def make_tensor(arr_nchw):
    t = aidge_core.Tensor(np.ascontiguousarray(arr_nchw, dtype=np.float32))
    t.set_datatype(aidge_core.dtype.float32)
    t.set_data_format(aidge_core.dformat.nchw)
    t.set_backend("cpu")
    return t


def build_classifier_graph(onnx_path="species_classifier_folded.onnx"):
    graph = aidge_onnx.load_onnx(onnx_path)
    for node in graph.get_nodes():
        if node.type() == "Producer":
            op = node.get_operator()
            if hasattr(op, "set_constant"):
                op.set_constant(True)
            elif hasattr(op, "attr") and hasattr(op.attr, "constant"):
                op.attr.constant = True
        elif node.type() == "Clip":
            op = node.get_operator()
            for idx, attr_name in ((1, "min"), (2, "max")):
                if idx < node.get_nb_inputs() and node.get_parent(idx) is not None:
                    parent = node.get_parent(idx)
                    if parent.type() == "Producer":
                        t = parent.get_operator().get_output(0)
                        if t is not None and t.size > 0:
                            setattr(op.attr, attr_name, float(np.array(t).reshape(-1)[0]))
    aidge_core.constant_folding(graph)
    aidge_core.remove_flatten(graph)
    graph.set_datatype(aidge_core.dtype.float32)
    graph.set_backend("cpu")
    graph.forward_dims([[1, 3, 128, 128]])
    return graph


def build_detector_graph(onnx_path="MDV6-yolov10-c_cut_folded.onnx"):
    graph = aidge_onnx.load_onnx(onnx_path)
    aidge_core.remove_flatten(graph)
    graph.set_datatype(aidge_core.dtype.float32)
    graph.set_backend("cpu")
    in_shape = [1, 3, 640, 640]
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
    clean_outputs = [
        (node, out_idx)
        for node, out_idx in graph.get_ordered_outputs()
        if out_idx == 0 and "Concat" in node.name()
    ]
    if len(clean_outputs) == 2:
        graph.set_ordered_outputs(clean_outputs)
    return graph


def compute_output_scales(fp32_builder, calib_tensors):
    """Calcule les facteurs d'echelle exacts de dequantification des sorties INT8."""
    nb_calib = getattr(export_mod, "NB_CALIB", len(calib_tensors))
    calib_slice = [t.clone() for t in calib_tensors[:nb_calib]]

    g_fp32 = fp32_builder()
    s_fp32 = aidge_core.SequentialScheduler(g_fp32)
    s_fp32.forward(data=[calib_slice[0]])
    fp32_outs = [
        np.array(node.get_operator().get_output(idx), dtype=np.float64)
        for node, idx in g_fp32.get_ordered_outputs()
    ]

    g_q = fp32_builder()
    g_q = export_mod.quantize_graph(g_q, [t.clone() for t in calib_slice])
    g_q.set_backend("cpu")
    q_in = export_mod.quantize_inputs([calib_slice[0].clone()], "int8")[0]
    s_q = aidge_core.SequentialScheduler(g_q)
    s_q.forward(data=[q_in])
    q_outs = [
        np.array(node.get_operator().get_output(idx), dtype=np.float64)
        for node, idx in g_q.get_ordered_outputs()
    ]

    scales = []
    for f_arr, q_arr in zip(fp32_outs, q_outs):
        denom = float(np.sum(q_arr ** 2))
        s = float(np.sum(f_arr * q_arr) / denom) if denom > 0.0 else 1.0
        scales.append(s)
    return scales


def export_int8_cpp(fp32_builder, calib_tensors, out_dir):
    out_scales = compute_output_scales(fp32_builder, calib_tensors)
    shutil.rmtree(out_dir, ignore_errors=True)
    sig = inspect.signature(aidge_export_cpp.ExportConfig)
    kwargs = {"export_folder": out_dir, "dtype": "int8", "cuda": False, "compile": False}
    if "use_zero_point" in sig.parameters:
        kwargs["use_zero_point"] = False
    if "channel_wise" in sig.parameters:
        kwargs["channel_wise"] = False

    cfg = aidge_export_cpp.ExportConfig(**kwargs)
    g_fresh = fp32_builder()
    export_inputs = [t.clone() for t in calib_tensors]
    aidge_core.export_utils.export(graph=g_fresh, config=cfg, input_tensors=export_inputs)
    patch_exported_headers(out_dir)
    return out_scales


def parse_forward_signature(forward_hpp_path):
    txt = Path(forward_hpp_path).read_text()
    m = re.search(r"void\s+model_forward\s*\(([^)]+)\)", txt, re.S)
    params = [p.strip() for p in m.group(1).split(",")]
    in_type = "int8_t" if "int8_t" in params[0] else "float"
    outs = []
    for p in params[1:]:
        otype = "int8_t" if "int8_t" in p else ("int64_t" if "int64_t" in p else "float")
        oname = p.split()[-1].lstrip("*")
        outs.append((otype, oname))
    return in_type, outs


def write_and_build_classifier(out_dir, out_scale):
    in_type, outs = parse_forward_signature(Path(out_dir) / "dnn/include/forward.hpp")
    out_type, out_name = outs[0]

    if in_type == "int8_t":
        in_prep = f"""    std::vector<int8_t> model_in(INPUT_ELEMENTS);
    for (size_t i = 0; i < INPUT_ELEMENTS; ++i) {{
        int q = static_cast<int>(std::round((input_data[i] / {CLS_NORM_DIV:.6f}f) * 127.0f));
        model_in[i] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
    }}"""
        in_arg = "model_in.data()"
    else:
        in_prep = ""
        in_arg = "input_data.data()"

    if out_type == "int8_t":
        out_dequant = f"""    float logits[OUTPUT_ELEMENTS];
    for (size_t i = 0; i < OUTPUT_ELEMENTS; ++i) {{
        logits[i] = static_cast<float>({out_name}[i]) * {out_scale:.10f}f;
    }}"""
    else:
        out_dequant = f"""    float logits[OUTPUT_ELEMENTS];
    for (size_t i = 0; i < OUTPUT_ELEMENTS; ++i) {{
        logits[i] = {out_name}[i];
    }}"""

    main_cpp = f"""#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include "forward.hpp"

int main()
{{
    constexpr size_t INPUT_ELEMENTS  = 1 * 128 * 128 * 3;
    constexpr size_t OUTPUT_ELEMENTS = 5;

    std::vector<float> input_data(INPUT_ELEMENTS);
    std::ifstream in_file("cls_input.bin", std::ios::binary);
    if (!in_file) {{
        std::cerr << "Erreur : impossible d'ouvrir cls_input.bin" << std::endl;
        return 1;
    }}
    in_file.read(reinterpret_cast<char*>(input_data.data()), INPUT_ELEMENTS * sizeof(float));
    in_file.close();

{in_prep}
    {out_type}* {out_name} = nullptr;
    model_forward({in_arg}, &{out_name});

{out_dequant}
    std::ofstream out_file("cls_logits.bin", std::ios::binary);
    out_file.write(reinterpret_cast<const char*>(logits), OUTPUT_ELEMENTS * sizeof(float));
    out_file.close();
    return 0;
}}
"""
    (Path(out_dir) / "main.cpp").write_text(main_cpp)
    subprocess.run(
        ["cmake", "-B", f"{out_dir}/build", "-S", out_dir, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_FLAGS=-O3"],
        check=True,
    )
    subprocess.run(["cmake", "--build", f"{out_dir}/build", "-j1"], check=True)


def write_and_build_detector(out_dir, out_scales):
    shutil.copy2("third_party/stb_image.h", Path(out_dir) / "stb_image.h")

    in_type, outs = parse_forward_signature(Path(out_dir) / "dnn/include/forward.hpp")
    boxes_type, boxes_name = outs[0]
    scores_type, scores_name = outs[1]
    s_boxes = out_scales[0] if boxes_type == "int8_t" else 1.0
    s_scores = out_scales[1] if scores_type == "int8_t" else 1.0

    extra_decls = "\n".join(f"    {t}* {n} = nullptr;" for t, n in outs[2:])
    extra_args = "".join(f", &{n}" for _, n in outs[2:])

    if in_type == "int8_t":
        in_conv = """    std::vector<int8_t> quant_in(input_data.size());
    for (size_t i = 0; i < input_data.size(); ++i) {
        int q = static_cast<int>(std::round(input_data[i] * 127.0f));
        quant_in[i] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
    }"""
        in_arg = "quant_in.data()"
    else:
        in_conv = ""
        in_arg = "input_data.data()"

    main_cpp = f"""#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "forward.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct Detection {{
    int class_id;
    float score;
    float x1, y1, x2, y2;
}};

static float compute_iou(const Detection& a, const Detection& b)
{{
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

static bool load_and_preprocess(const std::string& path, std::vector<float>& out_nhwc, int& orig_w, int& orig_h)
{{
    constexpr int DST_W = 640;
    constexpr int DST_H = 640;
    out_nhwc.resize(1 * DST_H * DST_W * 3);

    int channels = 0;
    unsigned char* img = stbi_load(path.c_str(), &orig_w, &orig_h, &channels, 3);
    if (!img) return false;

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

    stbi_image_free(img);
    return true;
}}

int main(int argc, char** argv)
{{
    const std::string input_path = (argc > 1) ? argv[1] : "../../test_wildlife_1mp.jpg";
    const float conf_thresh = (argc > 2) ? std::strtof(argv[2], nullptr) : 0.25f;
    const float iou_thresh = 0.70f;
    const char* LABELS[3] = {{"animal", "person", "vehicle"}};
    constexpr int NUM_ANCHORS = 8400;
    constexpr float S_BOXES = {s_boxes:.10f}f;
    constexpr float S_SCORES = {s_scores:.10f}f;

    std::vector<float> input_data;
    int orig_w = 640, orig_h = 640;
    if (!load_and_preprocess(input_path, input_data, orig_w, orig_h)) {{
        std::cerr << "Erreur : impossible de charger " << input_path << std::endl;
        return 1;
    }}

{in_conv}
    {boxes_type}* {boxes_name} = nullptr;
    {scores_type}* {scores_name} = nullptr;
{extra_decls}

    model_forward({in_arg}, &{boxes_name}, &{scores_name}{extra_args});

    std::vector<Detection> candidates;
    for (int i = 0; i < NUM_ANCHORS; ++i) {{
        int best_c = 0;
        float best_logit = static_cast<float>({scores_name}[i]) * S_SCORES;
        for (int c = 1; c < 3; ++c) {{
            float l = static_cast<float>({scores_name}[c * NUM_ANCHORS + i]) * S_SCORES;
            if (l > best_logit) {{
                best_logit = l;
                best_c = c;
            }}
        }}
        float best_score = 1.0f / (1.0f + std::exp(-best_logit));
        if (best_score < conf_thresh) continue;

        float dist[4];
        for (int side = 0; side < 4; ++side) {{
            float max_v = static_cast<float>({boxes_name}[(side * 16) * NUM_ANCHORS + i]) * S_BOXES;
            for (int b = 1; b < 16; ++b) {{
                float v = static_cast<float>({boxes_name}[(side * 16 + b) * NUM_ANCHORS + i]) * S_BOXES;
                if (v > max_v) max_v = v;
            }}
            float sum_e = 0.0f;
            float acc = 0.0f;
            for (int b = 0; b < 16; ++b) {{
                float v = static_cast<float>({boxes_name}[(side * 16 + b) * NUM_ANCHORS + i]) * S_BOXES;
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

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Image: " << input_path << " (" << orig_w << "x" << orig_h << ") -> Detections (" << final_dets.size() << "):" << std::endl;
    for (const auto& d : final_dets) {{
        std::cout << "  [" << LABELS[d.class_id] << "] conf=" << d.score
                  << " bbox=[" << d.x1 << ", " << d.y1 << ", " << d.x2 << ", " << d.y2 << "]"
                  << std::endl;
    }}
    return 0;
}}
"""
    (Path(out_dir) / "main.cpp").write_text(main_cpp)
    subprocess.run(
        ["cmake", "-B", f"{out_dir}/build", "-S", out_dir, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_FLAGS=-O3"],
        check=True,
    )
    subprocess.run(["cmake", "--build", f"{out_dir}/build", "-j1"], check=True)


def main():
    apply_aidge_export_fixes()

    print("=== 1/2 : EXPORT & COMPILATION INT8 DU CLASSIFIEUR (128x128) ===")
    with open("class_names.json") as f:
        class_names = json.load(f)

    def load_cls_img(path):
        im = Image.open(path).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        arr = ((np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD) / CLS_NORM_DIV
        return make_tensor(np.transpose(arr, (2, 0, 1))[None, ...])

    calib_cls = []
    for cname in class_names:
        d = os.path.join("dataset_wildlife", "train", cname)
        for fn in sorted(os.listdir(d))[:4]:
            calib_cls.append(load_cls_img(os.path.join(d, fn)))

    cls_out_scales = export_int8_cpp(build_classifier_graph, calib_cls, "export_cpp_classifier_int8")
    print(f"Scale sortie classifieur : {cls_out_scales[0]:.6f}")
    write_and_build_classifier("export_cpp_classifier_int8", cls_out_scales[0])

    print("\n=== 2/2 : EXPORT & COMPILATION INT8 DU DETECTEUR (640x640) ===")
    orig_img = Image.open("test_wildlife_1mp.jpg").convert("RGB")
    W, H = orig_img.size

    def load_det_pil(im):
        r = im.resize((640, 640), Image.Resampling.BILINEAR)
        arr = np.asarray(r, dtype=np.float32) / 255.0
        return make_tensor(np.transpose(arr, (2, 0, 1))[None, ...])

    calib_det = [load_det_pil(orig_img)]
    tw, th = int(W * 0.6), int(H * 0.6)
    for ox, oy in [(0, 0), (W - tw, 0), (0, H - th), (W - tw, H - th)]:
        calib_det.append(load_det_pil(orig_img.crop((ox, oy, ox + tw, oy + th))))
    for cname in class_names:
        d = os.path.join("dataset_wildlife", "train", cname)
        for fn in sorted(os.listdir(d))[:3]:
            calib_det.append(load_det_pil(Image.open(os.path.join(d, fn)).convert("RGB")))

    det_out_scales = export_int8_cpp(build_detector_graph, calib_det, "export_cpp_detector_int8")
    print(f"Scales sortie detecteur : {det_out_scales}")
    write_and_build_detector("export_cpp_detector_int8", det_out_scales)
    print("\nCompilation Release INT8 terminee avec succes.")


if __name__ == "__main__":
    main()
