import shutil
import subprocess
from pathlib import Path
import numpy as np
import onnxruntime as ort
import aidge_core
import aidge_onnx
import aidge_backend_cpu
import aidge_export_cpp

from mountain import paths

ONNX_MODEL = str(paths.CLASSIFIER_ONNX)
FOLDED_MODEL = str(paths.CLASSIFIER_ONNX_FOLDED)
EXPORT_DIR = str(paths.EXPORT_CLASSIFIER_FP32)

# 1. Pliage des nœuds Constant en initializers via ONNX Runtime (comme export_detector_v6.py)
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
opts.optimized_model_filepath = FOLDED_MODEL
ort.InferenceSession(ONNX_MODEL, opts, providers=["CPUExecutionProvider"])

# 2. Chargement et optimisation du graphe Aidge
graph = aidge_onnx.load_onnx(FOLDED_MODEL)
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

in_shape = [1, 3, 128, 128]
dummy_input = aidge_core.Tensor(np.zeros(in_shape, dtype=np.float32))
dummy_input.set_datatype(aidge_core.dtype.float32)
dummy_input.set_data_format(aidge_core.dformat.nchw)
graph.forward_dims([in_shape])

# 3. Export C++ propre
shutil.rmtree(EXPORT_DIR, ignore_errors=True)
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

# 4. Génération de main.cpp (lecture cls_input.bin -> écriture cls_logits.bin)
main_cpp = """#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include "forward.hpp"

int main()
{
    constexpr size_t INPUT_ELEMENTS  = 1 * 128 * 128 * 3;
    constexpr size_t OUTPUT_ELEMENTS = 5;

    std::vector<float> input_data(INPUT_ELEMENTS);
    std::ifstream in_file("cls_input.bin", std::ios::binary);
    if (!in_file) {
        std::cerr << "Erreur : impossible d'ouvrir cls_input.bin" << std::endl;
        return 1;
    }
    in_file.read(reinterpret_cast<char*>(input_data.data()), INPUT_ELEMENTS * sizeof(float));
    in_file.close();

    float* logits_ptr = nullptr;
    model_forward(input_data.data(), &logits_ptr);

    std::ofstream out_file("cls_logits.bin", std::ios::binary);
    out_file.write(reinterpret_cast<const char*>(logits_ptr), OUTPUT_ELEMENTS * sizeof(float));
    out_file.close();

    return 0;
}
"""
(Path(EXPORT_DIR) / "main.cpp").write_text(main_cpp)

# 5. Compilation Release -O3 sur 1 cœur

import re as _re
_hpp = Path(f"{EXPORT_DIR}/dnn/include/forward.hpp").read_text()
_m = _re.search(r"void\s+model_forward\s*\(([^)]+)\)", _hpp, _re.S)
_params = [x.strip() for x in _m.group(1).split(",")]
_decls, _args = [], ["input_data.data()"]
for _i, _param in enumerate(_params[1:], start=1):
    if "float**" in _param or "float **" in _param:
        _decls.append("    float* logits_ptr = nullptr;")
        _args.append("&logits_ptr")
    else:
        _t = _param.split()[0].rstrip("*")
        _decls.append(f"    {_t}* unused_out_{_i} = nullptr;")
        _args.append(f"&unused_out_{_i}")
_call_args = ", ".join(_args)
_main_p = Path(f"{EXPORT_DIR}/main.cpp")
_main_txt = _main_p.read_text()
_main_txt = _main_txt.replace("    float* logits_ptr = nullptr;\n", "")
_main_txt = _main_txt.replace(
    "    model_forward(input_data.data(), &logits_ptr);",
    "\n".join(_decls) + f"\n    model_forward({_call_args});"
)
_main_p.write_text(_main_txt)

subprocess.run(["cmake", "-B", f"{EXPORT_DIR}/build", "-S", EXPORT_DIR,
                "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_FLAGS=-O3"], check=True)
subprocess.run(["cmake", "--build", f"{EXPORT_DIR}/build", "-j1"], check=True)
print("Export et compilation du classifieur terminés.")
