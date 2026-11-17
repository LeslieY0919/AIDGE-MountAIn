import pathlib
import onnx
import numpy as np
import aidge_core
import aidge_onnx
import aidge_backend_cpu
import aidge_export_cpp

MODEL_PATH = "MDV6-yolov10-c.onnx"
FOLDED_PATH = "MDV6-yolov10-c_folded.onnx"
EXPORT_DIR = "export_cpp_detector"

def fold_onnx_constants(src: str, dst: str) -> str:
    # 1. Essai via onnxsim si installe
    try:
        import onnxsim
        model_opt, ok = onnxsim.simplify(src)
        if ok:
            onnx.save(model_opt, dst)
            print(f"Constantes pliees via onnxsim -> {dst}")
            return dst
    except ImportError:
        pass

    # 2. Essai via ONNX Runtime (ORT_ENABLE_BASIC plie les constantes et formes)
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.optimized_model_filepath = dst
        ort.InferenceSession(src, opts, providers=["CPUExecutionProvider"])
        print(f"Constantes pliees via ONNX Runtime -> {dst}")
        return dst
    except ImportError:
        pass

    print("Ni onnxsim ni onnxruntime trouves, utilisation du modele brut.")
    return src

clean_model = fold_onnx_constants(MODEL_PATH, FOLDED_PATH)

graph = aidge_onnx.load_onnx(clean_model)
aidge_core.remove_flatten(graph)
graph.set_datatype(aidge_core.dtype.float32)
graph.set_backend("cpu")

in_shape = [1, 3, 320, 320]
dummy_input = aidge_core.Tensor(np.zeros(in_shape, dtype=np.float32))
dummy_input.set_datatype(aidge_core.dtype.float32)
dummy_input.set_data_format(aidge_core.dformat.nchw)

graph.forward_dims([in_shape], allow_data_dependency=True)

# Diagnostic des eventuels Transpose restants avant export
for node in graph.get_nodes():
    if node.type() == "Transpose":
        in0 = node.get_operator().get_input(0)
        dims = in0.dims if in0 is not None else None
        print(f"Transpose trouve : {node.name()} | input0 dims = {dims}")

config = aidge_export_cpp.ExportConfig(
    export_folder=EXPORT_DIR,
    dtype="float32",
    cuda=False,
    compile=True,
)

artifact = aidge_core.export_utils.export(
    graph=graph,
    config=config,
    input_tensors=[dummy_input],
)
print(f"Export et compilation termines dans ./{EXPORT_DIR}")
