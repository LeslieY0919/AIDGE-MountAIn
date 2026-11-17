import pathlib
import numpy as np
import aidge_core
import aidge_onnx
import aidge_backend_cpu
import aidge_export_cpp

# Inspection rapide de adapt_graph.py autour de la ligne 723
adapt_file = pathlib.Path(aidge_core.export_utils.__file__).parent / "adapt_graph.py"
lines = adapt_file.read_text().splitlines()
print("=== adapt_graph.py (lignes 680-730) ===")
for idx in range(679, min(730, len(lines))):
    print(f"{idx+1}: {lines[idx]}")
print("=======================================\n")

MODEL_PATH = "MDV6-yolov10-c_folded.onnx"
EXPORT_DIR = "export_cpp_detector"

graph = aidge_onnx.load_onnx(MODEL_PATH)
aidge_core.remove_flatten(graph)
graph.set_datatype(aidge_core.dtype.float32)
graph.set_backend("cpu")

in_shape = [1, 3, 320, 320]
dummy_input = aidge_core.Tensor(np.zeros(in_shape, dtype=np.float32))
dummy_input.set_datatype(aidge_core.dtype.float32)
dummy_input.set_data_format(aidge_core.dformat.nchw)

graph.forward_dims([in_shape], allow_data_dependency=True)

# Promotion des scalaires 0D (dims == []) en [1, 1, 1, 1] pour supporter Transpose NCHW->NHWC
for node in graph.get_nodes():
    if node.type() == "Producer":
        t = node.get_operator().get_output(0)
        if t is not None and len(t.dims) == 0:
            children_types = {c.type() for c in node.get_children()}
            print(f"Scalaire 0D detecte : {node.name()} -> enfants: {children_types}")
            if children_types.issubset({"Mul", "Add", "Sub", "Div"}):
                arr = np.array(t).reshape(1, 1, 1, 1)
                new_t = aidge_core.Tensor(arr)
                new_t.set_datatype(t.dtype)
                new_t.set_data_format(aidge_core.dformat.nchw)
                node.get_operator().set_output(0, new_t)
                print("  -> Promu en [1, 1, 1, 1] (NCHW)")

graph.forward_dims([in_shape], allow_data_dependency=True)

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
print(f"\nExport et compilation termines avec succes dans ./{EXPORT_DIR}")
