import faulthandler
faulthandler.enable()

import onnx
import numpy as np
import aidge_core
import aidge_onnx
import aidge_backend_cpu
import aidge_export_cpp

from mountain import paths

MODEL_PATH = str(paths.DETECTOR_ONNX)

# 1. Inspection de la structure ONNX (entrees, sorties, fin du graphe)
onnx_model = onnx.load(MODEL_PATH)
print("=== ENTREES ONNX ===")
for inp in onnx_model.graph.input:
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {dims}")

print("\n=== SORTIES ONNX ===")
for out in onnx_model.graph.output:
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {dims}")

print("\n=== 15 DERNIERS NOEUDS ONNX ===")
for node in onnx_model.graph.node[-15:]:
    print(f"  [{node.op_type}] {node.name} -> {list(node.output)}")

# 2. Import Aidge et test d'export C++ FP32 (sans PTQ)
print("\n=== IMPORT AIDGE & EXPORT C++ FP32 ===")
graph = aidge_onnx.load_onnx(MODEL_PATH)
aidge_core.remove_flatten(graph)
graph.set_datatype(aidge_core.dtype.float32)
graph.set_backend("cpu")

# Adaptation automatique a la taille d'entree du modele ONNX
in_shape = [d.dim_value or 1 for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
dummy_input = aidge_core.Tensor(np.zeros(in_shape, dtype=np.float32))
graph.forward_dims([in_shape])

aidge_export_cpp.export(
    export_folder_name=str(paths.EXPORT_DETECTOR_FP32),
    graphview=graph,
    scheduler=aidge_core.SequentialScheduler(graph),
    inputs_tensor=[dummy_input]
)
print("Export C++ FP32 termine avec succes dans ./export_cpp_detector")
