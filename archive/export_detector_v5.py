import numpy as np
import aidge_core
import aidge_onnx
import aidge_backend_cpu
import aidge_export_cpp

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

# 1. Calcul initial des dimensions pour connaitre le rang de chaque branche
graph.forward_dims([in_shape], allow_data_dependency=True)

# 2. Promotion des scalaires 0D au rang exact de l'entree voisine (3D ou 4D)
for node in graph.get_nodes():
    if node.type() == "Producer":
        t = node.get_operator().get_output(0)
        if t is not None and len(t.dims) == 0:
            target_rank = 0
            for child in node.get_children():
                op = child.get_operator()
                for idx in range(child.get_nb_inputs()):
                    in_t = op.get_input(idx)
                    if in_t is not None and len(in_t.dims) > target_rank:
                        target_rank = len(in_t.dims)

            if target_rank > 0:
                new_shape = (1,) * target_rank
                new_t = aidge_core.Tensor(np.array(t).reshape(new_shape))
                new_t.set_datatype(t.dtype)
                new_t.set_data_format(
                    aidge_core.dformat.nchw if target_rank == 4 else aidge_core.dformat.default
                )
                node.get_operator().set_output(0, new_t)
                print(f"Scalaire {node.name()} promu en {list(new_shape)}")

# 3. Verification de la propagation des dimensions
graph.forward_dims([in_shape], allow_data_dependency=True)

# 4. Export et compilation C++
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
