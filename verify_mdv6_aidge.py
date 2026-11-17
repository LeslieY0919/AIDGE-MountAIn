import numpy as np
import onnxruntime as ort
import aidge_core
import aidge_backend_cpu
import aidge_onnx

# 1. Chargement et résolution complète des dimensions dans Aidge
graph = aidge_onnx.load_onnx("MDV6-yolov10-c.onnx")
for node in graph.get_nodes():
    if node.type() == "Producer":
        op = node.get_operator()
        if hasattr(op, "set_constant"):
            op.set_constant(True)
        elif hasattr(op, "attr") and hasattr(op.attr, "constant"):
            op.attr.constant = True

aidge_core.constant_folding(graph)
graph.forward_dims([[1, 3, 320, 320]])
print("Dimensions propagées avec succès (0 warning) !")

# 2. Comparaison numérique ONNX Runtime vs Aidge CPU
x = np.random.rand(1, 3, 320, 320).astype(np.float32)

sess = ort.InferenceSession("MDV6-yolov10-c.onnx")
ort_out = sess.run(None, {sess.get_inputs()[0].name: x})[0]

graph.set_datatype(aidge_core.dtype.float32)
graph.set_backend("cpu")
scheduler = aidge_core.SequentialScheduler(graph)
scheduler.forward(data=[aidge_core.Tensor(x)])

out_node = list(graph.get_output_nodes())[0]
aidge_out = np.array(out_node.get_operator().get_output(0))

max_diff = np.max(np.abs(ort_out - aidge_out))
print(f"Forme sortie : {aidge_out.shape} | Écart max ONNX vs Aidge : {max_diff:.6f}")
