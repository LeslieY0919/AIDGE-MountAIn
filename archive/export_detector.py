import inspect
import onnx
import numpy as np
import aidge_core
import aidge_onnx
import aidge_backend_cpu
import aidge_export_cpp

MODEL_PATH = "MDV6-yolov10-c.onnx"
CUT_MODEL_PATH = "MDV6-yolov10-c_cut.onnx"
EXPORT_DIR = "export_cpp_detector"

def run_cpp_export(onnx_path: str, out_dir: str):
    graph = aidge_onnx.load_onnx(onnx_path)
    aidge_core.remove_flatten(graph)
    graph.set_datatype(aidge_core.dtype.float32)
    graph.set_backend("cpu")

    in_shape = [1, 3, 320, 320]
    dummy_input = aidge_core.Tensor(np.zeros(in_shape, dtype=np.float32))

    # Propagation des dimensions avec evaluation des constantes (Split, Reshape, Slice)
    graph.forward_dims([in_shape], allow_data_dependency=True)

    scheduler = aidge_core.SequentialScheduler(graph)
    scheduler.generate_scheduling()

    sig = inspect.signature(aidge_export_cpp.export).parameters
    kwargs = {}
    if "export_folder_name" in sig:
        kwargs["export_folder_name"] = out_dir
    if "model" in sig:
        kwargs["model"] = graph
    elif "graph_view" in sig:
        kwargs["graph_view"] = graph
    elif "graphview" in sig:
        kwargs["graphview"] = graph
    if "scheduler" in sig:
        kwargs["scheduler"] = scheduler
    if "inputs_tensor" in sig:
        kwargs["inputs_tensor"] = [dummy_input]

    aidge_export_cpp.export(**kwargs)

try:
    print("=== TENTATIVE 1 : Export FP32 complet (sortie output0 [1, 7, 2100]) ===")
    run_cpp_export(MODEL_PATH, EXPORT_DIR)
    print(f"Succes : export complet genere dans ./{EXPORT_DIR}")
except Exception as e:
    print(f"\nEchec sur le graphe complet ({type(e).__name__}: {e})")
    print("=== TENTATIVE 2 : Coupe avant le post-traitement (/model.23/Concat_1_output_0) ===")
    onnx.utils.extract_model(
        MODEL_PATH,
        CUT_MODEL_PATH,
        input_names=["images"],
        output_names=["/model.23/Concat_1_output_0"],
    )
    run_cpp_export(CUT_MODEL_PATH, EXPORT_DIR)
    print(f"Succes : export sans post-traitement genere dans ./{EXPORT_DIR}")
