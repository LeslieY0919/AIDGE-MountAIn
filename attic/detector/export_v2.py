import inspect
import dataclasses
import onnx
import numpy as np
import aidge_core
import aidge_onnx
import aidge_backend_cpu
import aidge_export_cpp

MODEL_PATH = "MDV6-yolov10-c.onnx"
CUT_MODEL_PATH = "MDV6-yolov10-c_cut.onnx"
EXPORT_DIR = "export_cpp_detector"

print("Signature ExportConfig :", inspect.signature(aidge_export_cpp.ExportConfig))

def build_export_config(out_dir: str):
    sig = inspect.signature(aidge_export_cpp.ExportConfig).parameters
    kwargs = {}
    for key in ("export_path", "output_dir", "export_folder_name", "path"):
        if key in sig:
            kwargs[key] = out_dir
            break
    if "export_lib" in sig:
        kwargs["export_lib"] = aidge_export_cpp.ExportLibCpp
    return aidge_export_cpp.ExportConfig(**kwargs)

def run_new_export(onnx_path: str, out_dir: str):
    graph = aidge_onnx.load_onnx(onnx_path)
    aidge_core.remove_flatten(graph)
    graph.set_datatype(aidge_core.dtype.float32)
    graph.set_backend("cpu")

    in_shape = [1, 3, 320, 320]
    dummy_input = aidge_core.Tensor(np.zeros(in_shape, dtype=np.float32))
    dummy_input.set_datatype(aidge_core.dtype.float32)
    dummy_input.set_data_format(aidge_core.dformat.nchw)

    graph.forward_dims([in_shape], allow_data_dependency=True)

    config = build_export_config(out_dir)
    artifact = aidge_core.export_utils.export(
        graph=graph,
        config=config,
        input_tensors=[dummy_input],
    )
    return artifact

try:
    print("\n=== TENTATIVE 1 : Export complet via export_utils.export ===")
    run_new_export(MODEL_PATH, EXPORT_DIR)
    print(f"Succes complet dans ./{EXPORT_DIR}")
except Exception as e:
    print(f"\nEchec sur le graphe complet ({type(e).__name__}: {e})")
    print("=== TENTATIVE 2 : Export coupe avant post-traitement (/model.23/Concat_1_output_0) ===")
    onnx.utils.extract_model(
        MODEL_PATH,
        CUT_MODEL_PATH,
        input_names=["images"],
        output_names=["/model.23/Concat_1_output_0"],
    )
    run_new_export(CUT_MODEL_PATH, EXPORT_DIR)
    print(f"Succes sur le modele coupe dans ./{EXPORT_DIR}")
