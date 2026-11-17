import os
import json
import inspect
import numpy as np
from PIL import Image
import aidge_core
import aidge_backend_cpu
import aidge_onnx
import aidge_quantization
import aidge_export_cpp

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def replace_flatten_with_reshape(graph):
    for node in list(graph.get_nodes()):
        if node.type() == "Flatten":
            out_dims = node.get_operator().get_output(0).dims
            if isinstance(out_dims, list) and len(out_dims) > 0 and isinstance(out_dims[0], list):
                out_dims = out_dims[0]
            reshape_node = aidge_core.Reshape(shape=out_dims, name=node.name() + "_reshape")
            aidge_core.GraphView.replace({node}, {reshape_node})

def prepare_graph(onnx_path, input_shape, fuse_bn=True):
    graph = aidge_onnx.load_onnx(onnx_path)
    for node in graph.get_nodes():
        if node.type() == "Producer":
            op = node.get_operator()
            if hasattr(op, "set_constant"):
                op.set_constant(True)
            elif hasattr(op, "attr") and hasattr(op.attr, "constant"):
                op.attr.constant = True
    aidge_core.constant_folding(graph)
    if fuse_bn and hasattr(aidge_core, "fuse_batchnorm"):
        try:
            aidge_core.fuse_batchnorm(graph)
        except Exception:
            pass
    graph.forward_dims([input_shape])
    replace_flatten_with_reshape(graph)
    graph.forward_dims([input_shape])
    graph.set_datatype(aidge_core.dtype.float32)
    graph.set_backend("cpu")
    return graph

def apply_winning_ptq(graph, calib_tensors, input_shape):
    ptq = aidge_quantization.PTQ(graph)
    ptq.weight_precision = aidge_core.dtype.int8
    ptq.activation_precision = aidge_core.dtype.int8
    ptq.accumulation_precision = aidge_core.dtype.int32
    ptq.use_zero_point = False
    ptq.channel_wise = True
    ptq.fuse_dqq = True
    ptq.single_shift_approximation = False
    ptq.verbose = False
    q_graph = ptq.process(calib_tensors)
    q_graph.set_backend("cpu")
    q_graph.forward_dims([input_shape])
    return q_graph

def export_cpp_modern(fp32_graph, calib_tensors, out_dir):
    for t in calib_tensors:
        # Correction : dims est un attribut (propriété), pas une méthode
        t_dims = t.dims
        if isinstance(t_dims, list) and len(t_dims) > 0 and isinstance(t_dims[0], list):
            t_dims = t_dims[0]
        if len(t_dims) == 4:
            t.set_data_format(aidge_core.dformat.nchw)
            
    ExportConfigCls = getattr(aidge_export_cpp, "ExportConfig", aidge_core.export_utils.config.ExportConfig)
    cfg_params = inspect.signature(ExportConfigCls).parameters
    cfg_kwargs = {
        "target": "host",
        "export_folder": out_dir,
        "dtype": "int8",
    }
    if "use_zero_point" in cfg_params:
        cfg_kwargs["use_zero_point"] = False
    if "channel_wise" in cfg_params:
        cfg_kwargs["channel_wise"] = True
    config = ExportConfigCls(**cfg_kwargs)
    aidge_core.export_utils.export(
        graph=fp32_graph,
        config=config,
        input_tensors=calib_tensors,
        nb_test=1,
    )

# --- 1. Export C++ du Classifieur ---
print("1/2 - Export C++ INT8 du Classifieur...")
with open("class_names.json", "r") as f:
    class_names = json.load(f)

calib_cls = []
for cls_name in class_names:
    cls_dir = os.path.join("dataset_wildlife", "val", cls_name)
    for fname in sorted(os.listdir(cls_dir))[:8]:
        img = Image.open(os.path.join(cls_dir, fname)).convert("RGB").resize((128, 128), Image.BILINEAR)
        arr = (np.array(img, dtype=np.float32) / 255.0 - MEAN) / STD
        t = aidge_core.Tensor(np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...]))
        t.set_backend("cpu")
        calib_cls.append(t)

g_cls = prepare_graph("species_classifier.onnx", [1, 3, 128, 128], fuse_bn=True)
export_cpp_modern(g_cls, calib_cls, "export_cpp_classifier")
print(f" -> Export C++ Classifieur généré dans ./export_cpp_classifier : {os.listdir('export_cpp_classifier')}")

# --- 2. Quantification INT8 + Validation détection + Export C++ de MDV6 ---
print("\n2/2 - Quantification INT8 et export C++ de MDV6-yolov10-c...")
orig_img = Image.open("test_wildlife_1mp.jpg").convert("RGB")
W, H = orig_img.size

calib_det = []
crops_to_calib = [orig_img]
tw, th = int(W * 0.575), int(H * 0.575)
for ox, oy in [(0, 0), (W - tw, 0), (0, H - th), (W - tw, H - th)]:
    crops_to_calib.append(orig_img.crop((ox, oy, ox + tw, oy + th)))
for cls_name in class_names:
    cls_dir = os.path.join("dataset_wildlife", "val", cls_name)
    for fname in sorted(os.listdir(cls_dir))[:4]:
        crops_to_calib.append(Image.open(os.path.join(cls_dir, fname)).convert("RGB"))

for c in crops_to_calib:
    r = c.resize((320, 320), Image.BILINEAR)
    arr = np.ascontiguousarray(np.transpose(np.array(r, dtype=np.float32) / 255.0, (2, 0, 1))[None, ...])
    t = aidge_core.Tensor(arr)
    t.set_backend("cpu")
    calib_det.append(t)

# Validation numérique INT8 de MDV6
g_det_eval = prepare_graph("MDV6-yolov10-c.onnx", [1, 3, 320, 320], fuse_bn=True)
q_det = apply_winning_ptq(g_det_eval, calib_det, [1, 3, 320, 320])
sched_det = aidge_core.SequentialScheduler(q_det)
out_det = list(q_det.get_output_nodes())[0]
sched_det.forward(data=[calib_det[0]])
preds = np.array(out_det.get_operator().get_output(0))[0].T
best_row = preds[np.argmax(preds[:, 4])]
cx, cy, bw, bh, score = best_row[:5]
box_int8 = (int((cx - bw/2)*W/320), int((cy - bh/2)*H/320), int((cx + bw/2)*W/320), int((cy + bh/2)*H/320))
print(f" -> Détection MDV6 INT8 (passe globale) : confiance={score*100:.1f}% | box={box_int8}")

try:
    g_det_export = prepare_graph("MDV6-yolov10-c.onnx", [1, 3, 320, 320], fuse_bn=True)
    export_cpp_modern(g_det_export, calib_det, "export_cpp_mdv6")
    print(f" -> Export C++ MDV6 généré dans ./export_cpp_mdv6 : {os.listdir('export_cpp_mdv6')}")
except Exception as e:
    print(f" -> Note export C++ MDV6 : {e}")
