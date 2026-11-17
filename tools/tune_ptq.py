import os
import json
import numpy as np
from PIL import Image
import aidge_core
import aidge_backend_cpu
import aidge_onnx
import aidge_quantization

from mountain import paths

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize((128, 128), Image.BILINEAR)
    arr = (np.array(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...])

def build_graph(onnx_path):
    graph = aidge_onnx.load_onnx(onnx_path)
    for node in graph.get_nodes():
        if node.type() == "Producer":
            op = node.get_operator()
            if hasattr(op, "set_constant"):
                op.set_constant(True)
            elif hasattr(op, "attr") and hasattr(op.attr, "constant"):
                op.attr.constant = True
    aidge_core.constant_folding(graph)
    if hasattr(aidge_core, "fuse_batchnorm"):
        try:
            aidge_core.fuse_batchnorm(graph)
        except Exception:
            pass
    graph.forward_dims([[1, 3, 128, 128]])
    graph.set_datatype(aidge_core.dtype.float32)
    graph.set_backend("cpu")
    return graph

with open(str(paths.CLASS_NAMES), "r") as f:
    class_names = json.load(f)

val_samples = []
for cls_idx, cls_name in enumerate(class_names):
    cls_dir = os.path.join(str(paths.WILDLIFE_DATASET), "val", cls_name)
    for fname in sorted(os.listdir(cls_dir)):
        val_samples.append((os.path.join(cls_dir, fname), cls_idx))

# Référence FP32
g_fp32 = build_graph(str(paths.CLASSIFIER_ONNX))
s_fp32 = aidge_core.SequentialScheduler(g_fp32)
o_fp32 = list(g_fp32.get_output_nodes())[0]
fp32_preds = []
for path, _ in val_samples:
    t = aidge_core.Tensor(preprocess(path))
    t.set_backend("cpu")
    s_fp32.forward(data=[t])
    fp32_preds.append(int(np.argmax(np.array(o_fp32.get_operator().get_output(0))[0])))

acc_fp32 = 100.0 * sum(p == y for p, (_, y) in zip(fp32_preds, val_samples)) / len(val_samples)

# Instanciation des stratégies de clipping disponibles
clip_perc = None
if hasattr(aidge_quantization, "PercentileClipping"):
    try:
        clip_perc = aidge_quantization.PercentileClipping(0.999)
    except Exception:
        clip_perc = aidge_quantization.PercentileClipping()

clip_hist = None
if hasattr(aidge_quantization, "Histogram"):
    try:
        clip_hist = aidge_quantization.Histogram()
    except Exception:
        pass

configs = [
    {"name": "1. Asym + ChannelWise (MaxClip)",          "zp": True,  "cw": True,  "ssa": False, "clip": None},
    {"name": "2. Asym + ChannelWise + SingleShift",      "zp": True,  "cw": True,  "ssa": True,  "clip": None},
    {"name": "3. Asym + ChannelWise + Percentile",       "zp": True,  "cw": True,  "ssa": False, "clip": clip_perc},
    {"name": "4. Sym  + ChannelWise + Percentile",       "zp": False, "cw": True,  "ssa": False, "clip": clip_perc},
    {"name": "5. Asym + ChannelWise + Histogram",        "zp": True,  "cw": True,  "ssa": False, "clip": clip_hist},
]

print(f"Référence FP32 : {acc_fp32:.1f}%\n")
print(f"{'Configuration PTQ Aidge (FuseDQQ=True)':<45} | {'Précision INT8':<15} | {'Accord FP32'}")
print("-" * 78)

for cfg in configs:
    if cfg["clip"] is None and ("Percentile" in cfg["name"] or "Histogram" in cfg["name"]):
        continue
    try:
        g = build_graph(str(paths.CLASSIFIER_ONNX))
        calib = []
        for path, _ in val_samples:
            t = aidge_core.Tensor(preprocess(path))
            t.set_backend("cpu")
            calib.append(t)

        ptq = aidge_quantization.PTQ(g)
        ptq.weight_precision = aidge_core.dtype.int8
        ptq.activation_precision = aidge_core.dtype.int8
        ptq.accumulation_precision = aidge_core.dtype.int32
        ptq.fuse_dqq = True
        ptq.optimize_fork_precisions = False
        ptq.use_zero_point = cfg["zp"]
        ptq.channel_wise = cfg["cw"]
        ptq.single_shift_approximation = cfg["ssa"]
        ptq.verbose = False
        if cfg["clip"] is not None:
            ptq.clipping = cfg["clip"]

        q_graph = ptq.process(calib)
        q_graph.set_backend("cpu")
        q_graph.forward_dims([[1, 3, 128, 128]])

        sched = aidge_core.SequentialScheduler(q_graph)
        out_node = list(q_graph.get_output_nodes())[0]

        correct, agree = 0, 0
        for idx, (path, label) in enumerate(val_samples):
            t = aidge_core.Tensor(preprocess(path))
            t.set_backend("cpu")
            sched.forward(data=[t])
            pred = int(np.argmax(np.array(out_node.get_operator().get_output(0))[0]))
            correct += int(pred == label)
            agree += int(pred == fp32_preds[idx])

        acc = 100.0 * correct / len(val_samples)
        agr = 100.0 * agree / len(val_samples)
        print(f"{cfg['name']:<45} | {acc:>6.1f}% ({correct:02d}/80) | {agr:>6.1f}%")
    except Exception as e:
        print(f"{cfg['name']:<45} | Erreur : {e}")
