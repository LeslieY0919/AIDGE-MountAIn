import os
import json
import numpy as np
from PIL import Image
import aidge_core
import aidge_backend_cpu
import aidge_onnx
import aidge_quantization

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

# 1. Chargement des 80 images de validation
with open("class_names.json", "r") as f:
    class_names = json.load(f)

val_samples = []
for cls_idx, cls_name in enumerate(class_names):
    cls_dir = os.path.join("dataset_wildlife", "val", cls_name)
    for fname in sorted(os.listdir(cls_dir)):
        val_samples.append((os.path.join(cls_dir, fname), cls_idx))

# 2. Évaluation référence FP32
graph_fp32 = build_graph("species_classifier.onnx")
sched_fp32 = aidge_core.SequentialScheduler(graph_fp32)
out_fp32 = list(graph_fp32.get_output_nodes())[0]

fp32_preds = []
correct_fp32 = 0
for path, label in val_samples:
    t = aidge_core.Tensor(preprocess(path))
    t.set_backend("cpu")
    sched_fp32.forward(data=[t])
    logits = np.array(out_fp32.get_operator().get_output(0))[0]
    pred = int(np.argmax(logits))
    fp32_preds.append(pred)
    correct_fp32 += int(pred == label)

acc_fp32 = 100.0 * correct_fp32 / len(val_samples)

# 3. Calibration PTQ INT8 avec aidge_quantization.PTQ
def run_ptq_eval(use_fake_quant):
    g = build_graph("species_classifier.onnx")
    calib_tensors = []
    for path, _ in val_samples[::2]:
        t = aidge_core.Tensor(preprocess(path))
        t.set_backend("cpu")
        calib_tensors.append(t)

    ptq = aidge_quantization.PTQ(g)
    ptq.weight_precision = aidge_core.dtype.int8
    ptq.activation_precision = aidge_core.dtype.int8
    ptq.accumulation_precision = aidge_core.dtype.int32
    ptq.single_shift_approximation = True
    ptq.channel_wise = True
    ptq.fake_quantize = use_fake_quant
    ptq.verbose = False

    q_graph = ptq.process(calib_tensors)
    q_graph.set_backend("cpu")
    q_graph.forward_dims([[1, 3, 128, 128]])

    sched = aidge_core.SequentialScheduler(q_graph)
    out_node = list(q_graph.get_output_nodes())[0]

    correct, agree = 0, 0
    for idx, (path, label) in enumerate(val_samples):
        t = aidge_core.Tensor(preprocess(path))
        t.set_backend("cpu")
        sched.forward(data=[t])
        logits = np.array(out_node.get_operator().get_output(0))[0]
        pred = int(np.argmax(logits))
        correct += int(pred == label)
        agree += int(pred == fp32_preds[idx])

    return correct, agree, ptq.get_diagnostic()

# Exécution en entiers natifs (repli sur simulation quantifiée si un noyau CPU int8 manque)
try:
    mode_label = "INT8 natif (CPU)"
    correct_int8, agreement, diag = run_ptq_eval(use_fake_quant=False)
except Exception as e:
    print(f"Note : bascule sur simulation quantifiée INT8 ({e})")
    mode_label = "INT8 quantifié (QDQ)"
    correct_int8, agreement, diag = run_ptq_eval(use_fake_quant=True)

acc_int8 = 100.0 * correct_int8 / len(val_samples)
agree_rate = 100.0 * agreement / len(val_samples)

print(f"\n--- Bilan Quantification Aidge ({mode_label}) ---")
print(f"Précision FP32 : {acc_fp32:.1f}% ({correct_fp32}/{len(val_samples)})")
print(f"Précision INT8 : {acc_int8:.1f}% ({correct_int8}/{len(val_samples)})")
print(f"Écart FP32 -> INT8 : {acc_int8 - acc_fp32:+.1f} pts | Accord prédictions : {agree_rate:.1f}%")
