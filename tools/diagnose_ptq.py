import os
import json
import numpy as np
from PIL import Image
import aidge_core
import aidge_backend_cpu
import aidge_onnx
import aidge_quantization

from mountain import paths, preprocess

# Definition unique dans mountain.preprocess : la normalisation de la
# calibration PTQ doit etre rigoureusement identique a celle de
# l'entrainement, sinon la quantification s'etalonne sur la mauvaise
# distribution. Reexportees ici sous leurs anciens noms.
MEAN = preprocess.IMAGENET_MEAN
STD = preprocess.IMAGENET_STD

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

# 1. Inspecter les membres exacts de ClippingStrategy
if hasattr(aidge_quantization, "ClippingStrategy"):
    members = getattr(aidge_quantization.ClippingStrategy, "__members__", {})
    print("ClippingStrategy disponibles :", list(members.keys()))

# 2. Référence FP32 + logits des 3 premières images
g_fp32 = build_graph(str(paths.CLASSIFIER_ONNX))
s_fp32 = aidge_core.SequentialScheduler(g_fp32)
o_fp32 = list(g_fp32.get_output_nodes())[0]

fp32_preds = []
print("\nLogits FP32 (3 premières images) :")
for i, (path, _) in enumerate(val_samples):
    t = aidge_core.Tensor(preprocess(path))
    t.set_backend("cpu")
    s_fp32.forward(data=[t])
    logits = np.array(o_fp32.get_operator().get_output(0))[0]
    if i < 3:
        print(f"  Img {i} : {np.round(logits, 3)} -> classe {np.argmax(logits)}")
    fp32_preds.append(int(np.argmax(logits)))

# 3. Tests d'isolation (QDQ simulé vs INT8 natif symétrique)
tests = [
    {"name": "A. Simulation QDQ (Asym + CW, fake=True)",   "fake": True,  "dqq": False, "zp": True,  "cw": True,  "ssa": False},
    {"name": "B. INT8 Natif Sym + CW (zp=False, cw=True)", "fake": False, "dqq": True,  "zp": False, "cw": True,  "ssa": False},
    {"name": "C. INT8 Natif Sym + Layer (zp=False, cw=False)","fake": False, "dqq": True, "zp": False, "cw": False, "ssa": False},
    {"name": "D. INT8 Natif Asym + CW (zp=True, cw=True)", "fake": False, "dqq": True,  "zp": True,  "cw": True,  "ssa": False},
]

print("\n" + f"{'Test isolation PTQ':<48} | {'Précision':<13} | {'Accord FP32'}")
print("-" * 78)

for cfg in tests:
    try:
        g = build_graph(str(paths.CLASSIFIER_ONNX))
        calib = []
        for path, _ in val_samples[::2]:
            t = aidge_core.Tensor(preprocess(path))
            t.set_backend("cpu")
            calib.append(t)

        ptq = aidge_quantization.PTQ(g)
        ptq.weight_precision = aidge_core.dtype.int8
        ptq.activation_precision = aidge_core.dtype.int8
        ptq.accumulation_precision = aidge_core.dtype.int32
        ptq.fake_quantize = cfg["fake"]
        ptq.fuse_dqq = cfg["dqq"]
        ptq.use_zero_point = cfg["zp"]
        ptq.channel_wise = cfg["cw"]
        ptq.single_shift_approximation = cfg["ssa"]
        ptq.verbose = False

        q_graph = ptq.process(calib)
        q_graph.set_backend("cpu")
        q_graph.forward_dims([[1, 3, 128, 128]])

        sched = aidge_core.SequentialScheduler(q_graph)
        out_node = list(q_graph.get_output_nodes())[0]

        correct, agree = 0, 0
        sample_logits = []
        for idx, (path, label) in enumerate(val_samples):
            t = aidge_core.Tensor(preprocess(path))
            t.set_backend("cpu")
            sched.forward(data=[t])
            logits = np.array(out_node.get_operator().get_output(0))[0]
            if idx < 3:
                sample_logits.append(np.round(logits, 3))
            pred = int(np.argmax(logits))
            correct += int(pred == label)
            agree += int(pred == fp32_preds[idx])

        acc = 100.0 * correct / len(val_samples)
        agr = 100.0 * agree / len(val_samples)
        print(f"{cfg['name']:<48} | {acc:>5.1f}% ({correct:02d}/80) | {agr:>6.1f}%")
        print(f"   -> Logits 3 premières images : {sample_logits[0]} | {sample_logits[1]} | {sample_logits[2]}")

        if cfg["name"].startswith("B."):
            diag = ptq.get_diagnostic()
            diag_attrs = [a for a in dir(diag) if not a.startswith("_")]
            print("   -> PTQDiagnostic attrs :", diag_attrs)
    except Exception as e:
        print(f"{cfg['name']:<48} | Erreur : {e}")
