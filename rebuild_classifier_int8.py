import os
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

import aidge_core
import aidge_backend_cpu
import aidge_quantization
import aidge_export_cpp
import importlib
export_mod = importlib.import_module("aidge_core.export_utils.export")

from quantize_pipeline_int8 import (
    MEAN,
    STD,
    CLS_NORM_DIV,
    apply_aidge_export_fixes,
    build_classifier_graph,
    make_tensor,
    patch_exported_headers,
    write_and_build_classifier,
)

apply_aidge_export_fixes()

with open("class_names.json") as f:
    class_names = json.load(f)


def load_cls_img(path):
    im = Image.open(path).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    arr = ((np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD) / CLS_NORM_DIV
    return make_tensor(np.transpose(arr, (2, 0, 1))[None, ...])


calib_cls = []
for cname in class_names:
    d = os.path.join("dataset_wildlife", "train", cname)
    for fn in sorted(os.listdir(d))[:10]:
        calib_cls.append(load_cls_img(os.path.join(d, fn)))

val_cls = []
for cid, cname in enumerate(class_names):
    d = os.path.join("dataset_wildlife", "val", cname)
    for fn in sorted(os.listdir(d)):
        val_cls.append((load_cls_img(os.path.join(d, fn)), cid))

# 1. Reference FP32 pour calibrer le scale exact de sortie
g_fp32 = build_classifier_graph()
s_fp32 = aidge_core.SequentialScheduler(g_fp32)
o_fp32 = list(g_fp32.get_output_nodes())[0]
fp32_logits = []
for t, _ in val_cls:
    s_fp32.forward(data=[t])
    fp32_logits.append(np.array(o_fp32.get_operator().get_output(0), dtype=np.float64)[0])

# 2. PTQ haute precision (channel_wise=True, single_shift_approximation=False)
ptq = aidge_quantization.PTQ(
    build_classifier_graph(),
    activation_precision=aidge_core.dtype.int8,
    accumulation_precision=aidge_core.dtype.int32,
    inference_precision=aidge_core.dtype.int8,
)
ptq.single_shift_approximation = False
ptq.use_zero_point = False
ptq.channel_wise = True
ptq.merge_passthrough_quantizers = True
q_cls = ptq.process([t.clone() for t in calib_cls])

export_mod.remove_input_quantizers(q_cls)
q_cls.set_backend("cpu")
q_cls.forward_dims([[1, 3, 128, 128]])

q_val_inputs = export_mod.quantize_inputs([t.clone() for t, _ in val_cls], "int8")
s_q = aidge_core.SequentialScheduler(q_cls)
o_q = list(q_cls.get_output_nodes())[0]

q_logits = []
corr = 0
for q_in, (_, y) in zip(q_val_inputs, val_cls):
    s_q.forward(data=[q_in])
    out_arr = np.array(o_q.get_operator().get_output(0), dtype=np.float64)[0]
    q_logits.append(out_arr)
    corr += int(int(np.argmax(out_arr)) == y)

f_mat = np.stack(fp32_logits)
q_mat = np.stack(q_logits)
out_scale = float(np.sum(f_mat * q_mat) / np.sum(q_mat ** 2))
print(f"Validation Aidge INT8 (channel_wise=True) : {100.0 * corr / len(val_cls):.2f}% ({corr}/{len(val_cls)}) | scale={out_scale:.6f}")

# 3. Export C++ sans re-quantifier (model_prequantized=True)
OUT_DIR = "export_cpp_classifier_int8"
shutil.rmtree(OUT_DIR, ignore_errors=True)
cfg = aidge_export_cpp.ExportConfig(
    export_folder=OUT_DIR,
    dtype="int8",
    cuda=False,
    compile=False,
    model_prequantized=True,
)
aidge_core.export_utils.export(graph=q_cls, config=cfg, input_tensors=q_val_inputs[:1])
patch_exported_headers(OUT_DIR)
write_and_build_classifier(OUT_DIR, out_scale)
print("Compilation Release C++ INT8 du classifieur terminee.")
