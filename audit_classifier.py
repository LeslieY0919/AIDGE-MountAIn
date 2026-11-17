import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import onnx
import aidge_core
import aidge_onnx

NUM_CLASSES = 5
model = mobilenet_v2(weights=None, width_mult=0.5)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.eval()

onnx_path = "species_classifier.onnx"
dummy = torch.randn(1, 3, 128, 128)
torch.onnx.export(model, dummy, onnx_path, input_names=["input"], output_names=["logits"], opset_version=18, dynamo=False)

graph = aidge_onnx.load_onnx(onnx_path)
graph.forward_dims([[1, 3, 128, 128]])

nodes = graph.get_nodes()
generic_ops = [n.type() for n in nodes if type(n.get_operator()).__name__ == "GenericOperatorOp"]

onnx_model = onnx.load(onnx_path)
params = sum(onnx.numpy_helper.to_array(init).size for init in onnx_model.graph.initializer)

print(f"Noeuds classifieur Aidge : {len(nodes)} | Generic : {len(generic_ops)}")
print(f"Parametres classifieur : {params:,} | Taille INT8 : {params / (1024**2):.2f} Mo")
print(f"Empreinte INT8 cumulee (MDV6 + Classifieur) : {2.17 + params / (1024**2):.2f} Mo")
