import torch
from ultralytics import YOLO
import aidge_onnx

# On teste d'abord l'architecture backbone/head sur un détecteur nano (même structure que MegaDetector v6/v5)
model = YOLO("yolov8n.pt")
onnx_path = model.export(format="onnx", imgsz=320, opset=18, simplify=True, dynamic=False)

# Chargement et audit dans Aidge
graph = aidge_onnx.load_onnx(onnx_path)
nodes = graph.get_nodes()
generic_ops = [n.type() for n in nodes if "Generic" in n.type()]

print(f"Nombre total de noeuds Aidge : {len(nodes)}")
print(f"Types d'operateurs uniques : {sorted(set(n.type() for n in nodes))}")
print(f"Operateurs non natifs (Generic) : {len(generic_ops)}")
