import os
import onnx
from huggingface_hub import HfApi, hf_hub_download
from ultralytics import YOLO
import aidge_onnx

# 1. Recherche automatique du dépôt officiel MDV6 sur Hugging Face
api = HfApi()
pt_path = None

for model_info in api.list_models(search="MDV6"):
    try:
        files = api.list_repo_files(model_info.id)
        matches = [f for f in files if "yolov10" in f.lower() and f.endswith(".pt")]
        if matches:
            # Priorité à la variante 'c' (compact)
            target = next((f for f in matches if "-c" in f.lower()), matches[0])
            print(f"Dépôt trouvé : {model_info.id} -> téléchargement de {target}...")
            pt_path = hf_hub_download(repo_id=model_info.id, filename=target, local_dir=".")
            break
    except Exception:
        continue

if not pt_path:
    print("Dépôt MDV6 non trouvé via API, repli sur yolov10s.pt (même architecture YOLOv10)")
    pt_path = "yolov10s.pt"

# 2. Export ONNX statique (batch=1, 320x320)
model = YOLO(pt_path)
onnx_path = model.export(format="onnx", imgsz=320, opset=18, simplify=True, dynamic=False)

# 3. Audit de compatibilité et d'empreinte mémoire dans Aidge
graph = aidge_onnx.load_onnx(onnx_path)
nodes = graph.get_nodes()
generic_nodes = [n for n in nodes if type(n.get_operator()).__name__ == "GenericOperatorOp"]

onnx_model = onnx.load(onnx_path)
total_params = sum(
    onnx.numpy_helper.to_array(init).size
    for init in onnx_model.graph.initializer
)

print(f"\n--- Audit Aidge : {pt_path} ---")
print(f"Nombre total de noeuds : {len(nodes)}")
print(f"Operateurs uniques : {sorted(set(n.type() for n in nodes))}")
print(f"Operateurs non natifs (Generic) : {len(generic_nodes)}")
if generic_nodes:
    print("Detail des operateurs Generic :", sorted(set(n.type() for n in generic_nodes)))
print(f"Parametres totaux : {total_params:,}")
print(f"Taille poids FP32 : {total_params * 4 / (1024**2):.2f} Mo")
print(f"Taille poids INT8 (estimee) : {total_params / (1024**2):.2f} Mo")
