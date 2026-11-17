import os
import re
import sysconfig
import requests
import onnx
from ultralytics import YOLO
import aidge_onnx

pt_path = "MDV6-yolov10-c.pt"

if not os.path.exists(pt_path) or os.path.getsize(pt_path) < 5_000_000:
    pkg_dir = os.path.join(sysconfig.get_paths()["purelib"], "PytorchWildlife")
    url = None
    for root, _, files in os.walk(pkg_dir):
        for fname in files:
            if fname.endswith(".py"):
                with open(os.path.join(root, fname), "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                direct = re.findall(r'https?://[^\s"\']+MDV6-yolov10-?c\.pt[^\s"\']*', content, re.IGNORECASE)
                if direct:
                    url = direct[0]
                    break
                for candidate in ["MDV6-yolov10-c.pt", "MDV6-yolov10c.pt"]:
                    if candidate in content:
                        bases = re.findall(r'https?://zenodo\.org/records/\d+/files/?', content)
                        if bases:
                            url = bases[0].rstrip("/") + "/" + candidate
                            break
        if url:
            break

    if not url:
        print("URL directe non matchée, liens trouvés dans le paquet :")
        for root, _, files in os.walk(pkg_dir):
            for fname in files:
                if fname.endswith(".py"):
                    with open(os.path.join(root, fname), "r", encoding="utf-8", errors="ignore") as f:
                        for match in re.findall(r'https?://[^\s"\']+', f.read()):
                            if any(k in match.lower() for k in ["zenodo", "mdv6", "yolov10"]):
                                print(" -", match)
        raise SystemExit(1)

    print(f"Téléchargement officiel depuis : {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(pt_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)

size_mb = os.path.getsize(pt_path) / (1024**2)
print(f"Fichier téléchargé : {pt_path} ({size_mb:.2f} Mo)")

# Export ONNX statique (batch=1, 320x320)
model = YOLO(pt_path)
onnx_path = model.export(format="onnx", imgsz=320, opset=18, simplify=True, dynamic=False)

# Audit Aidge + propagation des dimensions
graph = aidge_onnx.load_onnx(onnx_path)
graph.forward_dims([[1, 3, 320, 320]])

nodes = graph.get_nodes()
generic_nodes = [n for n in nodes if type(n.get_operator()).__name__ == "GenericOperatorOp"]

onnx_model = onnx.load(onnx_path)
total_params = sum(onnx.numpy_helper.to_array(init).size for init in onnx_model.graph.initializer)

print(f"\n--- Validation finale MDV6 dans Aidge ---")
print(f"Fichier : {pt_path} | Classes : {model.names}")
print(f"Noeuds Aidge : {len(nodes)} | Operateurs Generic : {len(generic_nodes)}")
print(f"Parametres : {total_params:,} | Poids INT8 : {total_params / (1024**2):.2f} Mo")
