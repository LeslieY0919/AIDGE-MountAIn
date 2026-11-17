from pathlib import Path
import onnx
import torch.nn as nn
from ultralytics import YOLO

# --- PARAMETRES A ADAPTER ---
MDV6_WEIGHTS = "MDV6-yolov10-c.pt"     # Chemin vers les poids MegaDetector v6 (YOLOv10-c)
DATA_YAML = "[CHEMIN_VERS_DATA_YAML]"  # ex: "dataset/data.yaml"
IMGSZ = 320                            # Doit correspondre exactement au in_shape dans Aidge (ex: 320 ou 640)
EPOCHS = 100
BATCH_SIZE = 16                        # Baisser a 8 si manque de VRAM GPU
PROJECT_NAME = "wildlife_mdv6_yolov10c"
# ----------------------------

def disable_psa_attention(trainer):
    """Remplace uniquement model.10.attn par Identity tout en gardant les poids MDV6."""
    layer_10 = trainer.model.model[10]
    if hasattr(layer_10, "attn"):
        layer_10.attn = nn.Identity()
        print("[OK] model.10.attn neutralise pour compatibilite Aidge C++.")

def main():
    # 1. Chargement du checkpoint pre-entraine MDV6-yolov10-c
    model = YOLO(MDV6_WEIGHTS)
    model.add_callback("on_pretrain_routine_end", disable_psa_attention)

    # 2. Fine-tuning sur votre dataset
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        project="runs/detect",
        name=PROJECT_NAME,
        device=0,  # 0 pour GPU CUDA, "cpu" sinon
    )

    # 3. Chargement des meilleurs poids et export ONNX statique
    best_pt = Path("runs/detect") / PROJECT_NAME / "weights" / "best.pt"
    best_model = YOLO(str(best_pt))

    onnx_path = best_model.export(
        format="onnx",
        imgsz=IMGSZ,
        opset=12,
        simplify=True,
        dynamic=False,  # Obligatoire pour Aidge : dimensions fixes [1, 3, H, W]
        half=False,
    )

    # 4. Verification automatique avant de te renvoyer le fichier
    onnx_model = onnx.load(str(onnx_path))
    matmul_nodes = [n.name for n in onnx_model.graph.node if n.op_type == "MatMul"]
    assert len(matmul_nodes) == 0, f"Erreur : MatMul encore present ({matmul_nodes})"

    print(f"\n[SUCCES] Export pret pour Aidge (0 MatMul) : {onnx_path}")

if __name__ == "__main__":
    main()
