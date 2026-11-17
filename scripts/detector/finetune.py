import shutil
import urllib.request
import zipfile
from pathlib import Path

import onnx
import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

from mountain import paths

# ==============================================================================
# CONFIGURATION (deja calibree pour RTX 5070 Ti 16 Go + cible >= 85% precision)
# ==============================================================================
MDV6_WEIGHTS = str(paths.DETECTOR_PT)
FINAL_ONNX_NAME = str(paths.DETECTOR_ONNX_TRAINED)

# Optionnel : mets un lien direct .zip (Roboflow/Drive) si vous utilisez votre propre base.
# Laisse vide ("") pour telecharger automatiquement le dataset faune Ultralytics.
CUSTOM_DATASET_ZIP_URL = ""

IMGSZ = 640             # Resolution fixe (doit correspondre a Aidge : [1, 3, 640, 640])
EPOCHS = 150            # ~5-10 min sur RTX 5070 Ti (arret auto des que le plateau est atteint)
BATCH_SIZE = 32         # Exploite les 16 Go de VRAM en 640x640
TARGET_PRECISION = 0.85 # Objectif de 85% de precision (P) et mAP@50
PROJECT_NAME = "wildlife_mdv6_aidge"
# ==============================================================================


def prepare_dataset() -> Path:
    """Telecharge et prepare automatiquement le dataset sans intervention manuelle."""
    work_dir = Path("dataset_auto_ready").resolve()
    yaml_out = work_dir / "data.yaml"

    if yaml_out.exists():
        print(f"[INFO] Dataset deja pret : {yaml_out}")
        return yaml_out

    if CUSTOM_DATASET_ZIP_URL:
        print(f"[INFO] Telechargement du dataset personnalise depuis {CUSTOM_DATASET_ZIP_URL}...")
        zip_path = Path("custom_dataset.zip")
        urllib.request.urlretrieve(CUSTOM_DATASET_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(work_dir)
        zip_path.unlink(missing_ok=True)
        found_yamls = list(work_dir.rglob("data.yaml"))
        if not found_yamls:
            raise FileNotFoundError("Aucun fichier data.yaml trouve dans l'archive ZIP.")
        return found_yamls[0]

    # Telechargement automatique du dataset faune officiel Ultralytics (~1500 images)
    print("[INFO] Telechargement automatique du dataset 'african-wildlife'...")
    info = check_det_dataset("african-wildlife.yaml")
    src_root = Path(info["path"])

    # Remapping des especes (0..3) vers la classe 0 ('animal') de MegaDetector v6
    # afin de conserver intacte la tete de classification (animal, person, vehicle).
    for split in ["train", "val"]:
        src_img_dir = src_root / "images" / split
        src_lbl_dir = src_root / "labels" / split
        dst_img_dir = work_dir / "images" / split
        dst_lbl_dir = work_dir / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_file in src_img_dir.iterdir():
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            shutil.copy2(img_file, dst_img_dir / img_file.name)

            lbl_file = src_lbl_dir / f"{img_file.stem}.txt"
            out_lines = []
            if lbl_file.exists():
                for line in lbl_file.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = "0"  # Toutes les especes animales -> classe 0 (animal)
                        out_lines.append(" ".join(parts))
            (dst_lbl_dir / f"{img_file.stem}.txt").write_text("\n".join(out_lines))

    data_cfg = {
        "path": str(work_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "animal", 1: "person", 2: "vehicle"},
    }
    yaml_out.write_text(yaml.safe_dump(data_cfg, sort_keys=False))
    print(f"[OK] Dataset faune prepare automatiquement : {yaml_out}")
    return yaml_out


def disable_psa_attention(trainer):
    """Neutralise uniquement model.10.attn (MatMul) pour compatibilite Aidge C++."""
    layer_10 = trainer.model.model[10]
    if hasattr(layer_10, "attn"):
        layer_10.attn = nn.Identity()
        print("[OK] model.10.attn remplace par Identity (compatibilite Aidge C++).")


def log_target_status(trainer):
    """Affiche l'avancement vers l'objectif de 85% de precision a chaque epoque."""
    metrics = getattr(trainer, "metrics", {}) or {}
    prec = metrics.get("metrics/precision(B)", 0.0)
    map50 = metrics.get("metrics/mAP50(B)", 0.0)
    if prec >= TARGET_PRECISION and map50 >= TARGET_PRECISION:
        print(
            f"[OBJECTIF ATTEINT] Precision={prec*100:.1f}% | mAP@50={map50*100:.1f}% "
            f"(>= {TARGET_PRECISION*100:.0f}%) - poursuite jusqu'a convergence optimale..."
        )


def main():
    if not Path(MDV6_WEIGHTS).exists():
        raise FileNotFoundError(
            f"Fichier '{MDV6_WEIGHTS}' introuvable. Place-le dans le meme dossier que ce script."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU NVIDIA CUDA non detecte. Verifie l'installation de PyTorch avec support CUDA."
        )

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[INFO] GPU detecte : {gpu_name} ({vram_gb:.1f} Go VRAM)")

    # 1. Telechargement et preparation automatique des donnees
    data_yaml = prepare_dataset()

    # 2. Chargement de MegaDetector v6 + neutralisation du bloc MatMul
    model = YOLO(MDV6_WEIGHTS)
    model.add_callback("on_pretrain_routine_end", disable_psa_attention)
    model.add_callback("on_fit_epoch_end", log_target_status)

    # 3. Entrainement calibre pour atteindre >= 85% de precision sans sur-apprentissage
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=0,
        project="runs/detect",
        name=PROJECT_NAME,
        exist_ok=True,
        # Performances RTX 5070 Ti (16 Go)
        amp=True,
        cache="disk",
        workers=8,
        # Fine-tuning doux pour preserver les connaissances MDV6 et depasser 85%
        lr0=0.0015,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5.0,
        patience=35,
        close_mosaic=20,
        # Augmentations riches
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    final_prec = results.results_dict.get("metrics/precision(B)", 0.0)
    final_map50 = results.results_dict.get("metrics/mAP50(B)", 0.0)
    print(f"\n[BILAN] Meilleur modele -> Precision : {final_prec*100:.1f}% | mAP@50 : {final_map50*100:.1f}%")

    # 4. Export ONNX statique opset 17 compatible Aidge C++
    best_pt = Path("runs/detect") / PROJECT_NAME / "weights" / "best.pt"
    best_model = YOLO(str(best_pt))

    exported_onnx = best_model.export(
        format="onnx",
        imgsz=IMGSZ,
        opset=17,
        simplify=True,
        dynamic=False,
    )

    final_onnx = Path(FINAL_ONNX_NAME).resolve()
    shutil.copy2(exported_onnx, final_onnx)

    # 5. Verification stricte avant envoi
    onnx_model = onnx.load(str(final_onnx))
    matmuls = [n.name for n in onnx_model.graph.node if n.op_type == "MatMul"]
    assert len(matmuls) == 0, f"Erreur critique : MatMul encore present ({matmuls})"

    print("\n" + "=" * 65)
    print(f"[SUCCES] Fichier pret a renvoyer a Matteo : {final_onnx}")
    print("=" * 65)


if __name__ == "__main__":
    main()
