# Chemins, images de calibration et appel de Vela communs aux scripts d'évaluation. S'exécute uniquement sur l'ordinateur, rien n'est écrit sur la carte.
import csv
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

from mountain import paths

ROOT = paths.ROOT
OUT_DIR = paths.U55_PROFILE
MLEK = paths.MLEK
VELA = MLEK / "resources_downloaded" / "env" / "bin" / "vela"
# Cibles Vela : hp = M55-HP + U55-256 à 400 MHz (mêmes paramètres que H256 dans set_up_default_resources.py du kit),
# he = M55-HE + U55-128 à 160 MHz. Dans les deux cas, poids en MRAM et activations en SRAM.
TARGETS = {
    "hp": ("ethos-u55-256", "RTSS_HP_SRAM_MRAM"),
    "he": ("ethos-u55-128", "RTSS_HE_SRAM_MRAM"),
}


def vela_args(target):
    accelerator, system_config = TARGETS[target]
    return [
        f"--accelerator-config={accelerator}",
        "--optimise", "Performance",
        "--config", str(paths.VELA_INI),
        "--memory-mode=Shared_Sram",
        f"--system-config={system_config}",
        "--arena-cache-size=2097152",
        "--verbose-performance",
        "--show-cpu-operations",
    ]


def calibration_images():
    """Les 8 images de coco8 (dont girafes, zèbres, éléphants), RGB uint8."""
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset("coco8.yaml")
    files = sorted(Path(data["path"]).rglob("*.jpg"))
    return [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files]


def letterbox(img, size, pad=114):
    """Redimensionne en gardant les proportions puis complète en carré centré, comme le prétraitement d'Ultralytics."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    out = np.full((size, size, 3), pad, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out


def run_vela(tflite, target="hp"):
    """Compile le modèle int8 avec Vela et renvoie le temps d'inférence, l'occupation SRAM/MRAM et le nombre d'opérateurs
    renvoyés au CPU ; le journal et le rapport par couche restent dans le dossier vela/ (vela_he/ pour le HE)."""
    tflite = Path(tflite)
    out = tflite.parent / ("vela" if target == "hp" else f"vela_{target}")
    out.mkdir(exist_ok=True)
    p = subprocess.run([str(VELA), str(tflite), *vela_args(target), f"--output-dir={out}"], capture_output=True, text=True)
    log = out / f"{tflite.stem}_vela.log"
    log.write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"Échec de Vela : {log}")
    with open(out / f"{tflite.stem}_summary_{TARGETS[target][1]}.csv") as f:
        row = next(csv.DictReader(f))
    cpu = re.search(r"CPU operators = (\d+)", p.stdout)
    npu = re.search(r"NPU operators = (\d+)", p.stdout)
    return {
        "ms": round(float(row["inference_time"]) * 1e3, 2),
        "sram_kib": round(float(row["sram_memory_used"]), 1),
        "mram_kib": round(float(row["off_chip_flash_memory_used"]), 1),
        "gmac": round(float(row["nn_macs"]) / 1e9, 3),
        "npu_ops": int(npu.group(1)) if npu else None,
        "cpu_ops": int(cpu.group(1)) if cpu else None,
    }
