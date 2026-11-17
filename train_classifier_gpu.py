import inspect
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# --- Configuration optimisee RTX 5070 Ti (16 Go VRAM) + Aidge C++ ---
DATA_DIR = Path("dataset_wildlife")
CLASS_NAMES_PATH = Path("class_names.json")
ONNX_OUT = Path("species_classifier.onnx")
BEST_PT_OUT = Path("best_species_classifier.pt")

IMG_SIZE = 128          # Taille fixe attendue par export_cpp_classifier_int8
BATCH_SIZE = 128        # Exploite les 16 Go de VRAM et stabilise BatchNorm
NUM_WORKERS = 4
EPOCHS_HEAD = 6         # Phase 1 : chauffe de la tete de classification
EPOCHS_FULL = 44        # Phase 2 : fine-tuning complet du ResNet-34
LR_HEAD = 1.5e-3
LR_BACKBONE = 3e-4
WEIGHT_DECAY = 1e-2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_resnet34(num_classes: int) -> nn.Module:
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(in_features, num_classes),
    )
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_names: list[str]):
    model.eval()
    n = len(class_names)
    conf_mat = np.zeros((n, n), dtype=np.int64)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model(x).argmax(dim=1).cpu().numpy()
        for t, p in zip(y.numpy(), preds):
            conf_mat[int(t), int(p)] += 1

    total = int(conf_mat.sum())
    correct = int(np.trace(conf_mat))
    acc = 100.0 * correct / max(1, total)

    per_class = []
    for i, cname in enumerate(class_names):
        row_sum = int(conf_mat[i].sum())
        c_acc = 100.0 * conf_mat[i, i] / max(1, row_sum)
        per_class.append(f"{cname}: {c_acc:.0f}% ({conf_mat[i, i]}/{row_sum})")

    return acc, " | ".join(per_class), conf_mat


def print_confusion_matrix(conf_mat: np.ndarray, class_names: list[str]):
    print("\n=== Matrice de confusion (Lignes = Vrai | Colonnes = Predit) ===")
    header = "Vrai \\ Pred | " + " | ".join(f"{c[:8]:>8}" for c in class_names)
    print(header)
    print("-" * len(header))
    for i, cname in enumerate(class_names):
        row_vals = " | ".join(f"{conf_mat[i, j]:>8d}" for j in range(len(class_names)))
        print(f"{cname[:11]:<11} | {row_vals}")


def export_verified_onnx(model: nn.Module, onnx_path: Path):
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)

    export_kwargs = dict(
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(model_cpu, dummy, str(onnx_path), **export_kwargs)

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": dummy.numpy()})[0]
    with torch.no_grad():
        pt_out = model_cpu(dummy).numpy()

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"\nExport ONNX valide -> {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} Mo)")
    print(f"Parite numerique PyTorch vs ONNX Runtime : ecart max = {max_diff:.2e}")
    assert max_diff < 1e-4, f"Ecart ONNX anormalement eleve : {max_diff}"


def main():
    assert torch.cuda.is_available(), (
        "GPU CUDA non detecte. Verifie PyTorch cu128 : "
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
    )
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU actif : {gpu_name} ({vram_gb:.1f} Go VRAM)")

    class_names = json.loads(CLASS_NAMES_PATH.read_text())
    num_classes = len(class_names)
    print(f"Classes cibles ({num_classes}) : {class_names}")

    # Augmentations adaptees aux pieges photographiques (cadrage, lumiere jour/nuit, branches)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0), ratio=(0.80, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.04),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_tf)
    assert train_ds.classes == class_names, f"Ordre des dossiers incoherent : {train_ds.classes} vs {class_names}"

    # Evite un dernier batch de taille 1 qui ferait planter BatchNorm2d
    drop_last = (len(train_ds) > BATCH_SIZE) and (len(train_ds) % BATCH_SIZE == 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    # Poids par classe pour compenser tout desequilibre d'effectifs
    counts = np.bincount(train_ds.targets, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"Images train={len(train_ds)} | val={len(val_ds)} | effectifs/classe={counts.astype(int).tolist()}")

    model = build_resnet34(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    t0 = time.time()

    # --- Phase 1 : Entrainement de la tete uniquement ---
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    for _ in range(EPOCHS_HEAD):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    # --- Phase 2 : Fine-tuning complet du ResNet-34 ---
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_BACKBONE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FULL, eta_min=1e-6)

    best_acc = 0.0
    best_state = None
    best_cm = None

    for epoch in range(1, EPOCHS_FULL + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            seen += y.size(0)

        scheduler.step()

        val_acc, details, cm = evaluate(model, val_loader, device, class_names)
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_cm = cm

        if epoch % 5 == 0 or epoch == EPOCHS_FULL:
            print(f"[Epoch {epoch:02d}/{EPOCHS_FULL}] loss={running_loss/max(1, seen):.4f} | val_acc={val_acc:.2f}% (best={best_acc:.2f}%) | {details}")

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nEntrainement termine en {time.time() - t0:.1f}s | Pic VRAM utilise : {peak_vram:.2f} / {vram_gb:.1f} Go")
    print(f"Meilleure precision validation : {best_acc:.2f}%")

    model.load_state_dict(best_state)
    torch.save(best_state, BEST_PT_OUT)
    print_confusion_matrix(best_cm, class_names)
    export_verified_onnx(model, ONNX_OUT)


if __name__ == "__main__":
    main()
