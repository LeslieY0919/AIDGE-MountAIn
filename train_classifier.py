import os
import json
import random
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import aidge_onnx

# 1. Espèces cibles (Nom -> Nom scientifique résolu automatiquement via iNaturalist)
SPECIES = {
    "renard": "Vulpes vulpes",
    "sanglier": "Sus scrofa",
    "chevreuil": "Capreolus capreolus",
    "ecureuil": "Sciurus vulgaris",
    "blaireau": "Meles meles",
}
IMAGES_PER_CLASS = 80
DATA_DIR = "dataset_wildlife"

def download_one(args):
    url, dst = args
    if os.path.exists(dst):
        return
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(dst, "wb") as f:
                f.write(r.content)
            Image.open(dst).verify()
    except Exception:
        if os.path.exists(dst):
            os.remove(dst)

print("1/3 - Téléchargement des images depuis iNaturalist...")
for name, sci_name in SPECIES.items():
    train_dir = os.path.join(DATA_DIR, "train", name)
    val_dir = os.path.join(DATA_DIR, "val", name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    existing = len(os.listdir(train_dir)) + len(os.listdir(val_dir))
    if existing >= int(IMAGES_PER_CLASS * 0.8):
        print(f" -> {name} : {existing} images déjà en cache")
        continue

    taxa = requests.get(
        "https://api.inaturalist.org/v1/taxa",
        params={"q": sci_name, "rank": "species"},
        timeout=15,
    ).json().get("results", [])
    if not taxa:
        print(f" -> {name} ({sci_name}) : taxon introuvable, classe ignorée")
        os.rmdir(train_dir)
        os.rmdir(val_dir)
        continue

    taxon_id = taxa[0]["id"]
    obs = requests.get(
        "https://api.inaturalist.org/v1/observations",
        params={
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "photos": "true",
            "per_page": IMAGES_PER_CLASS,
        },
        timeout=15,
    ).json().get("results", [])

    urls = [o["photos"][0]["url"].replace("square", "medium") for o in obs if o.get("photos")]
    random.seed(42)
    random.shuffle(urls)
    split_idx = int(0.8 * len(urls))
    tasks = [(u, os.path.join(train_dir, f"{i}.jpg")) for i, u in enumerate(urls[:split_idx])]
    tasks += [(u, os.path.join(val_dir, f"{i}.jpg")) for i, u in enumerate(urls[split_idx:])]

    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(download_one, tasks))
    print(f" -> {name} (taxon_id={taxon_id}) : {len(urls)} images récupérées")

# 2. Préparation PyTorch & Entraînement MobileNetV2 x0.5
print("\n2/3 - Entraînement du classifieur MobileNetV2 (x0.5)...")
train_tf = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.RandomCrop((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tf)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

with open("class_names.json", "w") as f:
    json.dump(train_ds.classes, f)

model = timm.create_model("mobilenetv2_050", pretrained=True, num_classes=len(train_ds.classes))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
for epoch in range(1, 9):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / max(1, total)
    print(f"Epoch {epoch}/8 - Précision validation : {acc:.1f}%")
    if acc >= best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_classifier.pth")

# 3. Export ONNX et audit Aidge
print("\n3/3 - Export ONNX et vérification dans Aidge...")
model.load_state_dict(torch.load("best_classifier.pth"))
model.eval()
dummy = torch.randn(1, 3, 128, 128)
torch.onnx.export(model, dummy, "species_classifier.onnx", input_names=["input"], output_names=["logits"], opset_version=18, dynamo=False)

graph = aidge_onnx.load_onnx("species_classifier.onnx")
graph.forward_dims([[1, 3, 128, 128]])
generic_ops = [n.type() for n in graph.get_nodes() if type(n.get_operator()).__name__ == "GenericOperatorOp"]
print(f"Export terminé | Meilleure précision : {best_acc:.1f}% | Opérateurs Generic Aidge : {len(generic_ops)}")
