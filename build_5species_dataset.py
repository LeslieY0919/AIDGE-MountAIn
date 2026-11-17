import io
import json
import random
import shutil
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

CLASS_NAMES_PATH = Path("class_names.json")
DETECTOR_ONNX_PATH = Path("MDV6-yolov10-c_cut_folded.onnx")
DATASET_DIR = Path("dataset_wildlife")
BACKUP_DIR = Path("dataset_wildlife_backup")

# Uniquement tes 5 especes de base
SPECIES_SCIENTIFIC = {
    "blaireau": "Meles meles",
    "chevreuil": "Capreolus capreolus",
    "ecureuil": "Sciurus vulgaris",
    "renard": "Vulpes vulpes",
    "sanglier": "Sus scrofa",
}

TARGET_PER_CLASS = 650      # ~650 images recadrees par espece (~3250 au total)
MAX_TOTAL_MINUTES = 20.0    # ~4 minutes par espece
VAL_RATIO = 0.15            # 85% train / 15% val
CONF_THRESH = 0.45          # Confiance minimale YOLOv10
MIN_BOX_PX = 32
CROP_MARGIN = 0.10
WORKERS = 4
USER_AGENT = "AidgeWildlife5Species/1.0"


def normalize_key(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def http_get_json(url: str, retries: int = 3) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    return {}


def resolve_taxon_id(class_name: str) -> tuple[int, str]:
    key = normalize_key(class_name)
    if key not in SPECIES_SCIENTIFIC:
        raise ValueError(f"Espece non attendue : '{class_name}'. Attendu parmi {list(SPECIES_SCIENTIFIC.keys())}")
    sci_name = SPECIES_SCIENTIFIC[key]
    q_enc = urllib.parse.quote(sci_name)
    data = http_get_json(f"https://api.inaturalist.org/v1/taxa?q={q_enc}&locale=fr&per_page=1")
    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"Taxon introuvable sur iNaturalist pour '{sci_name}'")
    return int(results[0]["id"]), sci_name


def compute_dhash(img: Image.Image) -> bytes:
    small = img.convert("L").resize((17, 16), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.int16)
    return np.packbits(arr[:, 1:] > arr[:, :-1]).tobytes()


class YoloAnimalCropper:
    def __init__(self, onnx_path: Path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

    def crop_animal(self, img: Image.Image) -> Image.Image | None:
        W, H = img.size
        if W < 64 or H < 64:
            return None

        resized = img.resize((640, 640), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        in_nchw = np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...], dtype=np.float32)

        outs = self.sess.run(None, {self.in_name: in_nchw})
        boxes_arr = next(o for o in outs if o.shape == (1, 64, 8400))[0]
        scores_arr = next(o for o in outs if o.shape == (1, 3, 8400))[0]

        animal_scores = 1.0 / (1.0 + np.exp(-np.clip(scores_arr[0], -60.0, 60.0)))
        valid_idxs = np.where(animal_scores >= CONF_THRESH)[0]
        if len(valid_idxs) == 0:
            return None

        i = int(valid_idxs[np.argmax(animal_scores[valid_idxs])])
        dist = []
        for side in range(4):
            logits = boxes_arr[side * 16 : (side + 1) * 16, i]
            e = np.exp(logits - np.max(logits))
            dist.append(float(np.sum((e / np.sum(e)) * np.arange(16))))

        if i < 6400:
            gx, gy, stride = (i % 80) + 0.5, (i // 80) + 0.5, 8.0
        elif i < 8000:
            idx = i - 6400
            gx, gy, stride = (idx % 40) + 0.5, (idx // 40) + 0.5, 16.0
        else:
            idx = i - 8000
            gx, gy, stride = (idx % 20) + 0.5, (idx // 20) + 0.5, 32.0

        x1 = np.clip((gx - dist[0]) * stride, 0, 640) * W / 640.0
        y1 = np.clip((gy - dist[1]) * stride, 0, 640) * H / 640.0
        x2 = np.clip((gx + dist[2]) * stride, 0, 640) * W / 640.0
        y2 = np.clip((gy + dist[3]) * stride, 0, 640) * H / 640.0

        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
            return None

        cx1 = max(0, int(round(x1 - bw * CROP_MARGIN)))
        cy1 = max(0, int(round(y1 - bh * CROP_MARGIN)))
        cx2 = min(W, int(round(x2 + bw * CROP_MARGIN)))
        cy2 = min(H, int(round(y2 + bh * CROP_MARGIN)))
        if (cx2 - cx1) < MIN_BOX_PX or (cy2 - cy1) < MIN_BOX_PX:
            return None

        return img.crop((cx1, cy1, cx2, cy2))


def download_image(url: str) -> Image.Image | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    except Exception:
        return None


def main():
    t_start = time.time()
    max_seconds = MAX_TOTAL_MINUTES * 60.0

    class_names = json.loads(CLASS_NAMES_PATH.read_text())
    assert len(class_names) == 5, f"Attendu 5 classes dans class_names.json, trouve : {class_names}"
    print(f"Collecte ciblee sur tes 5 especes : {class_names}")

    cropper = YoloAnimalCropper(DETECTOR_ONNX_PATH)

    if DATASET_DIR.exists() and not BACKUP_DIR.exists():
        shutil.copytree(DATASET_DIR, BACKUP_DIR)
        print(f"Sauvegarde de l'ancien dataset -> {BACKUP_DIR}/")

    staging_dir = Path("dataset_wildlife_staging")
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True)

    seen_hashes: set[bytes] = set()
    seen_urls: set[str] = set()

    for idx_cls, cname in enumerate(class_names):
        remaining_time = max_seconds - (time.time() - t_start)
        remaining_classes = len(class_names) - idx_cls
        cls_deadline = time.time() + max(30.0, remaining_time / remaining_classes)

        taxon_id, sci_name = resolve_taxon_id(cname)
        print(f"\n=== [{idx_cls + 1}/5] {cname.upper()} ({sci_name}, ID {taxon_id}) ===")

        cls_dir = staging_dir / cname
        cls_dir.mkdir(parents=True, exist_ok=True)
        kept, rejected, page = 0, 0, 1

        while kept < TARGET_PER_CLASS and time.time() < cls_deadline and page <= 25:
            url = (
                f"https://api.inaturalist.org/v1/observations"
                f"?taxon_id={taxon_id}&quality_grade=research&photos=true"
                f"&per_page=200&page={page}&order_by=votes"
            )
            results = http_get_json(url).get("results", [])
            if not results:
                break

            photo_urls = []
            for obs in results:
                for p in obs.get("photos", [])[:2]:
                    u = p.get("url", "").replace("square.", "medium.")
                    if u and u not in seen_urls:
                        seen_urls.add(u)
                        photo_urls.append(u)

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = [pool.submit(download_image, u) for u in photo_urls]
                for fut in as_completed(futures):
                    if kept >= TARGET_PER_CLASS or time.time() >= cls_deadline:
                        break
                    im = fut.result()
                    if im is None:
                        rejected += 1
                        continue
                    h = compute_dhash(im)
                    if h in seen_hashes:
                        rejected += 1
                        continue
                    seen_hashes.add(h)

                    cropped = cropper.crop_animal(im)
                    if cropped is None:
                        rejected += 1
                        continue

                    cropped.save(cls_dir / f"{cname}_{kept:05d}.jpg", format="JPEG", quality=92)
                    kept += 1
                    if kept % 100 == 0 or kept == TARGET_PER_CLASS:
                        print(f"  [{cname}] {kept}/{TARGET_PER_CLASS} recadrees | rejetes: {rejected} | {(time.time() - t_start)/60:.1f} min")

            page += 1
            time.sleep(1.1)

    rng = random.Random(42)
    shutil.rmtree(DATASET_DIR, ignore_errors=True)
    for cname in class_names:
        imgs = sorted((staging_dir / cname).glob("*.jpg"))
        rng.shuffle(imgs)
        n_val = max(16, int(round(len(imgs) * VAL_RATIO)))
        (DATASET_DIR / "train" / cname).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "val" / cname).mkdir(parents=True, exist_ok=True)
        for p in imgs[n_val:]:
            shutil.move(str(p), str(DATASET_DIR / "train" / cname / p.name))
        for p in imgs[:n_val]:
            shutil.move(str(p), str(DATASET_DIR / "val" / cname / p.name))
        print(f"Dataset [{cname}] : {len(imgs) - n_val} train | {n_val} val")

    shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"\nCollecte terminee en {(time.time() - t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()
