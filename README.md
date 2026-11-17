# Wildlife Detection & Species Classification

Projet du Challenge Aidge 2026, sujet n° 3 « AIDGE for Wildlife Monitoring »
(proposé par MountAIn) : détecter les animaux sauvages et identifier l'espèce
sur un Alif Ensemble E7 (Cortex-M55 + Ethos-U55), avec la caméra de la carte
et une inférence entièrement embarquée. Le déploiement final passe par
Eclipse Aidge.

Le pipeline compte deux étapes : un détecteur (MegaDetector v6,
`MDV6-yolov10-c`) trouve les animaux, puis un classifieur reconnaît l'espèce
sur chaque découpe. Classes actuelles (`class_names.json`) : blaireau,
chevreuil, écureuil, renard, sanglier.

## Arborescence

| Dossier | Contenu |
|---|---|
| `dataset/` | Construction du jeu de données à 5 espèces (iNaturalist, découpes par MegaDetector) |
| `training/` | Récupération de MegaDetector v6, entraînement du détecteur et du classifieur |
| `export/` | Découpe/pliage ONNX, quantification INT8 et export C++ par Aidge, assemblage du pipeline autonome |
| `checks/` | Vérifications : C++ contre ONNX Runtime, pipeline Python de référence, diagnostic de la PTQ |
| `standalone_wildlife_cpp/` | Pipeline C++ autonome (détecteur + classifieur), généré par `export/build_standalone_cpp.py` |
| `samples/` | Images de test |
| `third_party/` | Dépendances C++ embarquées (`stb_image.h`) |
| `u55/` | Profilage Vela sur Ethos-U55, décodage de référence de YOLOv10 pour le M55 |
| `scripts/` | Environnement et compilation pour la carte Alif E7, Aidge dans Docker sur une machine Linux |
| `aidge/` | Image Docker Aidge (TOSA, export Ethos, Vela) |
| `archive/` | Scripts d'exploration et versions précédentes, conservés pour mémoire, non maintenus |

Tous les scripts Python se lancent **depuis la racine du dépôt** (les chemins
des modèles, de `class_names.json` et de `samples/` sont relatifs à la racine),
par exemple `python export/quantize_pipeline_int8.py`.

## Chaîne de production des modèles

1. `dataset/build_5species_dataset.py` : construit `dataset_wildlife/`.
2. `training/train_mdv6.py` : affine MegaDetector v6 en 640 et produit
   `detector_v6_trained_640.onnx` ; `training/train_classifier_gpu.py`
   entraîne le classifieur en 128 et produit `species_classifier.onnx`.
3. `export/export_detector_v6.py` : coupe le détecteur avant le
   post-traitement (`MDV6-yolov10-c_cut_folded.onnx`) et fait l'export C++
   float32 ; `export/export_classifier.py` produit
   `species_classifier_folded.onnx`.
4. `export/quantize_pipeline_int8.py` : quantification INT8 et export C++
   (`export_cpp_detector_int8/`, `export_cpp_classifier_int8/`).
5. `export/build_standalone_cpp.py` : assemble `standalone_wildlife_cpp/`.

Les modèles (`*.onnx`, `*.pt`), les jeux de données et les exports générés ne
sont pas versionnés ; ils se partagent via les GitHub Releases.

## Pipeline C++ sur une carte Linux

Le projet CMake cible Linux et nécessite CMake 3.16+, un compilateur C++17 et
les sources générées par Aidge. Sous Debian/Ubuntu, installez les outils avec :

```bash
sudo apt install build-essential cmake
```

Puis, depuis la racine du dépôt :

```bash
cmake -S standalone_wildlife_cpp -B standalone_wildlife_cpp/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build standalone_wildlife_cpp/build -j2
standalone_wildlife_cpp/build/wildlife_pipeline samples/test_wildlife_1mp.jpg
```

**Important :** le CMake s'attend à trouver les sources générées dans
`standalone_wildlife_cpp/detector_dnn/` et
`standalone_wildlife_cpp/classifier_dnn/`. Ces dossiers ne sont pas présents
dans ce dépôt ; le pipeline C++ ne peut donc pas être compilé à partir du dépôt
seul. Il faut d'abord les générer (étapes 3 à 5 ci-dessus) ou les récupérer.

## Environnement Python pour entraînement et export

Python 3.12 est utilisé pour l'environnement de développement.

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps aidge-quantization
```

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps aidge-quantization
```

L'installation de `aidge-quantization` avec `--no-deps` évite de télécharger
`aidge-backend-cuda`, qui n'est pas nécessaire aux flux CPU utilisés ici.
Les dépendances `aidge-core`, `aidge-backend-cpu` et `numpy` sont déjà
installées par `requirements.txt`.

La distribution officielle de `aidge-quantization` déclare néanmoins
`aidge-backend-cuda` comme dépendance obligatoire dans ses métadonnées. Par
conséquent, `pip check` peut signaler cette dépendance absente, même si les
imports et flux CPU concernés fonctionnent. Pour utiliser le backend CUDA,
installez `aidge-quantization` sans `--no-deps` et configurez le CUDA Toolkit
ainsi que `CUDA_PATH`.

## Carte Alif Ensemble E7 (Ethos-U55)

La carte cible est l'Alif Ensemble E7 AI/ML AppKit envoyée par les
organisateurs. Le profilage des modèles sur Ethos-U55 (`u55/`), la compilation
du firmware avec le kit d'évaluation ML d'Alif (`scripts/`) et l'environnement
Aidge → TOSA → Vela (`aidge/`) sont décrits dans [AGENTS.md](AGENTS.md).
