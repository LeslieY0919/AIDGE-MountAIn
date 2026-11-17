# Wildlife Detection & Species Classification

Pipeline de détection et de classification d'animaux sauvages avec Aidge.
Le dépôt contient les scripts Python, le wrapper C++ autonome et les fichiers
de configuration du projet.

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
standalone_wildlife_cpp/build/wildlife_pipeline test_wildlife_1mp.jpg
```

**Important :** le CMake s'attend à trouver les sources générées dans
`standalone_wildlife_cpp/detector_dnn/` et
`standalone_wildlife_cpp/classifier_dnn/`. Ces dossiers ne sont pas présents
dans ce dépôt; le pipeline C++ ne peut donc pas être compilé à partir du dépôt
seul. Il faut d'abord générer ou récupérer ces sources.

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
