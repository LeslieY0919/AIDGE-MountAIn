# Migration — ancienne arborescence → nouvelle

Table de correspondance complète. L'historique git est préservé : tous les
déplacements ont été faits avec `git mv`, donc `git log --follow <nouveau
chemin>` remonte au-delà de la réorganisation.

## Scripts de la chaîne — `scripts/`

| Avant | Après | Note |
|---|---|---|
| `fetch_real_mdv6.py` | `scripts/detector/fetch_weights.py` | |
| `train_mdv6.py` | `scripts/detector/finetune.py` | |
| `export_detector_v6.py` | `scripts/detector/export_cpp.py` | **réécrit** sur `mountain.aidge` ; son C++ est sorti dans `cpp/detector_standalone/main.cpp` |
| `build_5species_dataset.py` | `scripts/classifier/build_dataset.py` | |
| `train_classifier_gpu.py` | `scripts/classifier/train.py` | |
| `export_classifier.py` | `scripts/classifier/export_cpp.py` | |
| `quantize_pipeline_int8.py` | `scripts/quantize/pipeline_int8.py` | |
| `rebuild_classifier_int8.py` | `scripts/quantize/rebuild_classifier.py` | son import devient `from pipeline_int8 import ...` |
| `build_standalone_cpp.py` | `scripts/pipeline/build_standalone_cpp.py` | écrit désormais dans `exports/wildlife_pipeline/` |
| `run_full_pipeline.py` | `scripts/pipeline/run_reference.py` | renommé : c'est l'oracle Python du post-traitement |

## Contrôles — `tests/`

Préfixe `check_` et non `test_` : ce sont des scripts exécutables qui ont
besoin d'artefacts construits, pas des tests unitaires. Le préfixe les tient
hors de la collecte `pytest`.

| Avant | Après |
|---|---|
| `test_pipeline.py` | `tests/check_aidge_onnx_smoke.py` |
| `test_detector_export.py` | `tests/check_detector_export.py` |
| `verify_mdv6_aidge.py` | `tests/check_detector_aidge_parity.py` |
| `test_cpp_detector.py` | `tests/check_cpp_detector_fp32.py` |
| `run_cpp_check.py` | `tests/check_cpp_detector_int8.py` |
| `verify_cpp_detector_int8.py` | `tests/check_detector_int8_decode.py` |
| `test_full_cpp_pipeline.py` | `tests/check_cpp_pipeline_full.py` |

## Audits — `tools/`

| Avant | Après |
|---|---|
| `audit_mdv6.py` | `tools/audit_detector.py` |
| `check_detector.py` | `tools/audit_detector_arch.py` |
| `audit_classifier.py` | `tools/audit_classifier.py` |
| `diagnose_ptq.py` | `tools/diagnose_ptq.py` |
| `tune_ptq.py` | `tools/tune_ptq.py` |
| `stage1_tiling.py` | `tools/explore_tiling.py` |

## C++, configuration, données

| Avant | Après | Note |
|---|---|---|
| chaîne `main_cpp` de `export_detector_v6.py` | `cpp/detector_standalone/main.cpp` | 188 lignes de C++ extraites d'une chaîne Python |
| `standalone_wildlife_cpp/{main.cpp,CMakeLists.txt,*.map}` | `cpp/wildlife_pipeline/` | **instantané de référence** : ces fichiers sont générés, voir le README du dossier |
| `standalone_wildlife_cpp/stb_image.h` | `cpp/third_party/stb/stb_image.h` | source canonique, recopiée par les scripts d'export |
| `class_names.json` | `configs/classes.json` | |
| — | `configs/detector.yaml` | **nouveau** : résolutions, seuils, budgets mémoire |
| — | `configs/classifier.yaml` | **nouveau** : architecture, normalisation, hyperparamètres, dataset |
| — | `configs/quantization.yaml` | **nouveau** : nb de bits, calibration, options de PTQ |
| `test_wildlife_1mp.jpg` | `data/samples/test_wildlife_1mp.jpg` | |
| `Ecureuil-roux.jpg` | `data/samples/ecureuil-roux.jpg` | minusculisé |
| `exported_graph.mmd` | `docs/detector_graph.mmd` | |

## Itérations historiques — `attic/`

| Avant | Après |
|---|---|
| `export_detector.py` | `attic/detector/export_v1.py` |
| `export_detector_v2.py` … `_v5.py` | `attic/detector/export_v2.py` … `export_v5.py` |
| `export_detector_v6.py.bak` | `attic/detector/export_v6.py.bak` |
| `patch_onnx_split.py` | `attic/detector/finetune_no_psa.py` |
| `train_classifier.py` | `attic/classifier/train_v1.py` |
| `quantize_int8.py` | `attic/quantize/quantize_int8.py` |
| `quantize_and_export_cpp.py` | `attic/quantize/quantize_and_export_cpp.py` |
| `wildlife_cpp_pipeline/` | `attic/wildlife_cpp_pipeline/` |

Les chemins y sont restés sous leur forme ancienne : ces scripts ne tourneront
pas tels quels depuis la nouvelle arborescence. C'est volontaire — ce sont des
archives, pas du code à maintenir.

## Depuis `clean-main` après la PR #1

`clean-main` avait reçu entre-temps une autre réorganisation (PR #1 :
`training/`, `export/`, `checks/`, `dataset/`, `archive/`, `samples/`,
`third_party/`, `standalone_wildlife_cpp/`). Elle est **remplacée** par
l'arborescence décrite ici : pour tous les scripts de la chaîne, c'est la
version de cette branche qui est gardée. Seuls les fichiers qui n'existaient
que sur `clean-main` ont été déplacés :

| Sur `clean-main` | Ici | Note |
|---|---|---|
| `scripts/env.sh` | `scripts/board/env.sh` | définit aussi `$SETOOLS` ; réglages locaux dans `scripts/board/env.local.sh` |
| `scripts/build_img_class.sh` | `scripts/board/build_img_class.sh` | cible `make board-firmware` |
| `scripts/mlek_requirements_tflm.txt` | `scripts/board/mlek_requirements_tflm.txt` | |
| `scripts/dev2_aidge.sh` | `scripts/npu/aidge_docker.sh` | |
| `aidge/export_classifier_npu.py` | `scripts/npu/export_classifier.py` | chemins par `mountain.paths`, normalisation par `mountain.preprocess` ; sorties dans `exports/npu/` |
| `aidge/Dockerfile` | `docker/aidge/Dockerfile` | |
| `u55/*.py` | `tools/u55/*.py` | sorties dans `artifacts/u55_profile/` (avant : `build/ml_profile/`) |
| `AGENTS.md` | `AGENTS.md` | réduit à l'outillage de la carte et aux mesures ; le contexte du projet est dans `CLAUDE.md` |
| — | `notes/` | **nouveau** : notes d'échange |

Les autres dossiers de `clean-main` correspondent ainsi : `training/`,
`export/` et `dataset/` → `scripts/` ; `checks/` → `tests/` et `tools/` ;
`archive/` → `attic/` ; `samples/` → `data/samples/` ; `third_party/` →
`cpp/third_party/stb/` ; `standalone_wildlife_cpp/` → `cpp/wildlife_pipeline/` ;
`class_names.json` → `configs/classes.json`.

## Sorti du suivi git

### Exports générés — 1,2 Go, 4 780 fichiers

`export_cpp_detector/`, `export_tmp/`, `export_tmp_cmp_test/`,
`export_tmp_det_int8/` ont été retirés de l'index avec
`git rm -r --cached`. **Les fichiers sont toujours sur le disque**, et
l'historique de `clean-main` les conserve : rien n'est perdu, et un
`git checkout clean-main` les fait réapparaître.

Ils sont régénérables par la chaîne, et leur nouvelle place est `exports/`.
Une fois la branche validée, ils peuvent être supprimés localement :

```bash
rm -rf export_cpp_detector export_tmp export_tmp_cmp_test export_tmp_det_int8
```

### Sorties de run — supprimées de la branche

`crop_animal_0.jpg`, `debug_detections.jpg`, `detection_cpp_int8_result.jpg`,
`result_pipeline.jpg`, `test_wildlife_1mp_det.jpg`. Ce sont des images
annotées produites par les scripts ; elles atterrissent désormais dans
`artifacts/`, qui n'est pas versionné. L'historique de `clean-main` les garde.

### Scories — supprimées

- `dump_pixels.txt` — fichier vide ;
- `tiny_test.onnx.data` — sortie de l'ancien `test_pipeline.py`.

## Ce qui change pour qui écrit du code

**Installer le paquet une fois.** Les scripts importent `mountain`, qui doit
être installé :

```bash
pip install -e .
```

Sans cette installation, tous les scripts échouent sur
`ModuleNotFoundError: No module named 'mountain'`. C'est le prix du layout
`src/`, et c'est voulu : il n'y a plus d'import accidentel depuis le
répertoire courant.

**Ne plus écrire de chemin en dur.**

```python
# avant
graph = aidge_onnx.load_onnx("MDV6-yolov10-c_cut_folded.onnx")

# après
from mountain import paths
graph = aidge_onnx.load_onnx(str(paths.DETECTOR_ONNX_CUT_FOLDED))
```

**Les constantes des scripts actifs sont désormais lues dans `configs/`.** Les
23 valeurs qui étaient en tête de `train.py`, `build_dataset.py` et
`finetune.py` (résolutions, tailles de lot, époques, taux d'apprentissage,
seuils de recadrage, liste des espèces) viennent de `configs/*.yaml`. Les noms
de variables sont inchangés, donc le corps des scripts n'a pas bougé :

```python
# avant
IMG_SIZE = 128
BATCH_SIZE = 128
EPOCHS_FULL = 44

# après
_CFG = config.classifier()
IMG_SIZE = _CFG["input_size"]
BATCH_SIZE = _CFG["training"]["batch_size"]
EPOCHS_FULL = _CFG["training"]["epochs_full"]
```

Chacune des 23 valeurs a été vérifiée égale à l'ancienne, et l'ordre des cinq
espèces est préservé — il détermine l'indice des logits du classifieur.

**Ne plus redéfinir de constante.**

```python
# avant, recopié dans huit fichiers
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize((128, 128), Image.BILINEAR)
    arr = (np.array(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...])

# après
from mountain.preprocess import classifier_input
x = classifier_input(img_path)          # la taille vient de configs/classifier.yaml
```

**Préparer un graphe AIDGE en un appel.**

```python
# avant, avec des variantes divergentes selon les fichiers
graph = aidge_onnx.load_onnx(path)
for node in graph.get_nodes():
    if node.type() == "Producer":
        op = node.get_operator()
        if hasattr(op, "set_constant"):
            op.set_constant(True)
        elif hasattr(op, "attr") and hasattr(op.attr, "constant"):
            op.attr.constant = True
aidge_core.constant_folding(graph)
graph.forward_dims([[1, 3, 128, 128]])
graph.set_datatype(aidge_core.dtype.float32)
graph.set_backend("cpu")

# après
from mountain.aidge.graph import load_graph
graph = load_graph(path, [1, 3, 128, 128])
```

## Limite de cette migration

Le recâblage des chemins a été fait par substitution de littéraux, puis
vérifié par analyse syntaxique de tous les fichiers et par recherche des
littéraux résiduels. **Aucun script n'a pu être exécuté** : la chaîne demande
l'environnement AIDGE, les poids des modèles et le dataset, qui ne sont pas
versionnés.

Le premier passage effectif de `make detector-weights` → `make check` reste
donc à faire, et c'est lui qui validera la migration. Les points à surveiller
en priorité :

1. **`scripts/quantize/pipeline_int8.py`** — le fichier le plus substitué
   (neuf littéraux). Il manipule des chemins dans des chaînes C++ générées :
   l'ancien `"../../test_wildlife_1mp.jpg"`, relatif au dossier de build, est
   devenu un chemin absolu, ce qui devrait être plus robuste mais n'a pas été
   vérifié à l'exécution.
2. **`scripts/pipeline/build_standalone_cpp.py`** — lit les facteurs d'échelle
   int8 par expression régulière dans le `main.cpp` de l'export du détecteur.
   Dépend donc du format exact de ce fichier généré.
3. **Les types.** Les substitutions conservent `str(...)` partout où la
   variable d'origine était une chaîne, précisément pour ne pas transformer en
   `Path` une valeur sur laquelle le script fait des opérations de chaîne.
   Les scripts qui utilisaient déjà `Path(...)` reçoivent un `Path`.
