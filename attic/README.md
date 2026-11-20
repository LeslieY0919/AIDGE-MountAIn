# `attic/` — itérations historiques

Ces fichiers ne sont plus dans la chaîne. Ils sont conservés parce que les
tentatives d'export AIDGE qui **échouent** sont de l'information : elles
documentent quelles approches ne passent pas, ce qui évite de les retenter.

Rien ici n'est maintenu, et rien dans `scripts/`, `src/` ou `tests/` n'en
dépend. Les chemins de fichiers y sont restés sous leur ancienne forme
(littéraux relatifs à la racine) : ces scripts ne tourneront pas tels quels
depuis la nouvelle arborescence.

## `detector/` — la série des exports

Six tentatives successives d'export C++ de MegaDetector v6. Remplacées par
`scripts/detector/export_cpp.py`.

| Fichier | Approche | Pourquoi abandonnée |
|---|---|---|
| `export_v1.py` | export du graphe complet, repli sur une coupe en cas d'échec | le post-traitement YOLO n'est pas exportable ; la coupe est devenue le chemin nominal |
| `export_v2.py` | coupe explicite avant `Concat_1` | une seule sortie : le C++ a besoin de `boxes` **et** `scores` |
| `export_v3.py` | ajout du pliage de constantes | insuffisant seul, les scalaires 0-D bloquent encore |
| `export_v4.py` | variantes d'appel à l'API d'export | absorbé par `mountain.aidge.export` |
| `export_v5.py` | idem, autre signature | idem |
| `export_v6.py.bak` | sauvegarde de la version retenue | remplacée par la version suivie dans `scripts/` |
| `finetune_no_psa.py` | fine-tuning avec `model.10.attn` remplacé par `Identity`, pour supprimer les `MatMul` qu'AIDGE ne gérait pas | demande un `data.yaml` renseigné à la main ; remplacé par `scripts/detector/finetune.py`, qui automatise la préparation du dataset |

La leçon de la série : il faut **couper le graphe avant le post-traitement
DFL/Sigmoid**, plier les constantes avec ONNX Runtime, et promouvoir les
constantes de rang 0. Les trois ensemble, pas l'un ou l'autre.

## `classifier/`

- `train_v1.py` — premier entraînement, avec téléchargement iNaturalist en
  ligne (80 images par classe) et `timm` comme fournisseur d'architecture.
  Remplacé par `scripts/classifier/train.py` (ResNet-34, deux phases) et
  `scripts/classifier/build_dataset.py`, qui recadre les animaux avec le
  détecteur au lieu d'apprendre sur des images entières.

## `quantize/`

- `quantize_int8.py` — PTQ du classifieur seul, sans export C++.
- `quantize_and_export_cpp.py` — PTQ des deux modèles, mais sans les
  correctifs `Dequantizer`/`Quantizer` qui rendent l'export réellement
  compilable.

Les deux sont remplacés par `scripts/quantize/pipeline_int8.py`.

## `wildlife_cpp_pipeline/`

Première version du pipeline C++ à deux modèles. Les deux modèles exportés
définissent chacun un symbole `model_forward`, d'où une collision à l'édition
de liens. Cette version la résolvait en préfixant les symboles avec
`nm` + `objcopy --redefine-syms`, ce qui dépend des binutils GNU et ne marche
pas sur macOS.

L'approche retenue — bibliothèques partagées et scripts de version de
l'éditeur de liens, voir `scripts/pipeline/build_standalone_cpp.py` — obtient
le même résultat avec le seul éditeur de liens.
