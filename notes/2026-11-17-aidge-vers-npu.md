# Aidge → TOSA → Vela : exporter les modèles pour l'Ethos-U55

- **Auteur** : LeslieY0919
- **Date** : 2026-11-17, revue le même jour avec les outils du porteur
- **État** : [MESURÉ] sur PC, dans l'image `docker/aidge/` (Aidge 0.10.1, Vela 5.1.0). Les temps et la mémoire sont
  des estimations de Vela. [SOURCÉ] pour ce qui est lu dans le code du SDK `alif_aidge_sdk` (`eb35a81`), du fork
  `aidge-tosa-fork` (`8223ea3`) et du fork d'`aidge_export_ethos` (branche `feat/ethos-be-lowering`, `4cfbc5d`).
  **Rien n'a tourné sur la carte.**

## En bref

1. **Avec les outils du porteur, la chaîne Aidge → NPU fonctionne et donne des chiffres proches de TFLite.** Le
   MobileNetV2-1.0 en 224 passe entièrement sur le NPU : 68 opérateurs, 13,1 ms et 1,44 Mio de SRAM, contre
   10,7 ms et 1,44 Mio par TFLite.
2. **Le détecteur passe aussi entièrement sur le NPU.** C'est un YOLOv10n d'origine (SiLU + PSA), en 320×320 :
   344 opérateurs, 54 ms et 0,79 Mio de SRAM. Il faut deux conditions :
   - remplacer les paires de Slice des C2f par des Split, ce qui ne change pas le calcul ; sinon l'export du fork
     échoue ;
   - installer `aidge_export_ethos` depuis le **fork du porteur, branche `feat/ethos-be-lowering`**. Avec le dépôt
     officiel, celui qu'installent CLAUDE.md §3 et le README du SDK, Vela refuse le modèle.
3. **Le SDK impose sa convention d'entrée.** Le modèle flottant prend une image dans [0, 1] ; la carte lui envoie
   `code = round(pixel / 255 × 127)`, avec un zéro à 0, en NHWC. `scripts/npu/export_classifier.py` suit la
   convention du kit d'Alif (`pixel − 128`) : ses modèles ne conviennent pas au firmware du SDK.
4. Mes premières mesures avec l'`aidge_tosa` **officiel** (77 ms et 9,2 Mio pour MobileNetV2, échec en NHWC) ne
   valent que pour lui. Il ne faut pas l'utiliser pour la carte.

## Ce qui a été essayé

- **La chaîne du SDK** (`alif_aidge_sdk.export.export_model`, celle de `alif-aidge-export`) :
  1. import de l'ONNX en NCHW, puis quantification int8 par la PTQ d'Aidge (le Softmax reste en flottant) ;
  2. export TOSA puis Vela, par `aidge_export_arm_ethos`, avec la configuration du cœur HP (U55-256) ;
  3. comparaison du fichier TOSA avec le graphe Aidge, par le modèle de référence TOSA ;
  4. écriture de `model_io.json`.

  `scripts/npu/sdk_export_report.py` appelle cette fonction et résume le résultat. Pour MobileNetV2, l'ONNX est
  d'abord préparé comme dans l'exemple `examples/mobilenetv2` du SDK : la normalisation ImageNet est repliée dans
  la première convolution et l'entrée est dans [0, 1].
- **Notre script** `scripts/npu/export_classifier.py`, avec l'`aidge_tosa` officiel puis avec le fork.
- **Modèles** :
  - `mobilenetv2-7.onnx`, publié par Aidge sur Hugging Face ;
  - YOLOv10n d'Ultralytics (COCO, 80 classes), exporté en 320 par `tools/u55/profile_yolov10_u55.py` et coupé aux
    6 sorties brutes de la branche one-to-one.
- **Calibration** : 10 images (2 de `data/samples`, 8 de COCO8), et seulement 2 pour MobileNetV2 dans la chaîne
  du SDK. Cela suffit pour faire tourner la chaîne, pas pour juger la précision.

## Résultats

MobileNetV2-1.0, 224×224, cœur HP (U55-256) :

| Chemin | Opérateurs NPU / CPU | Temps | Pic SRAM | MRAM |
|---|---|---|---|---|
| TFLite → Vela (modèle fourni avec le kit d'Alif) | – | 10,7 ms | 1,44 Mio | 3,29 Mio |
| Aidge + fork, chaîne du SDK (entrée NHWC) | 68 / 0 | 13,1 ms | 1,44 Mio | 2,98 Mio |
| Aidge + fork, `export_classifier.py` (entrée NCHW) | 71 / 0 | 13,2 ms | 1,44 Mio | 2,98 Mio |
| Aidge + `aidge_tosa` officiel, entrée NCHW | 472 / 0 | 77,0 ms | 9,19 Mio | 2,98 Mio |
| Aidge + `aidge_tosa` officiel, entrée NHWC | refusé par Vela | – | – | – |

Les deux MobileNetV2 ne sont pas exactement le même fichier (0,30 et 0,43 GMAC). Avec l'`aidge_tosa` officiel,
le graphe reste en NCHW et chaque convolution est encadrée de deux Transpose. Le fork les retire.

YOLOv10n d'origine (SiLU + PSA), 320×320, cœur HP (U55-256), 0,85 GMAC :

| Chemin | Opérateurs NPU / CPU | Temps | Pic SRAM | MRAM |
|---|---|---|---|---|
| TFLite → Vela (`tools/u55/`) | 579 / 0 | 48,2 ms | 5,26 Mio | 2,03 Mio |
| Aidge + fork, chaîne du SDK, Slice des C2f remplacés par des Split | 344 / 0 | 54,3 ms | 0,79 Mio | 1,62 Mio |
| idem, ONNX tel quel | échec de l'export TOSA | – | – | – |
| idem, Slice remplacés, `aidge_export_ethos` officiel | refusé par Vela | – | – | – |

Les deux chemins utilisent les mêmes réglages de Vela (`Shared_Sram`, `RTSS_HP_SRAM_MRAM`, même fichier `.ini`) ;
seule la version diffère (5.0.0 pour TFLite, 5.1.0 pour Aidge).

Contrôles :

- **TOSA contre Aidge** (contrôle `--verify` du SDK). Pour MobileNetV2, l'écart moyen est de 1,4 code et l'écart
  maximal de 7. Pour YOLOv10n, l'écart moyen va de 0,5 à 1,5 code selon la sortie, et au plus 0,025 % des valeurs
  s'écartent de plus de 16 codes. C'est dans la tolérance du SDK.
- **Flottant contre int8.**
  - MobileNetV2 : même classe sur 2 images sur 2 dans la chaîne du SDK, sur 8 sur 10 avec notre script.
  - YOLOv10n : corrélation de 0,99 sur les trois sorties de boîtes, sur une seule image. Sur les sorties de
    classes, elle tombe à 0,86, 0,84 et 0,56 (cartes 40×40, 20×20 et 10×10). La quantification abîme donc
    surtout les scores de classes. Ce n'est pas une mesure de précision (pas de mAP), mais c'est le premier point
    à surveiller (CLAUDE.md §3, piège n° 2).

## Ce qu'il faut savoir

- **Slice et NHWC dans le fork.** Dans notre ONNX, chaque C2f sépare ses canaux par deux Slice sur l'axe 1. Le fork
  exporte le graphe en NHWC, comme le demande le SDK, mais ces Slice continuent de couper l'axe 1, qui est
  désormais la hauteur. L'export s'arrête alors sur
  `Graph dimensions could not be forwarded after transpose sinking`, précédé de
  `Wrong input size ([1, 32, 18, 82]) for Conv operator. Expected dims are [x, 16, x, x]`.
  - Le Split, qui porte un seul attribut `axis`, est bien converti. Hypothèse : l'attribut `axes` du Slice n'est
    pas converti lors du passage en NHWC.
  - `scripts/npu/onnx_slices_to_split.py` remplace chaque paire de Slice par un Split équivalent. Aidge donne
    exactement les mêmes sorties flottantes avant et après.
  - Reste à savoir si `MDV6-yolov10-c` a les mêmes Slice : cela dépend de son export (`model.export` d'Ultralytics
    ou `torch.onnx.export`). À signaler au porteur.
- **`ethos_be`.** Le frontal TOSA de Vela n'a pas de repli sur le CPU : le premier opérateur flottant arrête la
  compilation (docstring d'`aidge_export_arm_ethos.export`, branche du fork). Le SDK quantifie tout sauf le
  Softmax. `ethos_be` remplace les îlots Softmax et Sigmoid par des opérateurs entiers : une table, et un
  Softmax décomposé en entiers.
  - [SOURCÉ] Cette option n'existe que sur la branche `feat/ethos-be-lowering` de
    `https://gitlab.eclipse.org/olivierpds/aidge_export_ethos` (vérifié par `git ls-remote` : le dépôt officiel
    ne l'a pas).
  - Le SDK teste la signature et s'en passe sans prévenir.
  - [MESURÉ] Avec le dépôt officiel, Vela refuse YOLOv10n (`Validation error`, constantes flottantes :
    `Output data type not supported`).
  - MobileNetV2, qui n'a pas d'îlot flottant, passe avec les deux versions.
- **Convention d'entrée du SDK.** La PTQ suppose une entrée flottante dans [0, 1], codée en `round(x × 127)`. Le
  runtime du SDK calcule ce code à partir des pixels (`code = round(pixel / 255 × 127)`, zéro 0, NHWC).
  - Pour un classifieur entraîné avec la normalisation ImageNet, on replie cette normalisation dans la première
    convolution avant l'export, comme `examples/mobilenetv2/model/prepare_onnx.py`.
  - `export_classifier.py` la replie autrement, pour `q = pixel − 128` : c'est la convention du kit d'Alif
    (`scripts/board/build_img_class.sh`), pas celle du SDK.
- **Échelles de sortie.** Le `.tflite` produit par Vela n'a ni échelle ni zéro sur ses entrées et sorties. Le SDK
  ajuste une échelle par sortie (moindres carrés contre l'ONNX flottant) et l'écrit dans `model_io.json`.
  `alif-aidge-gen-model` la reporte ensuite dans `model_io.hpp`.
- **Opérateurs du fork** ([SOURCÉ], `aidge_tosa/operators/`) : Add, AvgPool, Cast, Concat, Conv, Div, FC,
  GlobalAvgPool, MatMul, MaxPool, Mul, ReLU, Reshape, Resize, Sigmoid, Slice, Softmax, Split, Sub, Transpose, et
  les versions quantifiées de Sigmoid et Softmax.
  - L'`aidge_tosa` officiel n'a ni Concat, ni Slice, ni Resize, ni Sigmoid, ni Softmax, ni MatMul.
- **Versions.**
  - Le fork fixe `tosa-tools==2026.5.0`, l'image a la 2026.8.0 : les mesures ci-dessus tournent avec la 2026.8.0
    sans problème visible.
  - `import aidge_core` doit précéder `import aidge_quantization`, sinon l'import échoue
    (`unknown base type Aidge::OperatorTensor`).
  - `import aidge_export_arm_ethos.export as m` donne la fonction et non le module : écrire
    `from aidge_export_arm_ethos.export import export`.
- **Notre script en NHWC avec le fork** se termine par un plantage (segfault). Je n'ai pas cherché pourquoi : la
  chaîne du SDK, elle, passe en NHWC.

## Ce que ça change

- La route Aidge → TOSA → Vela → TFLM tient pour les deux modèles, avec les outils du porteur. Pour la carte, le
  SDK remplace notre script et le kit d'Alif : voir [la note sur la carte](2026-11-17-contraintes-carte-et-kit.md).
- Le pic de 5,3 Mio de SRAM du YOLOv10n d'origine venait de la conversion TFLite. Par Aidge, il est de 0,79 Mio
  en 320, ce qui tient dans l'arène de 2,5 Mo. Le coût en temps de SiLU, lui, reste : 54 ms d'après Vela, et
  environ le double sur la carte selon Alif.
- Le prochain point à surveiller est la quantification des sorties de classes du détecteur.

## À vérifier

- Les vrais modèles : `MDV6-yolov10-c` (Slice ou Split ? temps et SRAM en 320 et au-delà) et le classifieur du
  groupe.
- La calibration sur des images représentatives, et la précision du graphe int8 (top-1, rappel).
- L'exécution sur la carte : aucun `.tflite` exporté n'a encore été flashé.
- Le défaut des Slice, à signaler au porteur (voir [questions-ouvertes.md](questions-ouvertes.md)).

## Reproduire

Les dépôts du porteur ne sont pas versionnés ici. Les cloner dans `external/organisateur/` :

```sh
O=https://gitlab.eclipse.org/olivierpds
git clone $O/alif_aidge_sdk.git external/organisateur/alif_aidge_sdk
git clone $O/aidge-tosa-fork.git external/organisateur/aidge-tosa-fork
git clone -b feat/ethos-be-lowering $O/aidge_export_ethos.git external/organisateur/aidge_export_ethos-fork
```

L'image `docker/aidge/` installe l'`aidge_tosa` officiel : y placer ces dépôts en tête du `PYTHONPATH`. Depuis un
Mac :

```sh
scripts/npu/aidge_docker.sh push external/organisateur/ models/yolov10n_silu_psa_320.onnx data/samples/ datasets/coco8/
scripts/npu/aidge_docker.sh run sh -c 'F=/work/external/organisateur
  export PYTHONPATH=$F/aidge_export_ethos-fork:$F/aidge-tosa-fork:$F/alif_aidge_sdk:$PYTHONPATH
  python scripts/npu/onnx_slices_to_split.py models/yolov10n_silu_psa_320.onnx models/yolov10n_silu_psa_320_split.onnx
  python scripts/npu/sdk_export_report.py --onnx models/yolov10n_silu_psa_320_split.onnx --size 320 \
      --calib data/samples datasets/coco8/images --verify'
```

Résultats dans `exports/npu/sdk_<modèle>_hp/` : `model.tosa`, `output/model_vela.tflite`, `model_io.json`,
`verify.json` et `result.json`. Pour MobileNetV2, préparer l'ONNX avec l'exemple du SDK
(`python -m model.fetch_model` puis `python -m model.prepare_onnx`, dans `examples/mobilenetv2`). Les anciennes
mesures avec l'`aidge_tosa` officiel se refont avec `scripts/npu/export_classifier.py`, dans l'image telle quelle.
