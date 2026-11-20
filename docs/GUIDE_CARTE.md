# Guide : de l'ONNX à la carte, avec les outils du porteur

> **État au 2026-11-19.** Chaîne déroulée de bout en bout le 2026-11-18, sur la carte du groupe, avec le
> MobileNetV2 de l'exemple du SDK : export par Aidge dans Docker sur une machine Linux, compilation et flash
> depuis un Mac. L'image de la caméra s'affiche sur l'écran et le modèle tourne sur le NPU du cœur HP. Ce qui n'a
> pas été refait ici est signalé à chaque étape.

Ce guide suit le chemin nominal du défi :

1. un ONNX flottant est quantifié en int8 par Aidge, puis compilé pour l'Ethos-U55 par Vela ;
2. le résultat est intégré au firmware du SDK du porteur (`alif_aidge_sdk`) ;
3. sur la carte, ce firmware filme, infère et affiche le résultat sur l'écran.

Contexte du défi : [CLAUDE.md](../CLAUDE.md). Outillage de la carte et mesures : [AGENTS.md](../AGENTS.md).

La carte du groupe est chez LeslieY0919. Sans carte, tout se fait jusqu'au §4 compris : lui envoyer ensuite le
`.mram.bin` pour le flash.

## Vue d'ensemble

```
ONNX fp32, entrée NCHW dans [0, 1]
  │ §2 exporter : PTQ int8 d'Aidge → fork aidge_tosa → Vela      Linux x86_64 ou Windows ; Docker depuis un Mac
  ▼
<export>/model_io.json, <export>/output/model_vela.tflite
  │ §3 générer les sources : gen_model.py                         toute machine, Python seul
  ▼
<export>/generated/model_data.cc, model_io.hpp
  │ §4 compiler : CMake + Arm GNU Toolchain 13.3                  toute machine
  ▼
build/<dossier>/mobilenetv2.mram.bin
  │ §5 flasher : tools/flash_mram.sh (SETOOLS)                    la machine reliée à la carte
  ▼
carte : caméra → NPU → écran
```

## 1. Installer

### Dépôts du porteur

```sh
O=https://gitlab.eclipse.org/olivierpds
git clone $O/alif_aidge_sdk.git external/organisateur/alif_aidge_sdk
git clone $O/aidge-tosa-fork.git external/organisateur/aidge-tosa-fork
git clone -b feat/ethos-be-lowering $O/aidge_export_ethos.git external/organisateur/aidge_export_ethos-fork
```

`external/` n'est pas versionné. Commits utilisés :

| Dépôt | Commit |
|---|---|
| SDK | `eb35a81` |
| `aidge-tosa-fork` | `8223ea3` |
| branche `feat/ethos-be-lowering` | `4cfbc5d` |

### Python pour l'export (Linux x86_64 ou Windows)

Aidge n'a de wheels que pour ces deux plateformes, avec Python 3.11 ou 3.12. Partir du venv du dépôt (voir le
[README](../README.md#démarrage)), puis ajouter les outils du porteur :

```sh
make venv && source .venv/bin/activate && make install       # Aidge 0.10.1, onnx, onnxruntime
pip install ./external/organisateur/aidge-tosa-fork ./external/organisateur/aidge_export_ethos-fork
pip install -e external/organisateur/alif_aidge_sdk
python -c "import aidge_tosa, aidge_export_arm_ethos, ethosu.vela, alif_aidge_sdk; print('ok')"
```

- **Différence avec CLAUDE.md §3 et le README du SDK** : ceux-ci installent `aidge_export_ethos` depuis le dépôt
  officiel. Or il faut la branche `feat/ethos-be-lowering` du fork du porteur. Sans elle, Vela refuse tout modèle
  qui garde un Softmax ou une Sigmoid en flottant, dont le détecteur.
- `pip` peut signaler un conflit sur `ml-dtypes`. D'après le README du SDK, il est sans conséquence.
- **Pas refait ici tel quel** : nous avons utilisé ces dépôts par le `PYTHONPATH`, dans l'image Docker décrite
  ci-dessous.

### Depuis un Mac

Aidge n'a pas de wheel macOS. L'export tourne alors dans l'image `docker/aidge/` (Aidge 0.10.1, Vela 5.1.0), sur
une machine Linux x86_64 jointe en SSH, par `scripts/npu/aidge_docker.sh` (hôte choisi avec `DEV2_HOST`). Cette
image a l'`aidge_tosa` officiel et pas d'onnxruntime :

- y envoyer les dépôts du porteur et les placer en tête du `PYTHONPATH` (exemple au §2) ;
- `scripts/npu/sdk_export_report.py` et `scripts/npu/sdk_mobilenetv2_animaux.py` calculent alors les sorties
  flottantes de référence avec Aidge.

Les §3 à §5 se font directement sous macOS.

### Firmware

- **Arm GNU Toolchain 13.3.rel1** (`arm-none-eabi`), pas plus récente : le kit d'Alif, dont le SDK reprend les
  briques, n'est testé qu'avec les versions 12.3 et 13.3.
- **CMake 3.21 ou plus**. Avec CMake 4, définir `CMAKE_POLICY_VERSION_MINIMUM=3.5` pour les anciens sous-modules
  (`scripts/board/env.sh` le fait).
- **Dépendances du SDK.**
  - En principe, `git -C external/organisateur/alif_aidge_sdk submodule update --init`, soit environ 500 Mo. Pas
    refait ici.
  - Sinon, désigner des copies existantes par des variables CMake. Elles doivent être aux commits que donne
    `git submodule status` dans le SDK.
  - Ces variables sont : `CMSIS_SRC_PATH`, `CORTEX_DFP_SRC_PATH`, `ALIF_CMSIS_PATH`, `ETHOS_U_NPU_DRIVER_SRC_PATH`,
    `TENSORFLOW_SRC_PATH`, `CMSIS_NN_SRC_PATH`, `AIPL_DIR`, `FLATBUFFERS_SRC_PATH`, `GEMMLOWP_SRC_PATH` et
    `RUY_SRC_PATH`.
  - Sur le Mac, faute de place, elles pointaient vers les dépendances du kit d'Alif (`make board-firmware`), et
    vers une extraction de `cmsis-alif` au commit du SDK (`git archive`).
- **SETOOLS** (propriétaire, sur alifsemi.com : `app-release-exec-linux` ou `app-release-exec-macos`).
  - Le configurer une fois, depuis la carte : `(cd $SETOOLS && ./tools-config -a -b 55000 -c <port>)`.
  - `cat $SETOOLS/utils/global-cfg.db` doit alors annoncer un E7 (AE722F80F55D5).
- **Port de la carte.**
  - Sous Linux : `/dev/ttyACM*`, ou `/dev/alif-se` avec les règles udev du SDK.
  - Sous macOS : un seul port, `/dev/cu.usbmodem*`. C'est l'UART du Secure Enclave quand les cavaliers J15 sont
    sur SEUART (position par défaut).
- **`source scripts/board/env.sh`** met la toolchain dans le `PATH` et définit `SETOOLS`, `ALIF_SETOOLS_ROOT` et
  `ALIF_SEUART_PORT`.
  - Ses chemins par défaut sont ceux d'une installation sous macOS dans `~/opt`.
  - Pour une autre installation, les redéfinir dans `scripts/board/env.local.sh`, non versionné.

## 2. Exporter le modèle

Conditions à respecter, quel que soit le modèle :

- **Entrée de l'ONNX** : fp32, NCHW, dans [0, 1]. Replier la normalisation (moyenne, écart type) dans le modèle.
  Sur la carte, le firmware donne au NPU `code = round(pixel / 255 × 127)`.
- **100 % des opérateurs sur le NPU** après Vela (`CPU operators = 0` dans son résumé). Le firmware n'enregistre
  que l'opérateur Ethos-U dans TFLM.
- **Cible** `hp` (Ethos-U55-256) ou `he` (Ethos-U55-128) : la même que le cœur compilé au §4. CMake le vérifie.
- **Calibration** sur des images représentatives : la précision int8 en dépend.

En sortie, on obtient `model_io.json` et `output/model_vela.tflite`. S'y ajoutent `model.tosa` et le résumé de Vela
(`output/*_summary_*.csv`). Avec `--verify`, le graphe TOSA est comparé au graphe Aidge.

### L'exemple du SDK

MobileNetV2, 1000 classes ImageNet, entrée 224. Dans `external/organisateur/alif_aidge_sdk/examples/mobilenetv2` :

```sh
python -m model.fetch_model      # models/mobilenetv2-7.onnx
python -m model.prepare_onnx     # models/mobilenetv2_224.onnx : normalisation repliée, entrée dans [0, 1]
python -m model.export --target hp --verify --calib-dir <images> --calib-dir <autres images>   # -> export_hp/
```

- `model.export` a besoin d'onnxruntime, absent de l'image Docker.
- `model.prepare_onnx` n'importe pas Aidge (seulement onnx, numpy et onnxruntime) : il devrait donc tourner sur
  un Mac (pas essayé).
- Le README de l'exemple indique où trouver des images de calibration.

### La variante « animaux », qui tourne sur la carte

Le même réseau, réduit aux 398 classes animales d'ImageNet (indices 0 à 397), sans réentraînement :
`scripts/npu/sdk_mobilenetv2_animaux.py`.

- Le modèle n'a pas de classe « pas d'animal » : il nomme toujours un animal, même devant une table. Dans la chaîne
  du défi, c'est le détecteur qui décide s'il y a un animal.
- Il prend en entrée l'ONNX préparé de l'exemple.

Depuis un Mac :

```sh
S=external/organisateur/alif_aidge_sdk
scripts/npu/aidge_docker.sh push $S external/organisateur/aidge-tosa-fork \
    external/organisateur/aidge_export_ethos-fork data/samples/ datasets/coco8/
scripts/npu/aidge_docker.sh run sh -c 'F=/work/external/organisateur
  export PYTHONPATH=$F/aidge_export_ethos-fork:$F/aidge-tosa-fork:$F/alif_aidge_sdk:$F/alif_aidge_sdk/examples/mobilenetv2:$PYTHONPATH
  python scripts/npu/sdk_mobilenetv2_animaux.py --calib data/samples datasets/coco8/images --verify \
      --onnx external/organisateur/alif_aidge_sdk/examples/mobilenetv2/models/mobilenetv2_224.onnx'
E=exports/npu/sdk_mobilenetv2_animaux_hp
scripts/npu/aidge_docker.sh fetch $E/model_io.json $E/output/model_vela.tflite
```

Le modèle de la carte a été calibré sur 19 images : `data/samples/`, `datasets/coco8/images/` et les exemples du
kit d'Alif. Résultat de l'export :

- 66 opérateurs sur le NPU, aucun sur le CPU ;
- contrôle TOSA réussi ;
- estimation de Vela : 12,1 ms, 1,44 Mio de SRAM, 2,39 Mio de MRAM.

Le 2026-11-19, le script, relancé sur les mêmes images, a redonné le même `model_vela.tflite` octet pour octet.
Il affiche le top-3 flottant et int8 de chaque image. Quelques références int8, sur le carré central de l'image
d'origine :

| Image | Top-1 int8 | Remarque |
|---|---|---|
| `datasets/coco8/images/train/000000000034.jpg` (zèbre) | zebra 100 % | |
| `data/samples/test_wildlife_1mp.jpg` (renard) | red fox 98 % | |
| `data/samples/ecureuil-roux.jpg` | fox squirrel 16 % | classe la plus proche, faible confiance |
| `datasets/coco8/images/train/000000000025.jpg` (girafes) | impala 36 % | pas de classe girafe dans ImageNet |
| `datasets/coco8/images/val/000000000061.jpg` (éléphants) | American black bear 16 % | animaux trop petits dans l'image : c'est le rôle du détecteur |

### Nos modèles

Deux outils possibles :

- `scripts/npu/sdk_export_report.py` : prend des dossiers d'images de calibration et résume le résultat dans
  `result.json` ;
- `alif-aidge-export`, l'outil du SDK : prend une calibration en tableaux `.npy`, et `--params` pour les
  constantes du modèle.

Pour le détecteur, passer d'abord `scripts/npu/onnx_slices_to_split.py` sur l'ONNX. Résultats et pièges :
[notes/2026-11-17-aidge-vers-npu.md](../notes/2026-11-17-aidge-vers-npu.md).

## 3. Générer les sources C++

```sh
E=exports/npu/sdk_mobilenetv2_animaux_hp      # dossier d'export
python3 external/organisateur/alif_aidge_sdk/alif_aidge_sdk/gen_model.py --export-dir $E --out $E/generated
```

C'est ce que fait la commande `alif-aidge-gen-model` quand le SDK est installé. `gen_model.py` n'utilise que la
bibliothèque standard : il tourne donc sur un Mac sans Aidge.

Contrôler dans `$E/generated/model_io.hpp` :

- `kTargetSubsystem`, ici `"HP"` ;
- pour un classifieur, `params::kClassesCount`, ici 398.

## 4. Compiler le firmware

```sh
source scripts/board/env.sh
cmake -S external/organisateur/alif_aidge_sdk -B build/sdk_mobilenetv2_animaux_hp \
    -DALIF_EXAMPLE=mobilenetv2 -DALIF_SUBSYSTEM=HP -DTARGET_BOARD=AppKit-e7 \
    -DALIF_CAMERA_MODULE=MT9M114 -DMODEL_DIR=$PWD/$E/generated
cmake --build build/sdk_mobilenetv2_animaux_hp -j8    # -> build/sdk_mobilenetv2_animaux_hp/mobilenetv2.mram.bin
```

- **`-DALIF_CAMERA_MODULE=MT9M114` est indispensable.** Le SDK suppose une ARX3A0 sur l'AppKit, alors que notre
  carte porte une MT9M114. Vérifié le 2026-11-18 : avec cette option, l'image s'affiche.
- `ALIF_SUBSYSTEM` doit correspondre à la cible de l'export, ce que CMake vérifie à la configuration.
- Sans les sous-modules du SDK, ajouter les variables de dépendances du §1.
- Durée et taille, constatées sur un Mac Apple Silicon :
  - 82 s de compilation ;
  - binaire de 2,7 Mo pour 398 classes, 3,3 Mo pour 1000.
- L'éditeur de liens avertit `LOAD segment with RWX permissions` : sans conséquence constatée.
- Autres options :
  - `ACTIVATION_BUF_SZ` : l'arène, 2 Mio par défaut, dans SRAM1 ;
  - `LCD_ROTATE_180` et `LCD_MIRROR` : actifs par défaut ;
  - `CAMERA_FLIP_VERTICAL`.

## 5. Flasher

**Opération à faire valider** (voir la dernière section d'AGENTS.md) : elle écrit la MRAM et remplace
l'application en place. La première fois, c'est la démo d'usine qui est remplacée.

Avant de flasher :

- la carte est reliée par **J2 « PRG USB »** ;
- l'interrupteur J-Link est sur EN ;
- les cavaliers **J15 sont sur SEUART**.

```sh
source scripts/board/env.sh
SETOOLS="$ALIF_SETOOLS_ROOT" SE_PORT="$ALIF_SEUART_PORT" \
    bash external/organisateur/alif_aidge_sdk/tools/flash_mram.sh --cpu HP build/sdk_mobilenetv2_animaux_hp/mobilenetv2.mram.bin
```

Le script génère la table des matières (TOC) avec `app-gen-toc`, puis écrit l'image avec
`app-write-mram -p -b 55000`. Environ 75 s sous macOS.

Le script est écrit pour Linux. Sous macOS, il demande trois précautions :

- **donner `SE_PORT`**, sinon le script cherche `/dev/ttyACM*` ;
- **le port doit déjà être sur la ligne `comport`** de `$SETOOLS/isp_config_data.cfg`, écrite par `tools-config`.
  Sinon, d'après son code, le script la corrige avec un `sed -i` de syntaxe GNU, qui échoue sous macOS ;
- **le message `stat: illegal option -- c` est sans conséquence** : il manque seulement la taille annoncée.

Vérifier ensuite, en lecture seule :

```sh
(cd "$ALIF_SETOOLS_ROOT" && ./maintenance -c "$ALIF_SEUART_PORT" -opt gettoc getcpustatus)
```

Résultat attendu : une entrée `HP_APP` de la taille du `.mram.bin`, et `M55-HP` démarré à `0x80008000`.

- **Une seule image à la fois** : `--cpu HE` écrit à la même adresse et remplace l'image HP. Voir la question Q4
  dans [notes/questions-ouvertes.md](../notes/questions-ouvertes.md).
- **Pour changer de modèle**, flasher l'autre `.mram.bin` de la même façon.

## 6. Ce qu'on voit sur la carte

D'après le code de l'exemple (`examples/mobilenetv2/main.cpp`) :

- l'inférence tourne en boucle, sans appui sur le joystick ;
- le modèle voit le **carré central** de l'image de la caméra, réduit à sa taille d'entrée : viser le sujet au
  centre ;
- en haut de l'écran s'affichent :
  - les images par seconde (`FPS`), le temps `NPU` en ms et le cœur (`HP`) ;
  - puis les temps `CAM`, `PRE` et `DRAW` en ms ;
  - en dessous, le top-5 en pourcentages ;
- la console (UART2, 115 200 bauds) écrit le top-5 de chaque image. Pour la lire, passer J15 sur UART2, puis le
  remettre sur SEUART avant de flasher. Pas encore essayé.

**Mesures sur la carte** : pas encore relevées. Pour comparaison, les estimations de Vela pour le temps NPU sont
de 12,9 ms pour l'exemple à 1000 classes et de 12,1 ms pour la variante à 398 classes.

Pour essayer sans animal sous la main, on peut afficher des photos sur un écran face à la caméra. Les scores
différeront de ceux du PC à cause des reflets, du moiré et de l'éclairage.

## 7. Pièges

| Symptôme | Cause | Remède |
|---|---|---|
| `Could not find a version that satisfies the requirement aidge-core` | pas de wheel Aidge sous macOS | Linux ou Windows, ou Docker depuis un Mac (§1) |
| Vela refuse le modèle (`Output data type not supported`) | `aidge_export_ethos` officiel, sans `ethos_be` | branche `feat/ethos-be-lowering` du fork (§1) |
| L'export de YOLOv10 échoue sur un Slice | le fork convertit mal un Slice sur l'axe des canaux lors du passage en NHWC | `scripts/npu/onnx_slices_to_split.py` avant l'export |
| `Segmentation fault` pendant la quantification, dans l'image Docker | non élucidé ; observé une fois sur quatre exports identiques | relancer la même commande |
| CMake refuse le modèle à la configuration | modèle exporté pour l'autre cœur | réexporter avec la bonne `--target` |
| Opérateurs sur le CPU après Vela | le firmware n'a que l'opérateur Ethos-U | modifier le modèle jusqu'à `CPU operators = 0` |
| Carte non détectée | port USB, câble ou interrupteur J-Link | J2 PRG USB, câble de données, J-Link sur EN |
| Caractères illisibles sur le port série | mauvais UART ou mauvais débit | J15 sur SEUART : 55 000 bauds (SETOOLS) ; sur UART2 : console à 115 200 bauds |

## 8. Et ensuite

- **Relever le FPS et le temps NPU** sur la carte, et les comparer à Vela.
- **Notre classifieur (Q2, Q3).** D'après `main.cpp`, le firmware de l'exemple accepte tout classifieur qui
  remplit trois conditions :
  - une entrée carrée RGB ;
  - une seule sortie de N logits, que `prepare_onnx` met en 4-D (`[1, N, 1, 1]`) pour MobileNetV2 ;
  - les constantes `classes` et `top_k`, fournies par l'export (`--params` d'`alif-aidge-export`).

  Pas encore essayé.
- **Le détecteur**, dont le décodeur YOLOv10 est à écrire en C++. La référence Python est
  `tools/u55/yolov10_decode.py`. Le dépôt `alif_yolo_fastest` du porteur montre comment organiser un tel décodeur.
- **Le détecteur sur HP et le classifieur sur HE** : question Q4.
