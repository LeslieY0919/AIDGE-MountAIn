# AGENTS.md — carte Alif E7 et Ethos-U55 : outillage et faits vérifiés

Complément de [CLAUDE.md](CLAUDE.md), qui porte le contexte du défi, les exigences et les décisions
d'architecture. L'organisation du dépôt est décrite dans [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Ce fichier ne contient que ce qui a été **installé, mesuré ou lu dans le code** avec la carte du groupe. Il ne
prend aucune décision : en cas de divergence, CLAUDE.md fait foi, et les écarts se discutent dans
[notes/](notes/README.md).

## Langue

- Tout ce qui est écrit dans le dépôt est en **français** : documentation, commentaires de code, messages de commit,
  descriptions de PR, rapports, notes de conception et plans. Les membres du groupe sont francophones.
- Restent tels quels : identifiants de code, commandes, chemins, noms de fichiers, sorties de terminal et noms
  propres (J-Link, SETOOLS, Ethos-U55, etc.).
- Dans la conversation, répondre dans la langue de la personne qui écrit.

## Échéances du challenge

Sujet n° 3 du Challenge Aidge 2026. Livrables (rapport et vidéo de démonstration, publiés sous CC BY 4.0) à rendre
avant le **2026-12-31**, soutenance le **2027-01-14**
([plaquette](https://deepgreen.ai/storage/PDF/Plaquette_ChallengeAidge.pdf),
[règlement](https://deepgreen.ai/storage/PDF/Reglement-ChallengeAidge.pdf)).

## Matériel

| Élément | Valeur | Source |
|---|---|---|
| Carte | Alif Ensemble E7 AI/ML AppKit (AK-E7-AIML), HW rev D3 | étiquette de la carte |
| Puce | AE722F80F55D5LS, revision B4, 13.5 MB SRAM, 5.5 MB MRAM | SETOOLS (`utils/global-cfg.db`) |
| Cœurs | M55-HP + Ethos-U55 (256 MAC), M55-HE + Ethos-U55 (128 MAC), 2× Cortex-A32 | documentation Alif |
| Caméra | module MT9M114 (AP-Vision-MT9M114-105), MIPI CSI-2 ; 1280×720 RGB565 dans le kit d'Alif | code du kit |
| Écran | LCD 480×800, RGB565 | code du kit |
| Sonde | J-Link OB embarquée (SEGGER) | – |
| Port série | un seul port (VCOM de la J-Link), relié à SEUART ou à UART2 selon les cavaliers J15 | guide AUGD0020 |

- **La carte doit être reliée à l'ordinateur par le port J2 « PRG USB ».** J1 « SOC USB » ne fait qu'alimenter
  la carte : avec le firmware d'usine, l'ordinateur ne détecte aucun périphérique.
- Si la carte n'est pas détectée, vérifier trois points : câble branché sur PRG USB, câble de données (pas
  seulement de charge), interrupteur J-Link EN-DIS sur EN.
- Sous macOS, cette carte fait apparaître **un seul** port `/dev/cu.usbmodem*`. Avec les cavaliers J15 par défaut,
  c'est l'UART du Secure Enclave : SETOOLS y dialogue à 55 000 bauds pour la revision B4
  (`utils/featuresDB.db`), et `connect.py --scan` y mesure une fenêtre de 55 000 à 60 000 bauds. Pour
  lire la console de l'application (UART2, 115 200 bauds dans le kit d'Alif), déplacer les cavaliers J15 de SEUART
  à UART2, puis les remettre sur SEUART avant de flasher.
- Noms de cible J-Link : `AE722F80F55D5_HP`, `AE722F80F55D5_HE`, `AE722F80F55D5_A32_0`, `AE722F80F55D5_A32_1`.
- Guide de référence : Alif AUGD0020, *AI/ML AppKit w/J-Link On-Board User Guide*.
- Le SDK du porteur suppose par défaut une caméra ARX3A0 sur l'AppKit (« The AppKit E7 ships with the ARX3A0 ») :
  pour notre MT9M114, compiler avec `-DALIF_CAMERA_MODULE=MT9M114`, comme CLAUDE.md §3. Vérifié le 2026-11-18 :
  avec cette option, l'image de la caméra s'affiche sur l'écran.

Les 8 Mo de SRAM et 4,5 Mo de MRAM de CLAUDE.md correspondent à ce que voit **un** cœur avec les scripts d'édition
de liens du kit d'Alif, que le SDK reprend à l'identique (`RTSS-HP.ld` : ITCM 256 Kio, DTCM 1 Mio, SRAM0 4 Mio,
SRAM1 2,5 Mio, MRAM 4,46 Mio). Les scripts HP et HE partent de la même adresse MRAM et utilisent les mêmes SRAM0
et SRAM1, et le SDK flashe une seule image à la fois : faire tourner deux binaires en même temps demande de
découper la mémoire.

## Environnement de la carte

Avant d'utiliser les outils de la carte, charger les variables d'environnement :

```sh
source scripts/board/env.sh
```

`scripts/board/env.sh` suppose une installation sous macOS dans `~/opt` ; chaque machine peut redéfinir les
chemins dans `scripts/board/env.local.sh` (non versionné). Il définit aussi `SETOOLS`, attendu par
`scripts/board/connect.py --setools`.

| Outil | Version | Emplacement par défaut |
|---|---|---|
| SEGGER J-Link | V9.84 | `~/opt/SEGGER/JLink_V984` |
| Alif SETOOLS | 1.112.00 (macOS) | `~/opt/alif/app-release-exec-macos` |
| Arm GNU Toolchain | 13.3.rel1 | `~/opt/arm/arm-gnu-toolchain-13.3.rel1-darwin-arm64-arm-none-eabi` |
| Kit d'évaluation ML Alif | main @ `45192b3` | `external/alif_ml-embedded-evaluation-kit` |
| Python (venv du kit) | 3.10.11 | `external/alif_ml-embedded-evaluation-kit/resources_downloaded/env` |
| Vela / CMake (venv du kit) | 5.0.0 / 3.30.5 | idem |
| Profilage U55 (`tools/u55/`) | Python 3.10.11 : Ultralytics 8.4.175, PyTorch 2.14.1, TensorFlow 2.19.0, onnx2tf 1.28.8 | `.venv-yolo` |

- SETOOLS doit être lancé depuis son dossier racine, car les outils utilisent des chemins relatifs. Il doit être
  configuré pour E7 (AE722F80F55D5LS) rev B4, avec le port série dans `isp_config_data.cfg`.
- Lecture seule de l'état de la puce :
  `cd "$ALIF_SETOOLS_ROOT" && ./maintenance -c "$ALIF_SEUART_PORT" -opt getrevision getbanner gettoc getcpustatus`
- Le kit d'évaluation ML d'Alif n'a été testé qu'avec GCC 12.3 et 13.3.rel1 : ne pas passer à une version plus
  récente.
- CMake 4 ne prend plus en charge `cmake_minimum_required` inférieur à 3.5 ; `scripts/board/env.sh` définit
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` pour les anciens sous-modules.
- Dans `.venv-yolo`, numpy est figé à 1.26.4 : TensorFlow 2.19 exige numpy < 2.2 et Ultralytics exclut numpy
  2.0–2.3.4 sous macOS. Désactiver l'envoi de statistiques d'Ultralytics (`yolo settings sync=False`). Y installer
  aussi le paquet du dépôt (`pip install -e .`), que les outils de `tools/u55/` importent.

## Kit d'évaluation ML d'Alif

Il s'agit du kit public d'Alif (`alif_ml-embedded-evaluation-kit`, TFLM + Ethos-U), utilisé ici pour vérifier la
carte et mesurer un modèle. **Ce n'est pas le SDK du porteur** (section suivante), qui en reprend les briques. Les
modèles du kit prennent `q = pixel − 128` en entrée, ceux du SDK non : ne pas mélanger les deux.

- Clone superficiel dans `external/` ; les sous-modules sont récupérés au minimum, **sans
  `dependencies/executorch`** (seul TFLM sert).
- Les ressources ne sont préparées que pour `img_class` et `object_detection` (détection de visages d'usine),
  avec la liste réduite `scripts/board/mlek_requirements_tflm.txt`, sans torch ni executorch :
  `cd "$MLEK_ROOT" && python3.10 set_up_default_resources.py --ml-frameworks tflm --use-case img_class --use-case object_detection --requirements-file ../../scripts/board/mlek_requirements_tflm.txt`
- Compiler et générer le paquet ATOC (sur l'ordinateur uniquement, rien n'est écrit sur la carte) :
  `scripts/board/build_img_class.sh` (ou `make board-firmware`).
  Pour un autre modèle, définir `MODEL_PATH` (modèle compilé par Vela pour ethos-u55-256), `LABELS_FILE` et
  `IMAGE_SIZE` ; `SHOW_INF_TIME=1` affiche le temps d'inférence sur l'écran.
- Résultats : `build/mlek_img_class_hp/bin/mlek_alif_img_class.axf`, dans SETOOLS
  `build/images/mlek_alif_img_class.bin` et `build/AppTocPackage.bin`.
- L'application tourne sur le M55-HP et écrit ses journaux sur UART2 (115 200 bauds).
- Limites de ses exemples pour notre chaîne : voir
  [notes/2026-11-17-contraintes-carte-et-kit.md](notes/2026-11-17-contraintes-carte-et-kit.md).

## SDK du porteur (`alif_aidge_sdk`)

Lu dans le code (`eb35a81`). Son exemple MobileNetV2 a été exporté, compilé sous macOS et flashé sur la carte le
2026-11-18 : marche à suivre dans [docs/GUIDE_CARTE.md](docs/GUIDE_CARTE.md). Détails du code dans
[notes/2026-11-17-contraintes-carte-et-kit.md](notes/2026-11-17-contraintes-carte-et-kit.md#ce-que-fait-le-sdk-du-porteur).

- **Installation** : celle de CLAUDE.md §3, à une différence près pour le détecteur.
  - Le SDK utilise l'option `ethos_be` d'`aidge_export_arm_ethos` quand elle existe : elle convertit les îlots
    flottants (Softmax, Sigmoid) en opérateurs entiers.
  - Cette option n'existe que sur la branche `feat/ethos-be-lowering` du fork du porteur
    (`https://gitlab.eclipse.org/olivierpds/aidge_export_ethos.git`). Le dépôt officiel ne l'a pas, et sans elle
    Vela refuse YOLOv10n.
- **Convention d'entrée** :
  - l'ONNX prend une image dans [0, 1], en NCHW ;
  - le graphe exporté prend des codes int8 en NHWC, `code = round(pixel / 255 × 127)`, zéro 0 ;
  - `model_io.json` donne une échelle par sortie.
- **Firmware** :
  - le modèle doit passer à 100 % sur le NPU : TFLM n'enregistre que l'opérateur Ethos-U ;
  - l'arène fait 2 Mio par défaut, dans SRAM1 (`ACTIVATION_BUF_SZ`) ;
  - la console est sur UART2 à 115 200 bauds, à lire en passant J15 sur UART2.
- **Flash** : `tools/flash_mram.sh --cpu HP|HE` appelle `app-write-mram -p -b 55000` (opération à faire valider,
  voir la dernière section). L'image HP et l'image HE vont à la même adresse : la seconde remplace la première.
  Sous macOS, le script marche si l'on donne `SE_PORT` et que ce port figure déjà dans `isp_config_data.cfg` de
  SETOOLS (environ 75 s) ; détails dans le guide.

## Mesures Vela sur l'Ethos-U55

Estimations de Vela, pas des mesures sur carte ; les mesures publiées par Alif sur carte valent environ 2 fois ces
estimations. Le tableau ci-dessous passe par la conversion TFLite (onnx2tf) ; les chiffres par Aidge sont dans la
section suivante. Les mesures portent sur **YOLOv10n**, de la même famille que `MDV6-yolov10-c`, qui reste à
profiler de la même façon. Poids en MRAM. HP = M55-HP + U55-256 à 400 MHz, HE = M55-HE + U55-128 à 160 MHz. Les
variantes modifiées ne sont pas réentraînées : seules la vitesse et la mémoire comptent.

| Modèle | Entrée | HP | HE | Pic SRAM | MRAM |
|---|---|---|---|---|---|
| YOLOv10n d'origine | 256 / 320 | 30.3 / 48.2 ms | – | 1.48 / 5.26 MiB | 2.03 MiB |
| YOLOv10n ReLU sans PSA | 256 / 320 / 416 | 10.8 / 16.1 / 27.5 ms | 41.1 / 63.6 / – ms | 0.44 / 0.69 / 1.16 MiB | 1.78 MiB |
| YOLOv10n ReLU sans PSA | 480 / 512 / 640 | 36.4 / 40.6 / 63.3 ms | – | 1.54 / 1.76 / 2.54 MiB | 1.78 MiB |
| MobileNetV2-0.35 | 128 | 1.6 ms | 4.8 ms | 0.25 MiB | 0.50 MiB |
| MobileNetV2-0.5 | 128 | 2.1 ms | – | 0.25 MiB | 0.77 MiB |
| MobileNetV2-0.5 | 160 | 2.8 ms | 9.2 ms | 0.39 MiB | 0.77 MiB |

Ce que montrent les rapports par couche :

- Le graphe mesuré s'arrête aux 6 sorties int8 brutes de la branche one-to-one ; seuillage, DFL et décodage des
  boîtes se font sur le M55 (ni l'U55 ni TFLM n'ont TOPK_V2). Référence Python de ce décodage sur les sorties
  TFLite : `tools/u55/yolov10_decode.py`.
- SiLU (table + multiplication) prend environ 45 % du temps NPU, alors que ReLU se fusionne dans la convolution.
- Le PSA : l'U55 émule MatMul par des multiplications élément par élément ; à 320 il prend environ 30 % du temps,
  à 416 le pic SRAM atteint 9,5 Mio.
- Sans SiLU ni PSA, Concat prend environ 25 % du temps NPU (38 MemoryCopy dans les C2f).
- Le même modèle est environ 3 à 4 fois plus lent sur le HE que sur le HP.
- L'image caméra est en 16:9 : une entrée non carrée (par exemple 512×288) éviterait de calculer sur les bandes
  de remplissage (pas encore mesuré).

Remplacer SiLU ou retirer le PSA demande un réentraînement, alors que CLAUDE.md demande de ne pas réentraîner
MegaDetector : ces chiffres sont des données pour la discussion, pas une décision.

Pour régénérer (résultats et CSV par couche dans `artifacts/u55_profile/`) :

```sh
.venv-yolo/bin/python tools/u55/profile_yolov10_u55.py --imgsz 256 320   # résultats fusionnés par nom de modèle
.venv-yolo/bin/python tools/u55/profile_classifier_u55.py
.venv-yolo/bin/python tools/u55/compare_hp_he.py <*.tflite>...           # comparaison HP–HE
```

## Aidge → Ethos-U55

Estimations de Vela, cœur HP, chaîne du SDK avec le fork `aidge-tosa-fork` et la branche `feat/ethos-be-lowering`
d'`aidge_export_ethos`. Résultats, contrôles et pièges dans
[notes/2026-11-17-aidge-vers-npu.md](notes/2026-11-17-aidge-vers-npu.md).

| Modèle | Entrée | NPU / CPU | Temps | Pic SRAM | MRAM | Par TFLite |
|---|---|---|---|---|---|---|
| MobileNetV2-1.0 | 224 | 68 / 0 | 13.1 ms | 1.44 MiB | 2.98 MiB | 10.7 ms, 1.44 MiB |
| YOLOv10n d'origine (SiLU + PSA) | 320 | 344 / 0 | 54.3 ms | 0.79 MiB | 1.62 MiB | 48.2 ms, 5.26 MiB |
| MobileNetV2-1.0, 398 classes animales (sur la carte) | 224 | 66 / 0 | 12.1 ms | 1.44 MiB | 2.39 MiB | – |

- `scripts/npu/sdk_export_report.py` : export par `alif_aidge_sdk.export.export_model`, puis résumé
  (`result.json` : estimation de Vela, corrélation flottant / int8 par sortie, contrôle TOSA).
- `scripts/npu/onnx_slices_to_split.py` : à passer sur le détecteur avant l'export. Dans notre ONNX de YOLOv10n,
  les C2f séparent leurs canaux par des Slice, que le fork convertit mal en NHWC ; le Split équivalent passe.
- `scripts/npu/sdk_mobilenetv2_animaux.py` : le MobileNetV2 de l'exemple du SDK réduit aux 398 classes animales
  d'ImageNet, sans réentraînement. C'est le modèle qui tourne sur la carte
  ([docs/GUIDE_CARTE.md](docs/GUIDE_CARTE.md)).
- Sur YOLOv10n, la quantification abîme surtout les sorties de classes (corrélation 0,56 à 0,86 avec le
  flottant, contre 0,99 pour les boîtes), sur 10 images de calibration seulement.
- Le fork gère, en plus de l'`aidge_tosa` officiel, Concat, Slice, Split, Resize, MaxPool avec padding, Sigmoid,
  Softmax et MatMul. Il retire aussi les Transpose que l'officiel place autour de chaque convolution.
- L'`aidge_tosa` **officiel** (celui de l'image `docker/aidge/`) ne convient pas à la carte. MobileNetV2 y donne
  77 ms et 9,2 Mio de SRAM en NCHW, et Vela le refuse en NHWC ; le détecteur ne passe pas.
- `scripts/npu/export_classifier.py` est notre premier script d'export. Il suit la convention du kit d'Alif
  (`q = pixel − 128`) et ne sert qu'avec `scripts/board/build_img_class.sh`. Avec le fork, il donne en NCHW le
  même résultat que le SDK (71 opérateurs, 13,2 ms) ; en NHWC, il plante.
- Autre chemin d'Aidge, non essayé : l'export hybride Cortex-M avec `--npu` (cible `ake7`), où les sous-graphes
  que Vela sait compiler vont sur le NPU et le reste sur le CPU (CMSIS-NN), sans TFLM.

### Aidge sous macOS

Aidge ne publie des wheels que pour Linux x86_64 et Windows. Sous macOS, l'export Aidge → TOSA → Vela tourne dans
l'image Docker `wildlife-aidge:0.10.1` (`docker/aidge/Dockerfile` : Aidge 0.10.1, tosa-tools 2026.8.0 précompilé,
`aidge_tosa` et `aidge_export_arm_ethos` à des commits figés, Vela 5.1.0) sur une machine Linux x86_64 jointe en
SSH. Tout passe par `scripts/npu/aidge_docker.sh` (build / status / push / run / fetch) ; l'hôte se choisit avec
`DEV2_HOST`. Le firmware reste compilé et flashé sur la machine à laquelle la carte est branchée.

L'image n'a ni les outils du porteur ni onnxruntime. Pour la chaîne du SDK, sans reconstruire l'image :

- cloner les dépôts du porteur dans `external/organisateur/` (non versionné) ;
- les envoyer avec `push` ;
- les placer en tête du `PYTHONPATH` dans `run`.

Les commandes sont dans [notes/2026-11-17-aidge-vers-npu.md](notes/2026-11-17-aidge-vers-npu.md#reproduire).
Faute d'onnxruntime, `sdk_export_report.py` et `sdk_mobilenetv2_animaux.py` calculent les sorties flottantes de
référence avec Aidge. Le fork fixe `tosa-tools==2026.5.0` ; l'image a la 2026.8.0, sans problème constaté. Une fois
sur quatre exports identiques (2026-11-19), Python s'est arrêté sur `Segmentation fault` pendant la quantification ;
la même commande, relancée, a abouti.

## Opérations à faire valider

Les opérations suivantes modifient la carte et doivent être validées par l'utilisateur avant exécution :

- `updateSystemPackage` : met à jour le firmware de sécurité (SE) de la puce ; impossible de revenir à une version
  antérieure. **Décision actuelle : pas de mise à jour**, on reste en SES v1.100.0. AUGD0005 v1.112 présente cette
  étape comme optionnelle et les changements de 1.112 ne concernent pas le développement ML sur E7 ; à reconsidérer
  seulement si un outil ou une fonction exige explicitement un firmware plus récent.
- `app-write-mram` (et tout script qui l'appelle, comme `flash_mram.sh` du SDK) : écrit une application dans la
  MRAM et remplace la démo d'usine. L'image n'est pas alignée sur 16 octets, il faut ajouter `-p` :
  `cd "$ALIF_SETOOLS_ROOT" && ./app-write-mram -p`
- Tout effacement de la MRAM, toute écriture OTP, et toute opération touchant aux clés ou à l'état de cycle de vie
  (LCS).
