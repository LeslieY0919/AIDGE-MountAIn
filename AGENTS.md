# AGENTS.md — AIDGE-MountAIn (wildlife monitoring)

## Langue

- Tout ce qui est écrit dans le dépôt est en **français** : documentation, commentaires de code, messages de commit,
  descriptions de PR, rapports, notes de conception et plans. Les membres du groupe sont francophones.
- Restent tels quels : identifiants de code, commandes, chemins, noms de fichiers, sorties de terminal et noms
  propres (J-Link, SETOOLS, Ethos-U55, etc.).
- Dans la conversation, répondre dans la langue de la personne qui écrit.

## Présentation du projet

Sujet n° 3 du Challenge Aidge 2026, « AIDGE for Wildlife Monitoring » (proposé par MountAIn) : détecter les
animaux sauvages et identifier l'espèce sur Cortex-M55 + Ethos-U55, la caméra capturant l'image et l'inférence
s'exécutant localement sur le NPU. Le matériel du challenge envoyé par les organisateurs est une AppKit Alif
Ensemble E7 (avec module caméra MT9M114 et écran) ; l'IBEX CORE cité dans la plaquette est le produit de MountAIn.
Les livrables sont publiés sous CC BY 4.0, à rendre avant le 2026-12-31, soutenance le 2027-01-14
([plaquette](https://deepgreen.ai/storage/PDF/Plaquette_ChallengeAidge.pdf),
[règlement](https://deepgreen.ai/storage/PDF/Reglement-ChallengeAidge.pdf)).

## Organisation du dépôt

- L'arborescence et la chaîne de production des modèles sont décrites dans [README.md](README.md).
- Les scripts Python se lancent depuis la racine du dépôt : leurs chemins (modèles, `class_names.json`,
  `samples/`, dossiers d'export) sont relatifs à la racine.
- Ne pas versionner ce qui se régénère : modèles (`*.onnx`, `*.pt`, `*.tflite`), jeux de données, exports C++
  d'Aidge (`export_cpp_*/`, `export_tmp*/`), dossiers `build/`. Les modèles se partagent via les GitHub Releases.
- `archive/` conserve les explorations et anciennes versions pour mémoire ; ne pas les modifier, ne pas en
  dépendre.
- Les réglages propres à une machine vont dans `scripts/env.local.sh` et `CLAUDE.local.md`, non versionnés.

## Matériel

| Élément | Valeur |
|---|---|
| Carte | Alif Ensemble E7 AI/ML AppKit (AK-E7-AIML), HW rev D3 |
| Puce | AE722F80F55D5LS, revision B4 |
| Cœurs | M55-HP + Ethos-U55 (256 MAC), M55-HE + Ethos-U55 (128 MAC), 2× Cortex-A32 |
| Mémoire | 13.5 MB SRAM, 5.5 MB MRAM |
| Caméra | module MT9M114 (AP-Vision-MT9M114-105), MIPI CSI-2 ; capture par défaut du kit : 1280×720 RGB565 |
| Écran | LCD 480×800, RGB565 |
| Sonde | J-Link OB embarquée (SEGGER) |
| Port série | VCOM de la J-Link → SEUART (cavaliers J15 par défaut) ; sous macOS `/dev/cu.usbmodem*` |

- **La carte doit être reliée à l'ordinateur par le port J2 « PRG USB ».** J1 « SOC USB » ne fait qu'alimenter
  la carte : avec le firmware d'usine, l'ordinateur ne détecte aucun périphérique.
- Si la carte n'est pas détectée, vérifier trois points : câble branché sur PRG USB, câble de données (pas
  seulement de charge), interrupteur J-Link EN-DIS sur EN.
- Noms de cible J-Link : `AE722F80F55D5_HP`, `AE722F80F55D5_HE`, `AE722F80F55D5_A32_0`, `AE722F80F55D5_A32_1`.
- Guide de référence : Alif AUGD0020, *AI/ML AppKit w/J-Link On-Board User Guide*.

## Environnement de la carte

Avant d'utiliser les outils de la carte, charger les variables d'environnement :

```sh
source scripts/env.sh
```

`scripts/env.sh` suppose une installation sous macOS dans `~/opt` ; chaque machine peut redéfinir les chemins
dans `scripts/env.local.sh`.

| Outil | Version | Emplacement par défaut |
|---|---|---|
| SEGGER J-Link | V9.84 | `~/opt/SEGGER/JLink_V984` |
| Alif SETOOLS | 1.112.00 (macOS) | `~/opt/alif/app-release-exec-macos` |
| Arm GNU Toolchain | 13.3.rel1 | `~/opt/arm/arm-gnu-toolchain-13.3.rel1-darwin-arm64-arm-none-eabi` |
| Kit d'évaluation ML Alif | main @ `45192b3` | `external/alif_ml-embedded-evaluation-kit` |
| Python (venv du kit) | 3.10.11 | `external/alif_ml-embedded-evaluation-kit/resources_downloaded/env` |
| Vela / CMake (venv du kit) | 5.0.0 / 3.30.5 | idem |
| Profilage U55 (`u55/`) | Python 3.10.11 : Ultralytics 8.4.175, PyTorch 2.14.1, TensorFlow 2.19.0, onnx2tf 1.28.8 | `.venv-yolo` |

- SETOOLS doit être lancé depuis son dossier racine, car les outils utilisent des chemins relatifs. Il doit être
  configuré pour E7 (AE722F80F55D5LS) rev B4, avec le port série dans `isp_config_data.cfg`.
- Lecture seule de l'état de la puce :
  `cd "$ALIF_SETOOLS_ROOT" && ./maintenance -c "$ALIF_SEUART_PORT" -opt getrevision getbanner gettoc getcpustatus`
- Le kit d'évaluation ML d'Alif n'a été testé qu'avec GCC 12.3 et 13.3.rel1 : ne pas passer à une version plus
  récente.
- CMake 4 ne prend plus en charge `cmake_minimum_required` inférieur à 3.5 ; `scripts/env.sh` définit
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` pour les anciens sous-modules.
- Dans `.venv-yolo`, numpy est figé à 1.26.4 : TensorFlow 2.19 exige numpy < 2.2 et Ultralytics exclut numpy
  2.0–2.3.4 sous macOS. Désactiver l'envoi de statistiques d'Ultralytics (`yolo settings sync=False`).

## Kit d'évaluation ML

- Clone superficiel dans `external/` ; les sous-modules sont récupérés au minimum, **sans
  `dependencies/executorch`** (le projet n'utilise que TFLM).
- Les ressources ne sont préparées que pour `img_class` et `object_detection` (détection de visages d'usine),
  avec la liste réduite `scripts/mlek_requirements_tflm.txt`, sans torch ni executorch :
  `cd "$MLEK_ROOT" && python3.10 set_up_default_resources.py --ml-frameworks tflm --use-case img_class --use-case object_detection --requirements-file ../../scripts/mlek_requirements_tflm.txt`
- Compiler et générer le paquet ATOC (sur l'ordinateur uniquement, rien n'est écrit sur la carte) :
  `scripts/build_img_class.sh`.
  Pour un autre modèle, définir `MODEL_PATH` (modèle compilé par Vela pour ethos-u55-256), `LABELS_FILE` et
  `IMAGE_SIZE`.
- Résultats : `build/mlek_img_class_hp/bin/mlek_alif_img_class.axf`, dans SETOOLS
  `build/images/mlek_alif_img_class.bin` et `build/AppTocPackage.bin`.
- L'application tourne sur le M55-HP et écrit ses journaux sur UART2 ; pour les voir sur l'ordinateur, passer les
  cavaliers J15 de SEUART à UART2 (et revenir à SEUART pour flasher).
- La démo de détection de visages d'usine peut être recompilée avec le cas d'usage `object_detection` du kit.

## Modèles

### Pipeline actuel

Deux étapes : un détecteur trouve les animaux sur l'image entière réduite, puis chaque boîte est découpée dans
l'image d'origine et confiée à un classifieur d'espèces.

- Détecteur : MegaDetector v6 `MDV6-yolov10-c` (classes animal / personne / véhicule), affiné en 640 par
  `training/train_mdv6.py`, avec l'attention du PSA neutralisée (`model.10.attn` remplacé par `Identity`).
- Classifieur : entrée 128, 5 espèces (`class_names.json` : blaireau, chevreuil, écureuil, renard, sanglier),
  entraîné par `training/train_classifier_gpu.py`.
- Export : quantification INT8 et export C++ par Aidge (`aidge_export_cpp`), exécuté pour l'instant sur CPU Linux
  (`standalone_wildlife_cpp/`). Il reste à le porter sur la carte (cible Aidge `ake7`, NPU).

Pourquoi deux étapes : avec une entrée de 320, l'image caméra 1280×720 est réduite 4 fois et un animal lointain ne
garde que quelques dizaines de pixels ; le classifieur travaille sur la découpe de l'image d'origine, avec 4 fois
plus de détails, pour environ 2 à 3 ms (Vela) de plus par animal. Un modèle unique qui monterait en résolution ne
gagnerait que 2 fois plus de détails en passant à 640, pour 3,9 fois le temps de 320. À valider : comparer sur le
jeu de validation avec un détecteur qui sort directement l'espèce, et ne garder la deuxième étape que si elle
apporte un gain net.

### Contraintes de l'Ethos-U55 (mesures Vela sur YOLOv10n)

Les mesures ci-dessous portent sur YOLOv10n (même famille que `MDV6-yolov10-c`) ; `MDV6-yolov10-c` reste à
profiler de la même façon avec `u55/profile_yolov10_u55.py`.

- Le graphe s'arrête aux 6 sorties int8 brutes de la branche one-to-one ; seuillage, DFL et décodage des boîtes
  se font sur le M55 (ni l'U55 ni TFLM n'ont TOPK_V2). Implémentation de référence : `u55/yolov10_decode.py`.
- SiLU → ReLU : SiLU (table + multiplication) prend environ 45 % du temps NPU, alors que ReLU se fusionne dans la
  convolution. Nécessite un réentraînement.
- Retrait du PSA : l'U55 émule MatMul par des multiplications élément par élément ; à 320 le PSA prend environ
  30 % du temps, à 416 le pic SRAM atteint 9,5 Mio.
- Prochaine optimisation : après ces changements, Concat prend environ 25 % du temps NPU (38 MemoryCopy dans les
  C2f).
- L'image caméra est en 16:9 : le détecteur peut prendre une entrée non carrée (par exemple 512×288) pour ne pas
  dépenser de calcul sur les bandes de remplissage (pas encore mesuré).

Estimations Vela (poids en MRAM ; les variantes modifiées ne sont pas réentraînées, seules la vitesse et la
mémoire comptent). HP = M55-HP + U55-256 à 400 MHz, HE = M55-HE + U55-128 à 160 MHz :

| Modèle | Entrée | HP | HE | Pic SRAM | MRAM |
|---|---|---|---|---|---|
| YOLOv10n d'origine | 256 / 320 | 30.3 / 48.2 ms | – | 1.48 / 5.26 MiB | 2.03 MiB |
| YOLOv10n ReLU sans PSA | 256 / 320 / 416 | 10.8 / 16.1 / 27.5 ms | 41.1 / 63.6 / – ms | 0.44 / 0.69 / 1.16 MiB | 1.78 MiB |
| YOLOv10n ReLU sans PSA | 480 / 512 / 640 | 36.4 / 40.6 / 63.3 ms | – | 1.54 / 1.76 / 2.54 MiB | 1.78 MiB |
| MobileNetV2-0.35 | 128 | 1.6 ms | 4.8 ms | 0.25 MiB | 0.50 MiB |
| MobileNetV2-0.5 | 160 | 2.8 ms | 9.2 ms | 0.39 MiB | 0.77 MiB |

- Les mesures publiées par Alif sur carte valent environ 2 fois les estimations de Vela ; sur la carte, se fier
  aux compteurs PMU.
- Pour régénérer (résultats et CSV par couche dans `build/ml_profile/`) :
  `.venv-yolo/bin/python u55/profile_yolov10_u55.py --imgsz 256 320` (résultats fusionnés par nom de modèle),
  `.venv-yolo/bin/python u55/profile_classifier_u55.py` ; comparaison HP–HE :
  `.venv-yolo/bin/python u55/compare_hp_he.py <*.tflite>...`

### Répartition du calcul et mémoire

- M55-HP + U55-256 : tout le pipeline (capture, prétraitement, les deux modèles, décodage, affichage). Les deux
  modèles s'exécutent l'un après l'autre et partagent la mémoire des tenseurs.
- M55-HE + U55-128 : pas utilisé dans la première version. Le même modèle y est environ 3 à 4 fois plus lent que
  sur le HP (Vela) ; il pourra plus tard servir de réveil basse consommation permanent (détection de mouvement ou
  très petit modèle) qui réveille le HP quand quelque chose bouge.
- 2× Cortex-A32 : non utilisés.
- Pas de découpage d'un modèle entre HP et HE : le découpage bicœur actuel d'Aidge passe uniquement par CMSIS-NN
  (sans NPU), les deux cœurs s'exécutent l'un après l'autre (aucun gain en parallèle) et la frontière ne peut
  porter qu'un seul tenseur d'au plus 4 Ko.
- Budget mémoire : sur les 5,5 Mo de MRAM, environ 0,9 Mo de programme (mesuré sur img_class du kit), 1,8 Mo pour
  le détecteur et 0,5 à 0,8 Mo pour le classifieur, il reste environ 2 Mo ; sur les 13,5 Mo de SRAM, 1,76 Mio
  pour l'image caméra, 0,73 Mio pour le LCD, 0,29 Mio de tampon RGB et 0,7 à 1,8 Mio de zone de tenseurs (selon la
  taille d'entrée, partagée par les deux modèles), soit environ 5 Mio au total. Chiffres établis avec YOLOv10n : à
  revérifier pour `MDV6-yolov10-c` en 640.

## Aidge

- Le déploiement final doit passer par Eclipse Aidge ; TFLM + Vela ne sert que de référence pour la précision, la
  vitesse et l'analyse par couche.
- Pour cette carte, la cible Aidge s'appelle `ake7` ; le chemin NPU est Aidge → TOSA (`aidge_tosa`, pré-alpha) →
  Vela. En int8, `aidge_tosa` ne gère pour l'instant que Conv, DWConv, pooling (sans padding), FC, ReLU, Add,
  Reshape et Transpose ; il manque Sigmoid/Mul, Concat, Resize, Split/Slice, Pad, Softmax et MatMul.
- Export hybride (export Cortex-M avec `--npu`) : les plus grands sous-graphes que Vela sait compiler vont sur le
  NPU, le reste reste sur le CPU (CMSIS-NN, etc.) ; le choix se fait à l'export, sans repli à l'exécution. Chaque
  région NPU n'accepte pour l'instant qu'une entrée int8 et une sortie int8.
- Le classifieur MobileNetV2 fait partie des réseaux déjà testés par `aidge_tosa` : il peut passer entièrement sur
  le NPU dès maintenant. YOLOv10n ReLU sans PSA en 320 compte 182 opérateurs, dont 34 que `aidge_tosa` ne gère pas
  encore : 16 Slice, 13 Concat, 3 MaxPool avec padding (SPPF) et 2 Resize ; il a en outre 6 sorties. Le frontal
  TOSA de Vela 5.0/5.1 gère Concat, Slice, Pad et le pooling, mais pas RESIZE ; le suréchantillonnage ×2 au plus
  proche voisin peut se réécrire en Reshape + Concat. Pour que tout le détecteur passe sur le NPU, il faut ajouter
  à `aidge_tosa` Concat, Slice et le pooling avec padding, et accepter plusieurs sorties par région NPU. Ce sont
  autant de contributions possibles à Aidge (`aidge_tosa` est développé surtout par MountAIn).
- Si l'on porte un jour le projet sur l'IBEX CORE (produit de MountAIn, peut-être basé sur un Alif Ensemble E3),
  le plan M55 + U55 reste valable, mais le budget mémoire est à revérifier.

### Aidge sous macOS

Aidge ne publie des wheels que pour Linux x86_64 et Windows. Sous macOS, l'export Aidge → TOSA → Vela tourne dans
l'image Docker `wildlife-aidge:0.10.1` (`aidge/Dockerfile` : Aidge 0.10.1, tosa-tools 2026.8.0 précompilé,
`aidge_tosa` et `aidge_export_arm_ethos` à des commits figés, Vela 5.1.0) sur une machine Linux x86_64 jointe en
SSH. Tout passe par `scripts/dev2_aidge.sh` (build / status / push / run / fetch) ; l'hôte se choisit avec
`DEV2_HOST`. Le firmware reste compilé et flashé sur la machine à laquelle la carte est branchée.

## Opérations à faire valider

Les opérations suivantes modifient la carte et doivent être validées par l'utilisateur avant exécution :

- `updateSystemPackage` : met à jour le firmware de sécurité (SE) de la puce ; impossible de revenir à une version
  antérieure. **Décision actuelle : pas de mise à jour**, on reste en SES v1.100.0. AUGD0005 v1.112 présente cette
  étape comme optionnelle et les changements de 1.112 ne concernent pas le développement ML sur E7 ; à reconsidérer
  seulement si un outil ou une fonction exige explicitement un firmware plus récent.
- `app-write-mram` : écrit une application dans la MRAM et remplace la démo d'usine. L'image n'est pas alignée sur
  16 octets, il faut ajouter `-p` : `cd "$ALIF_SETOOLS_ROOT" && ./app-write-mram -p`
- Tout effacement de la MRAM, toute écriture OTP, et toute opération touchant aux clés ou à l'état de cycle de vie
  (LCS).
