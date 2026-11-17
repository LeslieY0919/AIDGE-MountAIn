# AIDGE MountAIn — contexte projet

> **Contexte projet.** Autoportant : aucun autre document n'est nécessaire pour démarrer.
> Pour l'organisation du dépôt, voir [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
>
> Deux conventions pour la suite de ce fichier :
> - **[SOURCÉ]** = tiré des slides du kick-off, de la réunion de lancement ou des README des dépôts. Fiable.
> - **[POSÉ]** = hypothèse de dimensionnement de l'équipe, à confirmer sur la carte. **Ne pas traiter comme un fait.**

---

## 1. Le projet en un paragraphe

Challenge AIDGE 2026, défi **MountAIn** (porteur : MountainEdge, contact `olivier@mountainedge.ai`).
Il s'agit de faire tourner une **chaîne de détection puis d'identification d'animaux sauvages** sur le kit
de développement **Alif Ensemble E7 AppKit**, en utilisant la chaîne outil **Eclipse AIDGE**.

La chaîne de traitement imposée [SOURCÉ] :

```
image caméra → REDIMENSIONNER → DÉTECTER (MegaDetector) → DÉCOUPER → CLASSIFIER (notre modèle) → « Chamois »
```

Le cœur de la difficulté est explicite dans les slides : pour détecter un animal, il faut une **grande**
image ; or le microcontrôleur a **peu de mémoire**. Un animal à 30 m ne fait que quelques dizaines de
pixels — si on redimensionne trop, le modèle ne trouve plus rien. Tout le travail consiste à garder le
maximum de résolution dans le budget mémoire disponible.

Livrable final : l'application tournant sur l'AppKit. MountainEdge la portera ensuite sur ses caméras
autonomes en énergie déployées en nature.

---

## 2. Décisions d'architecture déjà prises

Ces décisions sont **arrêtées** et formalisées dans un modèle MBSE (voir §8). Les respecter, ou les
renégocier explicitement avec l'équipe — mais ne pas les contourner silencieusement.

### Répartition sur les deux domaines de calcul

L'Ensemble E7 a deux domaines Cortex-M55, chacun avec son NPU Ethos-U55 [SOURCÉ] :

| Domaine | CPU | NPU | Rôle décidé |
|---|---|---|---|
| **HP** (high performance) | M55 @ 400 MHz | Ethos-U55-**256** | le **détecteur** (gros modèle) |
| **HE** (high efficiency) | M55 @ 160 MHz | Ethos-U55-**128** | le **classifieur** (petit modèle) |

Les deux domaines communiquent par une **zone de mémoire partagée** qui transporte les boîtes
englobantes du HP vers le HE [POSÉ — c'est une des questions ouvertes du défi ; des exemples de deux
modèles en parallèle existent sur le GitHub d'Alif].

**Un binaire par domaine**, à programmer séparément (`--cpu HP` puis `--cpu HE`) [SOURCÉ].

### Les deux modèles

| | Détecteur | Classifieur |
|---|---|---|
| Modèle | **MDv6-YOLOv10-c** (MegaDetector) [SOURCÉ] | **MobileNetV2** [décision d'équipe] |
| Origine | pré-entraîné, **à ne pas réentraîner** [SOURCÉ] | **à entraîner** par l'équipe [SOURCÉ] |
| Entrée | 320 × 320 px [POSÉ, objectif] | 96 × 96 px [POSÉ] |
| Licence | AGPL v3 (Ultralytics) [SOURCÉ] | selon le dataset choisi |

Le choix du classifieur a été arbitré sur 6 critères pondérés : **MobileNetV2 (57 pts)** devant
ResNet-18 (47 pts, écarté par ses ~11 Mo de poids, hors budget MRAM) et un CNN dédié (45 pts, trop
coûteux à entraîner dans le temps imparti). MobileNetV2 et ResNet sont **déjà disponibles dans
AIDGE** [SOURCÉ], ce qui en fait un bon point de départ.

Le dataset du classifieur est un **choix libre** [SOURCÉ] — faune de montagne, jungle, Amérique du
Nord, peu importe. **Vérifier les licences** : beaucoup de jeux de données d'animaux sur le web ne
sont pas libres.

### Budget mémoire — le contrat le plus contraignant

L'E7 offre **~8 Mo de SRAM** et **4,5 Mo de MRAM** (non volatile, type flash) [SOURCÉ].
Repère donné en réunion : une image 1 Mpx en RGB, c'est 3 Mo — **la moitié de la SRAM** d'un seul coup.

| Poste | Budget | |
|---|---|---|
| **SRAM totale** | **≤ 8 Mo** | [SOURCÉ] |
| ├ tampon d'image capteur | ≤ 1,5 Mo | [POSÉ] |
| ├ arène de travail du détecteur | ≤ 2,5 Mo | [POSÉ] → `ACTIVATION_BUF_SZ` du build HP |
| └ arène de travail du classifieur | ≤ 512 Ko | [POSÉ] → `ACTIVATION_BUF_SZ` du build HE |
| **MRAM totale** | **≤ 4,5 Mo** | [SOURCÉ] |
| ├ détecteur quantifié | ≤ 3 Mo | [POSÉ] |
| └ classifieur quantifié | ≤ 1 Mo | [POSÉ] |

Piste donnée en réunion [SOURCÉ] : **reconfigurer la caméra** pour qu'elle envoie directement des
images plus petites, plutôt que de capturer en pleine résolution puis réduire. La taille mémoire
décroît naturellement au fil de la chaîne (image → boîtes → imagettes).

### Post-traitement des détections

| Paramètre | Valeur | |
|---|---|---|
| Seuil de confiance | 0,25 | [POSÉ — valeur usuelle YOLO] |
| Seuil IoU du NMS | 0,45 | [POSÉ — valeur usuelle YOLO] |
| Détections retenues par image | **5 au maximum** | [POSÉ] → cf. `DETECTION_MAX_DET` |

Borner le nombre de détections est une **suggestion explicite du porteur** [SOURCÉ] : chaque imagette
consomme de la SRAM. S'il y a plus d'animaux, soit on n'en garde qu'un nombre fixé, soit on les traite
**séquentiellement**.

### Objectifs chiffrés de performance

| | Cible | |
|---|---|---|
| Latence de la chaîne complète | ≤ 1 000 ms / image | [POSÉ] |
| Cadence | ≥ 1 image / s | [POSÉ] |
| Rappel du détecteur | ≥ 70 % | [POSÉ — MDv6 le plus compact annonce 76 % de rappel et 87 % de mAP [SOURCÉ]] |
| Top-1 du classifieur | ≥ 80 % | [POSÉ] |
| Perte de mAP due à l'int8 | ≤ 5 points | [POSÉ] |
| Taille apparente minimale détectable | 24 × 24 px | [POSÉ] |
| Consommation moyenne | ≤ 150 mW | [POSÉ] |

---

## 3. Chaîne outil — commandes exactes

### Les trois dépôts

| Dépôt | Rôle |
|---|---|
| `https://gitlab.eclipse.org/olivierpds/alif_aidge_sdk` | **le SDK** : export Python + runtime C++ (caméra, LCD, TFLM + Ethos-U), CMake, scripts de flash |
| `https://gitlab.eclipse.org/olivierpds/alif_yolo_fastest` | **l'exemple à copier** : montre comment bâtir son projet sur le SDK (détecteur de visages YOLO-Fastest) |
| `https://gitlab.eclipse.org/olivierpds/aidge-tosa-fork.git` | **le fork d'`aidge_tosa`** — voir l'avertissement ci-dessous |

### ⚠️ Pièges connus

1. **Utiliser le fork `aidge-tosa-fork`, pas l'`aidge_tosa` officiel.** Le fork contient le support de
   couches du modèle MegaDetector qui n'est **pas encore mergé** en amont. Sans lui, l'export échoue
   sur certaines couches. [SOURCÉ, insisté deux fois par le porteur]
2. **La quantification est l'étape piégeuse.** « Quelques lignes dans un script », mais selon le
   modèle et les données de calibration fournies, on peut **perdre énormément de précision** — et se
   retrouver avec un modèle qui ne marche plus du tout. Y passer du temps. [SOURCÉ]
3. **Python 3.11 ou 3.12 uniquement.** Pas 3.13. [SOURCÉ]
4. L'ONNX d'entrée doit avoir ses **entrées dans [0, 1]** et être en **NCHW**. [SOURCÉ]
5. Le build **vérifie** que le modèle a été exporté pour le sous-système qu'on compile. Un export
   `--target hp` ne se compile pas en `ALIF_SUBSYSTEM=HE`. [SOURCÉ]
6. **55 000 bauds, c'est l'UART du Secure Enclave** (celle que SETOOLS utilise), pas la console
   applicative. Ne pas confondre. [SOURCÉ]
7. `git submodule update --init` télécharge **~500 Mo**. [SOURCÉ]
8. **AGPL v3 contaminante** : le modèle YOLOv10 d'Ultralytics impose de rouvrir les développements
   autour. Sans problème pour le challenge et pour un usage non commercial, **mais impossible à
   réutiliser tel quel pour un produit commercial**. [SOURCÉ]
9. `pip` peut avertir que `tosa-tools` veut `ml-dtypes==0.5.1` alors qu'`onnx` a installé 0.6 :
   **sans conséquence** pour cette chaîne. [SOURCÉ]

### Installation

```bash
# 1. environnement Python (les outils aidge vivent hors du dépôt)
python3 -m venv .venv && source .venv/bin/activate      # ou : uv venv .venv --python 3.12
pip install "aidge-core>=0.10.1,<0.11" "aidge-backend-cpu>=0.10.1,<0.11" \
            "aidge-onnx>=0.10.1,<0.11" "aidge-quantization>=0.10.1,<0.11" \
            onnx onnxruntime pillow numpy

# 2. aidge_tosa (LE FORK) et aidge_export_arm_ethos — absents de PyPI
git clone https://gitlab.eclipse.org/olivierpds/aidge-tosa-fork.git aidge_tosa
pip install ./aidge_tosa
git clone https://gitlab.eclipse.org/eclipse/aidge/aidge_export_ethos.git aidge_export_arm_ethos
pip install ./aidge_export_arm_ethos

# 3. le SDK + contrôle que tout s'importe
pip install -e alif_aidge_sdk
python -c "import aidge_core, aidge_backend_cpu, aidge_onnx, aidge_quantization, aidge_tosa, \
    aidge_export_arm_ethos, onnxruntime, ethosu.vela, alif_aidge_sdk; print('ok')"

# 4. toolchain + submodules
git submodule update --init                              # ~500 Mo
export PATH=/chemin/vers/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin:$PATH

# 5. SETOOLS (propriétaire, depuis alifsemi.com), configuré une fois depuis la carte
export SETOOLS=/chemin/vers/app-release-exec-linux
(cd $SETOOLS && ./tools-config -a -b 55000 -c /dev/alif-se)
cat $SETOOLS/utils/global-cfg.db                         # attend un E7 (AE722F80F55D5)
```

Prérequis : Python 3.11/3.12, git, **CMake ≥ 3.21**, **Arm GNU toolchain 13.x**, SETOOLS, et la carte
sur son **port PRG USB**. [SOURCÉ]

Note : le dépôt s'appelle `aidge_export_ethos` mais le **paquet** s'installe sous le nom
`aidge_export_arm_ethos`.

### Cycle de développement

```bash
# export d'un ONNX vers une cible (hp ou he)
alif-aidge-export --onnx model.onnx --calib-dir calib_npy/ --out export_hp --target hp --verify

# génération des sources C++ (model_data.cc, model_io.hpp)
alif-aidge-gen-model --export-dir export_hp --out generated

# compilation du firmware
cmake -S . -B build-hp -DALIF_SUBSYSTEM=HP -DALIF_CAMERA_MODULE=MT9M114
cmake --build build-hp -j8                               # -> build-hp/<cible>.mram.bin

# programmation de la MRAM
alif_aidge_sdk/tools/flash_mram.sh --cpu HP build-hp/<cible>.mram.bin
```

Options CMake utiles [SOURCÉ] : `ALIF_SUBSYSTEM` (HP, HE), `ALIF_CAMERA_MODULE` (MT9M114, ARX3A0),
`TARGET_BOARD` (AppKit-e7, DevKit-e7), `ACTIVATION_BUF_SZ`, `MODEL_DIR` (pour un build HE à côté d'un
build HP), `DETECTION_MAX_DET`, `LCD_IMAGE_SCALE`, `LCD_ROTATE_180`, `LCD_MIRROR`,
`CAMERA_FLIP_VERTICAL`.

### Brancher son projet sur le SDK

Ajouter le SDK en submodule, puis dans `CMakeLists.txt` [SOURCÉ] :

```cmake
include(${SDK}/cmake/alif_toolchain.cmake)      # AVANT project()
project(mountain LANGUAGES C CXX ASM)
include(${SDK}/cmake/alif_runner.cmake)
alif_add_runner(mountain_runner SOURCES src/main.cpp MODEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated)
```

---

## 4. Décomposition logicielle attendue

Elle découle de l'architecture fonctionnelle du modèle MBSE. Chaque module correspond à une fonction
allouée à un composant :

| Module | Fonction | Domaine |
|---|---|---|
| pilote caméra (fourni par le SDK) | capturer l'image du capteur | HP |
| `preprocess` | convertir le format de pixel, redimensionner | HP |
| runtime d'inférence (SDK) | préparer le tenseur int8, inférer le détecteur | HP → NPU-256 |
| `detect_postprocess` | décoder les ancres, filtrer (seuil + NMS) | HP |
| `mailbox` | synchroniser les deux domaines (mémoire partagée) | HP ↔ HE |
| `crop` | recadrer l'imagette 96 × 96 | HE |
| runtime d'inférence (SDK) | inférer le classifieur, arg max de l'espèce | HE → NPU-128 |
| `overlay` | annoter l'image (rectangles + libellés) | HE ou HP |
| pilote LCD (SDK) | afficher l'image annotée | — |
| console UART (SDK) | tracer les temps d'inférence | — |

Structure de dépôt suggérée à l'origine, calquée sur `alif_yolo_fastest`. **Elle a été
remplacée** par l'arborescence effective du dépôt — voir
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) et [docs/MIGRATION.md](docs/MIGRATION.md). Les
intentions restent les mêmes (séparer modèles, sources et sorties générées ; une référence
Python du post-traitement comparée à un build hôte du C++), la disposition diffère :

```
alif_aidge_sdk/        submodule du SDK
model/
  detector/            MDv6 : fetch, prepare_onnx, quantize_export
  classifier/          entraînement + export ONNX + quantize_export
  models/              ONNX float nettoyés (versionnés)
src/
  main_hp.cpp          acquisition + détection
  main_he.cpp          classification
  preprocess.{h,cpp}
  detect_postprocess.{h,cpp}
  crop.{h,cpp}
  mailbox.{h,cpp}
  overlay.{h,cpp}
generated_hp/ generated_he/    sorties d'alif-aidge-gen-model (NON versionnées)
tests/                 build hôte du post-traitement, comparé à la référence Python
```

`alif_yolo_fastest` fait exactement cela pour son décodeur : une **référence Python** du
pré/post-traitement, un **build hôte** du C++, et un script qui compare les deux. **Reprendre ce
schéma** — c'est le seul moyen de déboguer un décodeur d'ancres sans y passer des jours.

---

## 5. Machine à états de l'application

```
Initial → Démarrage
Démarrage       → Attente d'image        [initialisation terminée]
Attente d'image → Détection              [nouvelle image disponible]
Détection       → Classification         [détection terminée ; au moins une boîte retenue]
Détection       → Restitution            [détection terminée ; aucune boîte au-dessus du seuil]
Classification  → Restitution            [classification terminée]
Restitution     → Attente d'image        [affichage rafraîchi]
Détection       → Défaut                 [erreur d'inférence]
Classification  → Défaut                 [erreur d'inférence]
Défaut          → Démarrage              [after 2 s]
Attente d'image → Mise à jour du modèle  [carte connectée au port PRG]
Mise à jour     → Démarrage              [programmation terminée]
```

Le chemin « Détection → Restitution » sans classification (aucune boîte retenue) est un cas **nominal
fréquent** sur le terrain, pas une exception : la plupart des images sont vides.

---

## 6. Exigences — contrat de développement

51 exigences, identifiants stables. Les citer dans les commits et les commentaires (`cf. REQ-012`).
Ci-dessous celles qui contraignent directement le code.

**Chaîne de traitement**
- `REQ-007` redimensionner l'image du capteur à la résolution d'entrée du détecteur
- `REQ-008` localiser par une boîte englobante chaque animal présent
- `REQ-009` associer un score de confiance à chaque boîte
- `REQ-010` écarter toute détection de score < 0,25
- `REQ-011` fusionner les boîtes dont l'IoU dépasse 0,45
- `REQ-012` retenir au maximum 5 détections par image
- `REQ-013` extraire une imagette par boîte retenue
- `REQ-014` attribuer une espèce à chaque imagette
- `REQ-015` enchaîner la classification après la détection, pour chaque image

**Matériel et répartition**
- `REQ-024` détecteur sur le domaine M55-HP · `REQ-025` classifieur sur le domaine M55-HE
- `REQ-026` échange des boîtes par mémoire partagée
- `REQ-019` poids sur 8 bits entiers · `REQ-020` convolutions exécutées par l'Ethos-U55
- `REQ-110` HP à 400 MHz · `REQ-111` HE à 160 MHz · `REQ-108` 256 MAC · `REQ-109` 128 MAC

**Mémoire** — `REQ-004` ≤ 8 Mo SRAM · `REQ-005` ≤ 4,5 Mo MRAM · `REQ-101` à `REQ-105` (détail §2)

**Performance** — `REQ-002` entrée ≥ 320 × 320 · `REQ-003` ≤ 1 000 ms · `REQ-006` ≥ 1 img/s ·
`REQ-016` top-1 ≥ 80 % · `REQ-017` rappel ≥ 70 % · `REQ-018` perte mAP ≤ 5 pts ·
`REQ-033` boîte ≥ 24 × 24 px · `REQ-034` ≤ 150 mW

**Restitution** — `REQ-021` image annotée sur le LCD · `REQ-022` nom d'espèce au-dessus de chaque
boîte · `REQ-023` temps d'inférence sur la console série

**Chaîne outil** — `REQ-027` entrée ONNX fp32 · `REQ-028` binaire régénérable par une commande unique ·
`REQ-029` programmation MRAM par SETOOLS/PRG USB · `REQ-030` diffusion sous AGPL v3

**Robustesse** — `REQ-031` 50 à 100 000 lux · `REQ-032` -10 à +40 °C · `REQ-035` reprise NPU < 2 s ·
`REQ-036` initialisation caméra < 3 s

### Chiffres à confirmer en priorité

Ces valeurs sont **[POSÉ]** et doivent être relevées dès que la carte est disponible :

| | Valeur supposée | À vérifier |
|---|---|---|
| `REQ-107` | caméra 1280 × 720 @ 10 img/s | le SDK nomme `MT9M114` mais pas son mode ; le natif est 1296 × 976 |
| `REQ-112` | LCD 480 × 800 | relever la définition réelle de la dalle de l'AppKit |
| `REQ-113` | console à 115 200 bauds | ne pas confondre avec les 55 000 bauds de l'UART du Secure Enclave |
| `REQ-106` | imagette 96 × 96 | compromis texture du pelage / coût sur NPU-128, à mesurer |
| `REQ-002` | entrée détecteur 320 × 320 | les slides montrent 240 × 240 par défaut ; 320 est **l'objectif** |

---

## 7. Travaux ouverts, par ordre de valeur

Le point 1 est une recommandation d'ordre de travail ; les points 2 à 5 sont les quatre problèmes
listés dans les slides [SOURCÉ], remis dans l'ordre où ils débloquent le reste :

1. **Faire tourner un premier modèle de bout en bout.** Suivre le tutoriel
   `aidge_export_ethos/-/blob/main/examples/mobilenetv2_alif_hp.ipynb`, puis l'exemple
   `examples/mobilenetv2` du SDK. Ne rien optimiser avant que ça tourne.
2. **Exploiter au mieux la mémoire du MCU.** Partitionner pour les modèles et les images : activer les
   mémoires, configurer la compilation avec le **fichier de link**. Les blocs mémoire n'ont pas tous
   les mêmes performances — certains sont plus rapides que d'autres.
3. **Appliquer du tiling** aux différentes couches du modèle — c'est le levier pour monter en
   résolution d'entrée sans exploser l'arène.
4. **Faire tourner les deux modèles** sur le système : répartir la mémoire, **synchroniser les CPU**.
5. **Améliorer le pré/post-traitement.**

---

## 8. Le modèle MBSE

Toutes les décisions ci-dessus viennent d'un modèle MBSE complet (démarche boîte noire → boîte
blanche), réalisé dans **MBSE Studio** : 51 exigences, 9 acteurs, 5 phases de vie, 8 modes, 8 cas
d'utilisation, 6 scénarios, 25 fonctions, 23 composants, 30 connecteurs, 23 cas de test.
Vérificateur : **0 alerte**, 44/44 items de checklist.

Le modèle est le **référentiel des exigences** ; ce fichier n'en est qu'un extrait orienté
développement. En cas de divergence, le modèle fait foi.

- Projet : `aidge-mountain` dans MBSE Studio (`workspace/projects/aidge-mountain.mbse.json`)
- Générateur : `scripts/build-aidge-mountain.ts` — **écrase** le fichier projet, ne pas le relancer
  après une édition dans l'interface
- Les 23 cas de test (`tc1`–`tc23`) donnent les procédures de validation de chaque exigence —
  s'y référer avant d'écrire un test

---

## 9. Ressources

| | |
|---|---|
| AIDGE (doc, tutoriels, quick start) | https://eclipse.dev/aidge/ |
| AIDGE (dépôt) | https://gitlab.eclipse.org/eclipse/aidge/aidge |
| Tutoriel d'export recommandé | `aidge_export_ethos/-/blob/main/examples/mobilenetv2_alif_hp.ipynb` |
| Kit de développement | https://alifsemi.com/support/kits/ensemble-e7appkit/ |
| Exemples Alif | https://github.com/alifsemi |
| PyTorch-Wildlife (modèles + datasets) | https://github.com/microsoft/Pytorch-Wildlife |
| MegaDetector (model zoo) | https://microsoft.github.io/Biodiversity/model_zoo/megadetector/ |
| YOLOv10 | https://docs.ultralytics.com/models/yolov10 |
| En cas de blocage | olivier@mountainedge.ai |

Recommandation du porteur [SOURCÉ] : commencer par les tutoriels AIDGE (entraîner un modèle simple,
utiliser de l'ONNX, faire une *post-training quantization*) **avant** de toucher à la carte.

---

## 10. Consignes de travail pour Claude Code

- **Ne pas réentraîner MegaDetector.** Le défi est de le *déployer efficacement*, pas de le refaire.
  Créer un détecteur performant demanderait bien plus que le temps imparti. [SOURCÉ]
- **Ne jamais présenter une valeur [POSÉ] comme un fait.** Si une décision dépend d'un chiffre non
  confirmé, le dire et proposer comment le mesurer.
- **Mesurer avant d'optimiser.** La console série (`REQ-023`) existe pour ça : instrumenter d'abord,
  conclure ensuite. Pas d'optimisation mémoire spéculative.
- **Tout changement de post-traitement se valide contre la référence Python** (build hôte, cf. §4).
- Commenter en **français**, comme le reste du projet ; garder les identifiants, commandes et noms de
  fichiers verbatim en anglais.
- Citer les exigences concernées dans les messages de commit.
- En cas de doute sur une couche non supportée à l'export : vérifier d'abord qu'on utilise bien le
  **fork** `aidge-tosa-fork`, puis remonter le cas à la communauté AIDGE.
