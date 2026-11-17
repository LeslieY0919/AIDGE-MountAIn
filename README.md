# AIDGE MountAIn — détection et identification d'animaux sauvages

Challenge AIDGE 2026, défi **MountAIn**. Chaîne de traitement complète :

```
image caméra → REDIMENSIONNER → DÉTECTER (MegaDetector v6) → DÉCOUPER → CLASSIFIER → « Chamois »
```

Cible finale : le kit **Alif Ensemble E7 AppKit** (deux Cortex-M55, chacun avec
son NPU Ethos-U55), via la chaîne outil **Eclipse AIDGE**. Ce dépôt porte
l'étape qui précède le portage : entraîner, exporter, quantifier en int8 et
valider les deux modèles sur hôte Linux avec `aidge_export_cpp`.

Le contexte complet du défi (exigences, budgets mémoire, décisions
d'architecture, pièges connus de la chaîne outil) est dans **[CLAUDE.md](CLAUDE.md)**.

---

## Démarrage

```bash
make venv                      # crée .venv avec un Python 3.11/3.12
source .venv/bin/activate
make install                   # chaîne complète
```

Trois points sur lesquels on se casse les dents sans ces cibles :

- **la version.** AIDGE exige Python **3.11 ou 3.12** — ni 3.13, ni moins.
  `make` ne prend donc pas `python` au hasard dans le `PATH` : il cherche le
  premier interpréteur compatible (`python3.12`, `python3.11`, `python3`), en
  laissant gagner un venv déjà actif. Sur macOS, `python3` est très souvent un
  3.9 livré par le système, et `python` n'existe pas du tout. Forcer avec
  `make PYTHON=/chemin/vers/python3.12 <cible>` ;
- **l'environnement virtuel.** Homebrew et les Python de distribution se
  déclarent « externally managed » (PEP 668) et refusent toute installation
  en dehors d'un venv. `make install` le détecte et dit quoi faire, au lieu de
  laisser `pip` répondre par un pavé ;
- **l'installation elle-même n'est pas optionnelle.** Les scripts importent le
  paquet `mountain` ; sans `make install`, chacun échoue sur
  `ModuleNotFoundError: No module named 'mountain'`. C'est le prix du layout
  `src/`, et c'est voulu — il n'y a plus d'import accidentel depuis le
  répertoire courant.

Pour ne parler qu'à la carte, `make install-board` suffit : cinq paquets, ni
torch ni AIDGE.

`make install` ajoute aussi `aidge-quantization` avec `--no-deps`. Ce
`--no-deps` est volontaire : les métadonnées du paquet déclarent
`aidge-backend-cuda` comme obligatoire alors que la chaîne CPU utilisée ici
n'en a pas besoin. `pip check` signalera donc une dépendance absente — sans
conséquence. Pour le backend CUDA, installer `aidge-quantization` sans
`--no-deps` et configurer `CUDA_PATH`.

`make help` liste toutes les cibles.

## La chaîne, dans l'ordre

| # | Commande | Produit |
|---|---|---|
| 1 | `make detector-weights` | `models/detector/MDV6-yolov10-c.{pt,onnx}` |
| 2 | `make dataset` | `data/datasets/wildlife/{train,val}/<espèce>/` |
| 3 | `make train-classifier` | `models/classifier/species_classifier.onnx` |
| 4 | `make export-detector` | `exports/detector_fp32/` |
| 5 | `make export-classifier` | `exports/classifier_fp32/` |
| 6 | `make quantize` | `exports/detector_int8/`, `exports/classifier_int8/` |
| 7 | `make build-cpp` | `exports/wildlife_pipeline/build/wildlife_pipeline` |
| 8 | `make check` | contrôles de parité Python ↔ C++ |

`make test` est à part : il lance `pytest` sur les `test_*.py`, et il n'y en a
aucun pour l'instant. Les contrôles de `tests/` sont préfixés `check_` parce
qu'ils ont besoin d'artefacts construits — voir [tests/README.md](tests/README.md).

Les étapes 4 et 5 sont des contrôles de faisabilité en fp32 : elles valident
qu'AIDGE sait exporter les graphes avant d'ajouter la difficulté de la
quantification. On peut passer directement de 3 à 6, mais en cas d'échec il
sera plus difficile de savoir si le problème vient du graphe ou de la PTQ.

### Exécuter le pipeline C++

```bash
make build-cpp
exports/wildlife_pipeline/build/wildlife_pipeline data/samples/test_wildlife_1mp.jpg
```

`make build-cpp` **génère intégralement** `exports/wildlife_pipeline/` : il
efface le dossier, y recopie les sources `dnn/` des deux exports int8, puis
écrit `main.cpp`, `CMakeLists.txt` et les deux scripts de version de
l'éditeur de liens. L'étape 6 est donc un prérequis.

Ce qui est versionné sous `cpp/wildlife_pipeline/` n'est qu'un **instantané de
référence** de cette sortie, pour pouvoir la relire et en suivre les
changements — voir le [README du dossier](cpp/wildlife_pipeline/README.md).

## Se connecter à la carte

```bash
make venv && source .venv/bin/activate
make install-board                   # léger : pyserial, sans torch ni AIDGE
make board-list                      # quels ports série sont visibles ?
make connect                         # console applicative, journalisée dans artifacts/
```

`make install-board` installe cinq paquets. Une machine qui ne sert qu'à
parler à la carte n'a aucune raison d'installer torch — c'est `make install`
qui ajoute la chaîne d'entraînement et d'export.

Le port **PRG USB** de l'AppKit expose **deux UART**, et les confondre est le
piège le plus coûteux de la chaîne :

| | Débit | Rôle |
|---|---|---|
| console applicative | 115 200 **[POSÉ]** | les temps d'inférence (REQ-023) — défaut du script |
| UART Secure Enclave | 55 000 | SETOOLS, programmation de la MRAM (`--se`) |

Se tromper ne lève aucune erreur : on obtient du binaire illisible, ou un flash
qui échoue parce qu'on parlait sur le lien pendant la programmation. Les
115 200 bauds sont une **hypothèse** (REQ-113), à confirmer sur la carte et à
corriger dans `configs/board.yaml`.

Sur la carte du groupe (AppKit avec J-Link embarquée), macOS ne voit qu'**un
seul** port : les cavaliers J15 le relient soit à l'UART du Secure Enclave
(position par défaut), soit à la console applicative (UART2). Détails dans
[AGENTS.md](AGENTS.md#matériel).

Quand deux ports indiscernables apparaissent, le script ne devine pas — les
deux interfaces CDC d'un même composite USB ont le même genre de nom. Il
demande `--port`.

```bash
make board-check                                     # SETOOLS voit-il un E7 ?
python scripts/board/connect.py --port /dev/cu.usbmodem103
python scripts/board/connect.py --se --read-only     # écouter le Secure Enclave
```

`make board-check` suppose `$SETOOLS` défini et déjà configuré depuis la carte :

```bash
export SETOOLS=/chemin/vers/app-release-exec-linux
(cd $SETOOLS && ./tools-config -a -b 55000 -c /dev/alif-se)
```

### Outils de la carte et firmware de démonstration

```bash
source scripts/board/env.sh          # J-Link, SETOOLS, toolchain Arm, kit d'Alif
make board-firmware                  # compile la démo img_class du kit, sans flasher
```

`scripts/board/env.sh` fixe les chemins des outils (redéfinissables dans
`scripts/board/env.local.sh`, non versionné) et définit `$SETOOLS`.
`make board-firmware` compile l'exemple de classification du kit d'évaluation
ML d'Alif avec le modèle de son choix et prépare le paquet ATOC ; **rien n'est
écrit sur la carte**. Versions des outils, mode d'emploi et opérations à faire
valider avant tout flash : [AGENTS.md](AGENTS.md).

## Vers l'Ethos-U55

Deux compléments à la chaîne sur hôte, tous deux sans carte :

- **`tools/u55/`** estime avec Vela le temps d'inférence et la mémoire d'un
  modèle sur l'Ethos-U55, couche par couche. Résultats dans
  [AGENTS.md](AGENTS.md#mesures-vela-sur-lethos-u55).
- **`scripts/npu/export_classifier.py`** exporte un classifieur ONNX avec
  AIDGE : PTQ int8 → TOSA (`aidge_tosa`) → Vela → `.tflite` pour le NPU. Il
  demande l'environnement de `docker/aidge/` ; sur une machine sans wheel
  AIDGE (macOS), `scripts/npu/aidge_docker.sh` l'exécute sur une machine Linux
  jointe en SSH.

## Notes d'échange

[notes/](notes/README.md) rassemble ce que chacun a essayé, mesuré ou
découvert, et les questions à trancher ensemble. Ce ne sont pas des documents
de référence : on y écrit pour se tenir au courant, et on reporte ici ou dans
CLAUDE.md ce qui est devenu stable.

## Carte du dépôt

```
configs/        toute la configuration — seuils, résolutions, hyperparamètres
src/mountain/   le socle commun, importable : chemins, config, pré-traitement, AIDGE
scripts/        les points d'entrée exécutables, groupés par étape de la chaîne
                (detector, classifier, quantize, pipeline, board, npu)
cpp/            sources C++ : harnais du détecteur, instantané du pipeline, tiers
tests/          contrôles de parité entre la référence Python et le C++
tools/          audits et diagnostics ponctuels, hors chaîne nominale
                (dont u55/ : estimations Vela pour l'Ethos-U55)
docker/         image de l'environnement AIDGE → TOSA → Vela
docs/           architecture et table de migration
notes/          notes d'échange entre membres du groupe
attic/          itérations historiques conservées pour référence
```

Et les quatre dossiers de données, dont un seul est versionné :

| Dossier | Contenu | Versionné |
|---|---|---|
| `data/samples/` | images de référence des contrôles | **oui** |
| `data/datasets/` | datasets construits par script | non |
| `models/` | `.pt` et `.onnx` — à partager via GitHub Releases | non |
| `exports/` | sources C++ générées par AIDGE | non |
| `artifacts/` | images annotées, tampons `.bin`, journaux | non |

La règle est simple : **si un script peut le régénérer, git ne le suit pas.**

## Deux principes à respecter

**Aucun chemin en dur.** Tout fichier se désigne par une constante de
`mountain.paths`. C'est ce qui permet de lancer n'importe quel script depuis
n'importe quel répertoire, et de déplacer un artefact sans corriger quinze
fichiers.

```python
from mountain import paths
graph = load_graph(paths.DETECTOR_ONNX_CUT_FOLDED, [1, 3, 640, 640])
```

**Aucune constante magique.** Seuils, résolutions et hyperparamètres viennent
de `configs/`, lus par `mountain.config`. Le seuil de confiance du détecteur
doit avoir la même valeur dans la référence Python et dans le C++ ; une
constante dupliquée est une divergence en attente.

```python
from mountain import config
conf = config.detector()["postprocess"]["conf_threshold"]
```

## Déboguer le post-traitement

Le décodage des ancres DFL + sigmoïde + NMS existe en deux exemplaires : la
référence Python (`scripts/pipeline/run_reference.py`) et le C++
(`cpp/detector_standalone/`). C'est assumé — le C++ est la cible, Python est
l'oracle. **Tout changement de post-traitement se valide en comparant les deux**
(`make check`). C'est le seul moyen raisonnable de déboguer un décodeur
d'ancres, et `alif_yolo_fastest` procède exactement ainsi.

## Écarts connus avec le modèle MBSE

À trancher avant le portage sur carte, documentés dans `configs/classifier.yaml` :

- le classifieur est un **ResNet-34 à 128×128**, là où la décision d'équipe
  retenait **MobileNetV2 à 96×96**. ResNet-18 avait été écarté pour ses ~11 Mo
  de poids, hors budget MRAM : un ResNet-34 ne tiendra pas dans le 1 Mo alloué
  (REQ-105) ;
- le détecteur est exporté à **640×640** sur hôte, contre un objectif de
  320×320 (REQ-002). Une image 640×640 ne tient pas dans l'arène de 2,5 Mo
  visée pour le domaine HP.

Les deux écarts sont sans conséquence pour la validation sur hôte Linux, et
bloquants pour l'embarqué. Mesurer avant de décider (cf. CLAUDE.md §10).

## Dépannage

**`Package 'mountain' requires a different Python: 3.9.x not in '<3.13,>=3.11'`**

Le `pip` utilisé appartient à un Python trop ancien. AIDGE impose 3.11 ou
3.12. Les cibles `make` choisissent l'interpréteur pour vous :

```bash
make venv && source .venv/bin/activate && make install
```

Si aucun Python compatible n'est présent, `make` liste les interpréteurs
visibles et leurs versions plutôt que d'échouer sur un message de `pip`.

**`error: externally-managed-environment`**

Le Python visé refuse d'installer hors environnement virtuel (PEP 668), ce qui
est le cas de Homebrew. Créer le venv : `make venv && source .venv/bin/activate`.
En dernier recours, `make PIP_FLAGS=--break-system-packages install` passe
outre — au risque d'abîmer l'installation Python du système.

**`ModuleNotFoundError: No module named 'mountain'`**

Le paquet n'est pas installé dans l'environnement courant. C'est l'erreur la
plus probable sur une machine neuve :

```bash
cd <racine du dépôt>
make install          # chaîne complète
make install-board    # ou le minimum, pour la carte uniquement
```

Si l'erreur persiste, c'est que `pip` et `python` ne pointent pas vers le même
environnement. Vérifier avec `python -c "import sys; print(sys.prefix)"` et
`pip -V`.

**Aucun port série trouvé, ou plusieurs ports candidats**

`make board-list` écarte déjà les pseudo-ports permanents de macOS
(Bluetooth, console de debug, debug Wi-Fi) et les nœuds `/dev/tty.*`. Sur
macOS, n'utiliser que `/dev/cu.*` : `/dev/tty.*` désigne le même port côté
« appel entrant » et son ouverture bloque indéfiniment en attente de porteuse
(DCD), que la carte n'assure pas. La liste des exclusions est dans
`configs/board.yaml`.

S'il reste plusieurs candidats, le script ne devine pas et demande `--port`.

**`cmake` ne trouve pas `detector_dnn/` ou `classifier_dnn/`**

Ces sources sont générées : lancer `make quantize` puis `make build-cpp`. Le
pipeline C++ ne se compile pas depuis le dépôt seul.
