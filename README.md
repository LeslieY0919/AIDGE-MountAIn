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
python3.12 -m venv .venv && source .venv/bin/activate   # Python 3.11 ou 3.12, pas 3.13
make install                                            # chaîne complète
```

`make install` installe le paquet `mountain` en mode éditable. **Cette étape
n'est pas optionnelle** : les scripts l'importent, et sans elle chacun échoue
sur `ModuleNotFoundError: No module named 'mountain'`. C'est le prix du layout
`src/`, et c'est voulu — il n'y a plus d'import accidentel depuis le
répertoire courant.

Pour ne parler qu'à la carte, `make install-board` suffit et reste léger.

`make install` ajoute aussi `aidge-quantization` avec `--no-deps`. Ce
`--no-deps` est volontaire : les
métadonnées du paquet déclarent `aidge-backend-cuda` comme obligatoire alors
que la chaîne CPU utilisée ici n'en a pas besoin. `pip check` signalera donc
une dépendance absente — sans conséquence. Pour le backend CUDA, installer
`aidge-quantization` sans `--no-deps` et configurer `CUDA_PATH`.

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

## Carte du dépôt

```
configs/        toute la configuration — seuils, résolutions, hyperparamètres
src/mountain/   le socle commun, importable : chemins, config, pré-traitement, AIDGE
scripts/        les points d'entrée exécutables, groupés par étape de la chaîne
                (detector, classifier, quantize, pipeline, board)
cpp/            sources C++ : harnais du détecteur, instantané du pipeline, tiers
tests/          contrôles de parité entre la référence Python et le C++
tools/          audits et diagnostics ponctuels, hors chaîne nominale
docs/           architecture et table de migration
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
