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
make install
```

`make install` installe le paquet `mountain` en mode éditable, puis
`aidge-quantization` avec `--no-deps`. Ce `--no-deps` est volontaire : les
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

## Carte du dépôt

```
configs/        toute la configuration — seuils, résolutions, hyperparamètres
src/mountain/   le socle commun, importable : chemins, config, pré-traitement, AIDGE
scripts/        les points d'entrée exécutables, groupés par étape de la chaîne
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
