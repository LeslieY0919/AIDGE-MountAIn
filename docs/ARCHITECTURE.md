# Architecture du dépôt

Ce document explique **pourquoi** le dépôt est rangé ainsi. Pour la liste des
commandes, voir le [README](../README.md) ; pour la correspondance avec
l'ancienne arborescence, voir [MIGRATION.md](MIGRATION.md).

## Le problème que cette structure résout

Le dépôt était un répertoire plat de ~45 scripts Python à la racine, mêlant
des itérations successives (`export_detector.py` jusqu'à `_v6.py`, plus un
`.bak`), quatre scripts de quantification qui se recouvraient, et 1,2 Go
d'exports C++ générés commités. Trois conséquences concrètes :

1. **Impossible de savoir quel script est le bon.** Entre `quantize_int8.py`,
   `quantize_and_export_cpp.py`, `quantize_pipeline_int8.py` et
   `rebuild_classifier_int8.py`, rien n'indiquait lequel appeler.
2. **Le code dupliqué divergeait.** Le bloc « charger l'ONNX, marquer les
   Producer constants, plier les constantes, propager les dimensions » était
   recopié dans huit fichiers, chaque copie ayant dérivé sur un détail. De
   même pour `MEAN`/`STD` et le pré-traitement — or le pré-traitement de la
   calibration PTQ **doit** être identique à celui de l'entraînement, sinon la
   quantification s'étalonne sur la mauvaise distribution.
3. **Les constantes vivaient dans le code.** Changer le seuil de confiance
   demandait de l'éditer dans quatre fichiers, en espérant n'en oublier aucun.

## Les cinq couches

```
configs/        ← les valeurs          (ce que la chaîne doit faire)
src/mountain/   ← le socle             (comment, de façon réutilisable)
scripts/        ← les points d'entrée  (quand, dans quel ordre)
cpp/            ← la cible             (le code qui tournera sur la carte)
tests/ tools/   ← la vérification      (preuve que ça marche)
```

La règle de dépendance est unidirectionnelle : `scripts/` importe
`src/mountain/`, qui lit `configs/`. Jamais l'inverse. Un script ne définit
pas de constante métier et ne contient pas d'algorithme réutilisable — il
orchestre.

### `configs/` — les valeurs

Trois YAML commentés plus la liste des classes. Toute valeur qui pilote le
comportement de la chaîne vit ici : résolutions d'entrée, seuils de
confiance et d'IoU, nombre maximal de détections, normalisation, hyper-
paramètres d'entraînement, options de PTQ, budgets mémoire de la cible.

Les commentaires portent la **traçabilité vers les exigences** (`REQ-010`,
`REQ-012`…) et distinguent les valeurs sourcées des hypothèses marquées
`[POSÉ]`, à confirmer sur la carte. C'est ce qui permet de relire la
configuration sans ouvrir le modèle MBSE.

### `src/mountain/` — le socle commun

Un paquet Python installable (`pip install -e .`), en layout `src/` pour qu'il
ne soit importable qu'une fois installé — pas d'import accidentel depuis le
répertoire courant.

| Module | Rôle |
|---|---|
| `paths.py` | **source unique de vérité des chemins.** Toute constante de fichier ou de dossier. Permet de lancer n'importe quel script depuis n'importe où. |
| `config.py` | lecture et mise en cache de `configs/*.yaml` et des noms de classes |
| `preprocess.py` | pré-traitement d'image — la seule définition de `MEAN`/`STD` et du passage en NCHW |
| `aidge/graph.py` | préparation d'un graphe AIDGE depuis un ONNX : marquage des Producer, pliage, propagation de dimensions, promotion des scalaires 0-D, fusion BatchNorm |
| `aidge/export.py` | appel de `aidge_export_cpp` en absorbant les changements d'API entre versions |

Le sous-paquet `aidge/` isole volontairement tout ce qui touche à l'API
d'AIDGE. Cette API a bougé entre 0.9 et 0.10 (signature d'export à mots-clés
variables, puis `ExportConfig`). Concentrer ces appels ici fait qu'une
nouvelle version d'AIDGE se gère en un seul endroit, au lieu de huit
introspections de signature recopiées.

### `scripts/` — les points d'entrée

Groupés par étape de la chaîne, pas numérotés : une numérotation préfixée
(`01_`, `02_`…) se périme dès qu'une étape s'insère au milieu. L'ordre est
porté par le `Makefile` et le tableau du README, qui sont faciles à corriger.

```
scripts/detector/      récupération des poids, fine-tuning, export C++
scripts/classifier/    construction du dataset, entraînement, export C++
scripts/quantize/      PTQ int8 des deux modèles
scripts/pipeline/      assemblage du C++ autonome, référence Python
```

### `cpp/` — la cible

```
cpp/detector_standalone/  main.cpp — VRAIE source, à éditer
cpp/wildlife_pipeline/    instantané de référence d'une sortie générée
cpp/third_party/stb/      stb_image.h (décodage JPEG, en-tête unique)
```

La distinction entre ces trois dossiers est le point le plus subtil de la
structure, et elle mérite d'être explicite.

`cpp/detector_standalone/main.cpp` est une **vraie source**. Elle vivait
auparavant dans une chaîne Python de 188 lignes au milieu du script d'export —
donc sans coloration syntaxique, sans compilation possible à l'unité, et sans
diff lisible. Elle est maintenant un fichier `.cpp`, que le script recopie dans
l'export avant de compiler.

`cpp/wildlife_pipeline/` est **un instantané**, pas une source.
`scripts/pipeline/build_standalone_cpp.py` génère intégralement le pipeline à
deux modèles : il efface son dossier cible, y recopie les sources `dnn/` des
deux exports int8, puis écrit `main.cpp`, `CMakeLists.txt` et les deux scripts
de version. Sa cible est `exports/wildlife_pipeline/`, non versionnée. Ce qui
est versionné sous `cpp/` est une copie de la dernière sortie connue, pour
pouvoir la relire — le générateur porte son C++ dans une f-string de ~260
lignes, bien moins lisible que son résultat — et pour voir ce que change une
modification du générateur.

C'est un compromis assumé, et il est documenté dans
`cpp/wildlife_pipeline/README.md` pour que personne n'édite l'instantané en
croyant modifier le build. La bonne cible serait un fichier gabarit, mais la
conversion d'une f-string contenant des accolades C++ à chaque ligne est
exactement le genre de changement qu'il faut pouvoir exécuter pour le valider.

Les deux `.map` sont des scripts de version de l'éditeur de liens. Ils
existent parce que les deux modèles exportés définissent tous les deux un
symbole `model_forward` : chaque modèle devient une bibliothèque partagée qui
n'exporte que son propre point d'entrée, ce qui évite la collision. La version
antérieure (`attic/wildlife_cpp_pipeline/`) renommait les symboles avec
`nm` + `objcopy`, ce qui dépendait des binutils GNU et ne marchait pas sur
macOS.

### `tests/` et `tools/`

`tests/` contient les contrôles de **parité Python ↔ C++**. Le post-traitement
(décodage DFL, sigmoïde, NMS) existe nécessairement en deux exemplaires : le
C++ est la cible, Python est l'oracle. Tout changement se valide en comparant
les deux — c'est le seul moyen raisonnable de déboguer un décodeur d'ancres,
et c'est la démarche de `alif_yolo_fastest`.

Ces fichiers sont préfixés `check_` et non `test_` : ce sont des scripts
exécutables qui ont besoin d'artefacts construits (modèles, exports, binaire
compilé), pas des tests unitaires. Le préfixe les tient hors de la collecte
pytest, laissant `test_*.py` disponible pour de vrais tests unitaires.

`tools/` contient les audits et diagnostics hors chaîne nominale : compter
les opérateurs `Generic` d'un graphe avant de tenter un export, balayer les
options de PTQ, explorer le tiling. On les lance ponctuellement pour
comprendre, pas pour produire.

### `attic/` — les itérations historiques

Les versions antérieures des scripts d'export et de quantification, avec un
README qui dit ce que chacune faisait et par quoi elle est remplacée. Elles
ne sont pas supprimées parce que les tentatives ratées d'export AIDGE sont de
l'information : elles documentent quelles approches ne passent pas.

## Les quatre familles de données

Un seul dossier de données est versionné. La règle : **si un script peut le
régénérer, git ne le suit pas.**

| Dossier | Contenu | Versionné | Pourquoi |
|---|---|---|---|
| `data/samples/` | images de référence | **oui** | quelques centaines de Ko, nécessaires aux contrôles |
| `data/datasets/` | datasets construits | non | reconstructible par `make dataset` |
| `models/` | `.pt`, `.onnx` | non | binaires volumineux → GitHub Releases |
| `exports/` | sources C++ générées, pipeline assemblé | non | régénérable, et c'était 1,2 Go |
| `artifacts/` | images annotées, `.bin`, journaux | non | sorties de run, jetables |

`models/` mérite une précision : les artefacts de modèles ne se versionnent
pas dans git, qui ne sait pas les compresser et garde chaque révision pour
toujours. Ils se publient en **GitHub Releases**, où ils sont téléchargeables
sans alourdir le clone.

## Ce que cette structure n'a pas changé

La réorganisation a déplacé le code et recâblé ses chemins ; elle n'a pas
réécrit la logique métier. Trois scripts restent gros et mériteraient d'être
découpés :

- `scripts/quantize/pipeline_int8.py` (~580 lignes) fait office de
  bibliothèque pour `rebuild_classifier.py`, qui en importe sept symboles.
  Ses correctifs `Dequantizer`/`Quantizer` et sa construction de graphe ont
  vocation à descendre dans `mountain/aidge/` ;
- `scripts/pipeline/build_standalone_cpp.py` (~380 lignes) **génère** du C++
  par concaténation de chaînes, en lisant les scales int8 par expression
  régulière dans un `main.cpp` existant. Fragile : ces scales devraient être
  écrites dans un JSON par l'étape de quantification ;
- `scripts/detector/export_cpp.py` contenait 188 lignes de C++ dans une
  chaîne Python. Ce C++ est désormais un fichier à part
  (`cpp/detector_standalone/main.cpp`) — c'est le seul morceau de logique que
  la réorganisation a effectivement sorti de son script. Les constantes de
  seuil y restent dupliquées depuis `configs/detector.yaml`, faute de pouvoir
  les injecter sans mécanisme de génération.

Ces trois points sont le chantier suivant, et ils demandent de pouvoir
exécuter la chaîne pour vérifier qu'on ne casse rien.
