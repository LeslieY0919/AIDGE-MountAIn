# `tests/` — contrôles de parité Python ↔ C++

Ces fichiers sont préfixés `check_` et non `test_` : ce sont des **scripts
exécutables**, pas des tests unitaires. Ils ont besoin d'artefacts construits
— modèles téléchargés, exports générés, binaire compilé — et peuvent prendre
plusieurs minutes. Le préfixe les tient volontairement hors de la collecte
`pytest`, qui ne ramasse que `test_*.py`, laissant la place à de vrais tests
unitaires sans mélanger les deux.

Lancer un contrôle :

```bash
python tests/check_cpp_detector_int8.py
```

| Contrôle | Vérifie | Prérequis |
|---|---|---|
| `check_aidge_onnx_smoke.py` | qu'AIDGE charge un ONNX trivial | aucun |
| `check_detector_export.py` | structure du graphe du détecteur, opérateurs non natifs | détecteur ONNX |
| `check_detector_aidge_parity.py` | AIDGE CPU ≡ ONNX Runtime sur le détecteur fp32 | détecteur ONNX |
| `check_cpp_detector_fp32.py` | C++ fp32 ≡ ONNX Runtime | `exports/detector_fp32/` compilé |
| `check_cpp_detector_int8.py` | C++ int8 ≡ ONNX Runtime, avec compilation | `exports/detector_int8/` |
| `check_detector_int8_decode.py` | décodage des sorties int8 déjà produites | sorties du contrôle précédent |
| `check_cpp_pipeline_full.py` | chaîne complète détection → découpe → classification | `make build-cpp` |

L'ordre du tableau est l'ordre utile : un échec en bas se diagnostique en
remontant. Si `check_cpp_detector_int8.py` échoue mais que
`check_detector_aidge_parity.py` passe, le problème est dans la quantification
ou le post-traitement C++, pas dans le graphe.

## Pourquoi deux exemplaires du post-traitement

Le décodage DFL, la sigmoïde et le NMS existent en Python
(`scripts/pipeline/run_reference.py`) **et** en C++
(`cpp/detector_standalone/`, `cpp/wildlife_pipeline/`). C'est assumé : le C++
est la cible, Python est l'oracle. C'est la démarche de `alif_yolo_fastest`,
et la seule qui rende un décodeur d'ancres débogable.
