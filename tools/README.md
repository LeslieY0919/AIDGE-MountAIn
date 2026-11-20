# `tools/` — audits et diagnostics

Scripts hors chaîne nominale. On les lance ponctuellement pour **comprendre**,
pas pour produire un artefact. Rien dans `scripts/` n'en dépend.

| Outil | Question à laquelle il répond |
|---|---|
| `audit_detector.py` | quel dépôt Hugging Face sert les poids MDv6, et le graphe passe-t-il dans AIDGE ? |
| `audit_detector_arch.py` | combien d'opérateurs non natifs (`Generic`) dans un YOLO nano, pris comme substitut de MegaDetector ? |
| `audit_classifier.py` | combien de nœuds et de paramètres pour une architecture de classifieur candidate ? |
| `diagnose_ptq.py` | où la quantification int8 perd-elle de la précision, couche par couche ? |
| `tune_ptq.py` | quelle combinaison d'options de PTQ donne le meilleur top-1 ? |
| `explore_tiling.py` | le découpage en tuiles rattrape-t-il les petits animaux qu'un simple redimensionnement fait disparaître ? |
| `u55/profile_yolov10_u55.py` | combien de temps et de mémoire pour un YOLOv10n sur l'Ethos-U55, selon la taille d'entrée, l'activation et le PSA ? |
| `u55/profile_classifier_u55.py` | même question pour les classifieurs candidats (MobileNetV2 et V3) |
| `u55/compare_hp_he.py` | quel écart entre le domaine HP (U55-256) et le domaine HE (U55-128) pour un même `.tflite` ? |
| `u55/yolov10_decode.py` | le décodage des 6 sorties int8 brutes du détecteur redonne-t-il les boîtes d'Ultralytics ? |

Les trois premiers servent avant un export : **un opérateur `Generic` est une
couche qu'AIDGE n'implémente pas**, et c'est exactement ce qui fait échouer
l'export C++. Les compter coûte quelques secondes et évite de chercher
longtemps la cause d'un échec d'export.

`diagnose_ptq.py` et `tune_ptq.py` existent parce que la quantification est
l'étape piégeuse de la chaîne : selon le modèle et les données de calibration,
on peut perdre énormément de précision — jusqu'à un modèle qui ne marche plus.
Mesurer avant de conclure.

Les outils de `u55/` donnent des **estimations de Vela**, sans carte : temps
d'inférence, pic de SRAM, taille en MRAM et rapport par couche, écrits dans
`artifacts/u55_profile/`. Ils passent par TFLite (onnx2tf) et non par AIDGE,
et servent à choisir une taille d'entrée ou une architecture avant d'y passer
du temps. Ils demandent un environnement à part (Ultralytics, TensorFlow,
onnx2tf) et le Vela du kit d'Alif : versions et résultats dans
[AGENTS.md](../AGENTS.md).

`explore_tiling.py` attaque le cœur de la difficulté du défi : pour détecter
un animal il faut une grande image, mais le microcontrôleur a peu de mémoire.
Un animal à 30 m ne fait que quelques dizaines de pixels.
