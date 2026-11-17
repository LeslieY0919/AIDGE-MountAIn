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

Les trois premiers servent avant un export : **un opérateur `Generic` est une
couche qu'AIDGE n'implémente pas**, et c'est exactement ce qui fait échouer
l'export C++. Les compter coûte quelques secondes et évite de chercher
longtemps la cause d'un échec d'export.

`diagnose_ptq.py` et `tune_ptq.py` existent parce que la quantification est
l'étape piégeuse de la chaîne : selon le modèle et les données de calibration,
on peut perdre énormément de précision — jusqu'à un modèle qui ne marche plus.
Mesurer avant de conclure.

`explore_tiling.py` attaque le cœur de la difficulté du défi : pour détecter
un animal il faut une grande image, mais le microcontrôleur a peu de mémoire.
Un animal à 30 m ne fait que quelques dizaines de pixels.
