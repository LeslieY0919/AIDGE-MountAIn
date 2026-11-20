"""``mountain`` — socle commun du pipeline de détection et classification.

Le défi MountAIn (Challenge AIDGE 2026) déploie une chaîne
« image -> détection MegaDetector -> découpe -> classification d'espèce »
sur un Alif Ensemble E7. Ce paquet porte le code *réutilisable* ; les points
d'entrée exécutables vivent dans ``scripts/``.

Sous-modules :

- :mod:`mountain.paths`      chemins du dépôt, source unique de vérité ;
- :mod:`mountain.config`     lecture de ``configs/*.yaml`` et des classes ;
- :mod:`mountain.preprocess` pré-traitement d'image partagé ;
- :mod:`mountain.aidge`      préparation de graphe et export C++ AIDGE.
"""

__all__ = ["aidge", "config", "paths", "preprocess"]

__version__ = "0.1.0"
