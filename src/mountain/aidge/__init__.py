"""Couche d'adaptation au-dessus d'AIDGE.

Regroupe ce qui dépend de l'API d'``aidge_core`` / ``aidge_onnx`` /
``aidge_export_cpp`` : préparation de graphe (``graph``) et export C++
(``export``). Isoler ces appels ici permet d'absorber les changements d'API
d'AIDGE en un seul endroit.
"""

from . import export, graph  # noqa: F401
