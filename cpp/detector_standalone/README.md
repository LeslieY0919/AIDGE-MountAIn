# `cpp/detector_standalone/` — harnais C++ du détecteur seul

`main.cpp` est une **vraie source**, à éditer librement. Il contient le
pré-traitement et le post-traitement du détecteur, côté C++ :

- chargement JPEG (via `../third_party/stb/stb_image.h`) et redimensionnement
  bilinéaire vers l'entrée du réseau, en NHWC et dans [0, 1] ;
- décodage des ancres **DFL** : *softmax* sur les 16 bins de chaque côté,
  puis espérance, pour retrouver les distances au centre de la cellule ;
- sigmoïde sur les logits de classe, seuil de confiance, et **NMS** par classe ;
- remise à l'échelle des boîtes vers la résolution de l'image d'origine.

`scripts/detector/export_cpp.py` le recopie dans `exports/detector_fp32/` puis
compile l'ensemble avec les sources générées par AIDGE.

C'est l'un des deux exemplaires du post-traitement — l'autre est la référence
Python de `scripts/pipeline/run_reference.py`. **Tout changement ici doit être
validé contre la référence Python** (`tests/check_cpp_detector_fp32.py`) :
c'est le seul moyen raisonnable de déboguer un décodeur d'ancres.

Les constantes de seuil doivent rester cohérentes avec
`configs/detector.yaml` (`conf_threshold`, `iou_threshold`, `dfl_bins`,
`strides`). Elles sont aujourd'hui dupliquées en dur dans ce fichier : une
divergence silencieuse est possible, à surveiller lors des revues.
