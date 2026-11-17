# Questions ouvertes

Liste tenue à jour. Répondre directement sous la question en signant, puis déplacer la question dans
« Réponses » une fois tranchée.

## En attente

1. **Qui a déjà essayé le SDK du porteur (`alif_aidge_sdk`) ?** Les trois dépôts de CLAUDE.md §3 sont publics
   (vérifié le 2026-11-17 avec `git ls-remote`). De mon côté, j'ai seulement utilisé le kit public d'Alif et
   l'`aidge_tosa` officiel. Si le SDK fournit déjà caméra, LCD, export et flash, je bascule dessus plutôt que de
   continuer sur le kit d'Alif.
   — LeslieY0919, 2026-11-17

2. **Peut-on partager les modèles entraînés** (`detector_v6_trained_640.onnx`, `species_classifier.onnx`), par
   exemple dans une GitHub Release ? Il me les faut pour mesurer `MDV6-yolov10-c` avec Vela (jusqu'ici, les
   mesures portent sur YOLOv10n) et pour tester l'export vers le NPU sur le vrai classifieur.
   — LeslieY0919, 2026-11-17

3. **Qui réentraîne le classifieur en MobileNetV2 ?** Le README note déjà l'écart : ResNet-34 en 128×128 au lieu
   de MobileNetV2 en 96×96. Côté NPU, un MobileNetV2-0.35 en 128×128 pèse 0,5 Mio et prend 1,6 ms (Vela) : la
   taille d'entrée peut rester à 128 si la précision en profite.
   — LeslieY0919, 2026-11-17

4. **Première démonstration : tout sur le HP, ou HP + HE dès le début ?** CLAUDE.md place le classifieur sur
   le HE. D'après Vela, il y prendrait 4,8 ms au lieu de 1,6 ms : négligeable. Mais il faut deux binaires, un
   découpage de la mémoire (les scripts d'édition de liens HP et HE du kit d'Alif se recouvrent) et une
   synchronisation. Je propose de faire tourner d'abord les deux modèles l'un après l'autre sur le HP, pour
   avoir une démonstration, puis de déplacer le classifieur sur le HE. Détails dans
   [la note sur la carte](2026-11-17-contraintes-carte-et-kit.md).
   — LeslieY0919, 2026-11-17

5. **Jusqu'où peut-on modifier le détecteur ?** Le porteur demande de ne pas réentraîner MegaDetector. Or, sur
   l'Ethos-U55, SiLU prend environ la moitié du temps et, à 320×320, le modèle d'origine demande 5,3 Mio de SRAM
   pour une arène prévue de 2,5 Mo (mesures sur YOLOv10n). Remplacer SiLU par ReLU demande un fine-tuning. Est-ce
   acceptable pour le porteur, ou faut-il chercher du côté du tiling et de la taille d'entrée uniquement ? À
   poser à Olivier.
   — LeslieY0919, 2026-11-17

6. **Quelle convention d'entrée pour le classifieur ?** CLAUDE.md §3 demande un ONNX dont les entrées sont dans
   [0, 1] ; le classifieur est entraîné avec la normalisation ImageNet. `scripts/npu/export_classifier.py` replie
   cette normalisation dans la première convolution. Le SDK du porteur fait-il la même chose, ou faut-il
   entraîner directement avec des entrées dans [0, 1] ?
   — LeslieY0919, 2026-11-17

7. **Fixe-t-on la version d'Aidge dans `pyproject.toml` ?** CLAUDE.md §3 demande `>=0.10.1,<0.11`, mais l'extra
   `ml` ne fixe rien et la 0.11.0 est sortie le 2026-11-06 : un `make install` fait aujourd'hui prendrait donc
   la 0.11. L'image `docker/aidge/` est en 0.10.1.
   — LeslieY0919, 2026-11-17

8. **Historique git** : les exports générés (1,2 Go de fichiers) restent dans l'historique, le `.git` pèse
   environ 300 Mo. Les retire-t-on en réécrivant l'historique (push forcé, chacun reclone) ?
   — LeslieY0919, 2026-11-17

## Réponses

- **A-t-on reçu des outils des organisateurs ?** Oui : le SDK `alif_aidge_sdk`, l'exemple `alif_yolo_fastest`
  et le fork `aidge-tosa-fork`, listés dans CLAUDE.md §3. — 2026-11-17
- **Quelle licence pour le code ?** AGPL v3 (REQ-030), à cause du modèle YOLOv10 d'Ultralytics. Le rapport et la
  vidéo du challenge sont, eux, publiés sous CC BY 4.0 d'après le règlement. — 2026-11-17
