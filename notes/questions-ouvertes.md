# Questions ouvertes

Liste tenue à jour. Répondre directement sous la question en signant, puis déplacer la question dans
« Réponses » une fois tranchée. Une question garde son numéro (Q1, Q2…) quand elle change de section.

## En attente

- **Q2 — Peut-on partager les modèles entraînés** (`detector_v6_trained_640.onnx`, `species_classifier.onnx`),
  par exemple dans une GitHub Release ? Il me les faut pour mesurer `MDV6-yolov10-c` avec Vela (jusqu'ici, les
  mesures portent sur YOLOv10n) et pour tester l'export vers le NPU sur le vrai classifieur.
  — LeslieY0919, 2026-11-17

- **Q3 — Qui réentraîne le classifieur en MobileNetV2 ?** Le README note déjà l'écart : ResNet-34 en 128×128 au
  lieu de MobileNetV2 en 96×96. Côté NPU, un MobileNetV2-0.35 en 128×128 pèse 0,5 Mio et prend 1,6 ms (Vela) :
  la taille d'entrée peut rester à 128 si la précision en profite. Le SDK attend un ONNX dont l'entrée est dans
  [0, 1] (voir Q6 dans les réponses).
  — LeslieY0919, 2026-11-17

- **Q4 — Première démonstration : tout sur le HP, ou HP + HE dès le début ?** CLAUDE.md place le classifieur sur
  le HE. D'après Vela, il y prendrait 4,8 ms au lieu de 1,6 ms : négligeable. Mais il faut deux binaires, un
  découpage de la mémoire et une synchronisation. Je propose de faire tourner d'abord les deux modèles l'un après
  l'autre sur le HP, pour avoir une démonstration, puis de déplacer le classifieur sur le HE.
  — LeslieY0919, 2026-11-17

  Complément : le SDK du porteur flashe **une seule image à la fois**. HP et HE sont écrits à la même adresse
  MRAM, et la seconde image remplace la première (`tools/flash_mram.sh`). HP + HE demande donc un travail que le
  SDK ne fait pas encore : répartir MRAM et SRAM, écrire deux entrées dans la TOC, synchroniser les cœurs.
  Détails dans [la note sur la carte](2026-11-17-contraintes-carte-et-kit.md#mémoire).
  — LeslieY0919, 2026-11-17

- **Q5 — Jusqu'où peut-on modifier le détecteur ?** Le porteur demande de ne pas réentraîner MegaDetector. Or,
  sur l'Ethos-U55, SiLU prend environ la moitié du temps NPU (mesures sur YOLOv10n). Remplacer SiLU par ReLU
  demande un fine-tuning. Est-ce acceptable pour le porteur, ou faut-il chercher du côté du tiling et de la
  taille d'entrée uniquement ? À poser à Olivier.
  — LeslieY0919, 2026-11-17

  Correction : j'avançais aussi que le modèle d'origine demandait 5,3 Mio de SRAM en 320, pour une arène de
  2,5 Mo. Ce chiffre vient de la conversion TFLite. Par Aidge et le fork, le pic est de 0,79 Mio pour 54 ms
  ([note sur l'export](2026-11-17-aidge-vers-npu.md)). Il ne reste donc que l'argument du temps.
  — LeslieY0919, 2026-11-17

- **Q7 — Historique git** : les exports générés (1,2 Go de fichiers) restent dans l'historique, le `.git` pèse
  environ 300 Mo. Les retire-t-on en réécrivant l'historique (push forcé, chacun reclone) ?
  — LeslieY0919, 2026-11-17

- **Q8 — Questions pour le porteur**, relevées en lisant ses dépôts. Je propose de les envoyer ensemble, avec Q5.
  - **HP + HE.** Le SDK flashe une image à la fois. A-t-il un exemple, ou un plan, pour faire tourner un modèle
    sur chaque cœur ? Un commentaire du code caméra cite un `alif_mdv6_runner` : est-il disponible ?
  - **`aidge_export_ethos`.** Le README du SDK installe le dépôt officiel. Or `ethos_be`, sans lequel Vela
    refuse un modèle avec Softmax ou Sigmoid, n'existe que sur la branche `feat/ethos-be-lowering` de son fork.
    Faut-il installer cette branche ?
  - **Slice dans le fork.** Un Slice sur l'axe des canaux est mal converti lors du passage en NHWC : il découpe
    la hauteur. Le Split équivalent, lui, passe. On contourne le problème avec
    `scripts/npu/onnx_slices_to_split.py`. Est-ce connu ?
  - **Mesures sur carte.** Le README de l'exemple MobileNetV2 renvoie à un `NOTES.md` du SDK, absent du dépôt
    public. Peut-on l'avoir ?

  — LeslieY0919, 2026-11-17

## Réponses

- **Q1 — Le SDK du porteur fournit-il caméra, LCD, export et flash ?** Oui :
  - l'export Aidge (`alif-aidge-export` : PTQ int8, fork `aidge_tosa`, Vela, contrôle par le modèle de référence
    TOSA) ;
  - la génération des sources du modèle (`alif-aidge-gen-model`) ;
  - un runtime C++ : caméra, LCD, TFLM + Ethos-U, compteurs du NPU ;
  - la compilation (`alif_add_runner`) et le flash (`tools/flash_mram.sh`).

  Son export est vérifié sur MobileNetV2 et sur YOLOv10n ; son firmware n'a pas encore été compilé ni flashé.
  Je bascule donc sur le SDK, que CLAUDE.md §3 désigne déjà, plutôt que de continuer sur le kit d'Alif. Détails dans
  [la note sur l'export](2026-11-17-aidge-vers-npu.md) et dans
  [la note sur la carte](2026-11-17-contraintes-carte-et-kit.md). — LeslieY0919, 2026-11-17

  Complément : le 2026-11-18, l'exemple MobileNetV2 du SDK a été compilé et flashé, et l'image de la caméra
  s'affiche. Marche à suivre : [docs/GUIDE_CARTE.md](../docs/GUIDE_CARTE.md). — LeslieY0919, 2026-11-19

- **Q6 — Quelle convention d'entrée pour les modèles ?** Celle du SDK :
  - l'ONNX prend une image dans [0, 1], en NCHW ;
  - la carte lui envoie `code = round(pixel / 255 × 127)`, avec un zéro à 0, en NHWC.

  Un classifieur entraîné avec la normalisation ImageNet n'a pas à être réentraîné : on replie la normalisation
  dans la première convolution avant l'export, comme `examples/mobilenetv2/model/prepare_onnx.py` du SDK.
  `scripts/npu/export_classifier.py` suit une autre convention (`pixel − 128`, celle du kit d'Alif) : ne pas
  l'utiliser pour le SDK. — LeslieY0919, 2026-11-17

- **Quelle version d'Aidge ?** `>=0.10.1,<0.11`, épinglée dans `pyproject.toml` (commit `6be6ceb`). L'image
  `docker/aidge/` est en 0.10.1. — 2026-11-17

- **A-t-on reçu des outils des organisateurs ?** Oui : le SDK `alif_aidge_sdk`, l'exemple `alif_yolo_fastest`
  et le fork `aidge-tosa-fork`, listés dans CLAUDE.md §3. — 2026-11-17

- **Quelle licence pour le code ?** AGPL v3 (REQ-030), à cause du modèle YOLOv10 d'Ultralytics. Le rapport et la
  vidéo du challenge sont, eux, publiés sous CC BY 4.0 d'après le règlement. — 2026-11-17
