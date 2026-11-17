# Carte, mémoire et kit d'Alif : ce qu'on a vérifié

- **Auteur** : LeslieY0919
- **Date** : 2026-11-17
- **État** : [SOURCÉ] pour ce qui est lu dans le code du kit d'évaluation ML d'Alif et dans SETOOLS ;
  estimations de Vela pour les temps et la mémoire. **Rien n'est encore mesuré sur la carte** : aucun de nos
  firmwares n'a été flashé.

Le « kit d'Alif » est ici le dépôt public `alif_ml-embedded-evaluation-kit` (TFLM + Ethos-U), pas le SDK du
porteur (`alif_aidge_sdk`), que je n'ai pas encore essayé.

## En bref

1. Trois valeurs [POSÉ] de CLAUDE.md §6 se retrouvent dans le code du kit d'Alif : caméra 1280×720, LCD 480×800,
   console applicative à 115 200 bauds sur UART2.
2. Notre carte n'expose qu'**un** port série ; ce sont les cavaliers J15 qui choisissent entre l'UART du Secure
   Enclave et la console applicative.
3. Les 8 Mo de SRAM et 4,5 Mo de MRAM correspondent à **un** cœur. Dans le kit d'Alif, les deux cœurs se
   recouvrent : HP + HE en même temps demande de découper la mémoire.
4. Sur le NPU, le détecteur coûte bien plus que le classifieur, et SiLU et le PSA en sont les deux postes les
   plus chers.

## Valeurs [POSÉ] de CLAUDE.md §6

| | Valeur supposée | Ce qu'on trouve | Où |
|---|---|---|---|
| `REQ-107` | caméra 1280×720 | le kit d'Alif configure le MT9M114 en 1280×720, RGB565 | `RTE_Device.h` du kit (configuration d'image n° 2) |
| `REQ-112` | LCD 480×800 | dalle ILI9806E, 480×800 | `RTE_Device.h` du kit |
| `REQ-113` | console à 115 200 bauds | dans le kit d'Alif : UART2, 115 200 bauds | `CONSOLE_UART` et `uart_tracelib.c` du kit |

Ces valeurs viennent du code, pas d'une observation sur la carte, et le SDK du porteur peut en choisir d'autres.

## Le port série

- [MESURÉ] Sur notre carte (AppKit avec J-Link embarquée), macOS ne voit qu'un seul port `/dev/cu.usbmodem*`, et
  SETOOLS dialogue avec la puce par ce port (`maintenance`, lecture de la révision). Avec les cavaliers dans leur
  position d'origine, ce port est donc l'UART du Secure Enclave.
- [SOURCÉ] D'après le guide de la carte (Alif AUGD0020), les cavaliers **J15** relient ce port soit à SEUART,
  soit à UART2. Pour lire la console de l'application, il faut les déplacer sur UART2, puis les remettre sur
  SEUART avant de flasher.
- [SOURCÉ] SETOOLS utilise 55 000 bauds pour notre révision B4 (`utils/featuresDB.db`).

Cela rejoint la mesure de `connect.py --scan` (fenêtre 55 000–60 000, flux qui n'est pas une trace
d'application) : ce port serait l'UART du Secure Enclave, et non la console applicative. **À vérifier** en
déplaçant J15 sur UART2 avec un firmware qui écrit sur sa console ; le firmware de démonstration compilé par
`make board-firmware` le fait, à 115 200 bauds.

## Mémoire

[SOURCÉ] SETOOLS annonce pour la puce « 5.5 MRAM / 13.5 SRAM ». Avec le script d'édition de liens du kit d'Alif
pour le cœur HP (`RTSS-HP.ld`), une application voit :

| Zone | Taille | Usage dans le kit |
|---|---|---|
| MRAM | 4,46 Mio | programme + poids des modèles |
| SRAM0 | 4 Mio | image caméra, image RGB, tampon LCD |
| SRAM1 | 2,5 Mio | arène des tenseurs |
| DTCM / ITCM | 1 Mio / 256 Kio | données et code rapides |

Soit environ 7,75 Mio de SRAM et 4,5 Mio de MRAM : les budgets de CLAUDE.md §2.

Point d'attention pour HP + HE : dans ce kit, `RTSS-HP.ld` et `RTSS-HE.ld` partent de la **même adresse MRAM**
(`0x80008000`) et utilisent les **mêmes** SRAM0 et SRAM1. Ils sont faits pour une application à la fois. Deux
binaires simultanés demandent de répartir MRAM et SRAM entre les deux cœurs (travaux ouverts n° 2 et n° 4 de
CLAUDE.md §7). Je n'ai pas regardé comment le SDK du porteur le fait.

## Temps et mémoire sur le NPU (estimations de Vela)

Mesures faites sur **YOLOv10n**, de la même famille que `MDV6-yolov10-c` (qui reste à profiler : il faut son
`.onnx` ou son `.pt`). Alif indique que les temps réels sur carte valent environ 2 fois l'estimation de Vela.
Tableau complet dans [AGENTS.md](../AGENTS.md#mesures-vela-sur-lethos-u55).

| Détecteur sur HP (U55-256) | 320×320 | 416×416 | 640×640 |
|---|---|---|---|
| YOLOv10n d'origine (SiLU + PSA) | 48 ms, 5,3 Mio de SRAM | – | – |
| YOLOv10n avec ReLU, sans PSA | 16 ms, 0,7 Mio | 27 ms, 1,2 Mio | 63 ms, 2,5 Mio |

| Classifieur | HP (U55-256) | HE (U55-128) | SRAM | MRAM |
|---|---|---|---|---|
| MobileNetV2-0.35, 128×128 | 1,6 ms | 4,8 ms | 0,25 Mio | 0,50 Mio |
| MobileNetV2-0.5, 160×160 | 2,8 ms | 9,2 ms | 0,39 Mio | 0,77 Mio |

Ce que j'en retiens :

- **Arène du détecteur.** Tel quel en 320, YOLOv10n demande 5,3 Mio de SRAM, soit le double de l'arène de 2,5 Mo
  prévue (REQ-102). Avec ReLU à la place de SiLU, le pic tombe à 0,7 Mio en 320, PSA conservé ; en 416, c'est le
  PSA qui le fait remonter à 9,5 Mio.
- **SiLU et PSA.** SiLU prend environ la moitié du temps NPU (48 ms → 23 ms en 320 en passant à ReLU : SiLU
  s'exécute comme une table suivie d'une multiplication, alors que ReLU se fusionne dans la convolution) ; le
  PSA environ 30 % du reste (23 ms → 16 ms). Les retirer change le modèle : il faudrait le réentraîner, ce que
  CLAUDE.md §10 exclut. `scripts/detector/finetune.py` neutralise déjà l'attention du PSA. Je pose la question
  dans [questions-ouvertes.md](questions-ouvertes.md).
- **Classifieur.** Un MobileNetV2 réduit tient largement dans 1 Mo de MRAM et coûte quelques millisecondes,
  même sur le HE. Je n'ai pas mesuré en 96×96.
- **HP et HE.** Le même modèle est 3 à 4 fois plus lent sur le HE, mais pour le classifieur l'écart (environ
  3 ms par animal) est négligeable devant le détecteur.

## Limites des exemples du kit d'Alif

[SOURCÉ] Lu dans `alif_img_class` et `alif_object_detection`. Elles ne comptent que si l'on part de ce kit ;
je ne sais pas si le SDK du porteur a les mêmes.

- L'image 1280×720 est **recadrée au carré central 720×720 puis réduite**, dans son propre tampon : l'image
  d'origine est écrasée, et l'entrée du modèle est limitée à 320×320. On ne peut donc pas, en l'état, découper
  les boîtes dans l'image d'origine pour le classifieur.
- Le seul décodeur de détection est celui de YOLO-Fastest (ancres + NMS). Il n'y a pas de décodeur YOLOv10.
- L'inférence ne se lance qu'à l'appui sur le joystick.
- Le kit accepte un cas d'usage placé hors de son dossier (variable CMake `USE_CASES_SEARCH_DIR_LIST`).

## Reproduire

```sh
.venv-yolo/bin/python tools/u55/profile_yolov10_u55.py --imgsz 320 416     # détecteur
.venv-yolo/bin/python tools/u55/profile_classifier_u55.py                  # classifieurs candidats
```

Environnement et versions dans [AGENTS.md](../AGENTS.md#environnement-de-la-carte) ; résultats et rapports par
couche dans `artifacts/u55_profile/`.
