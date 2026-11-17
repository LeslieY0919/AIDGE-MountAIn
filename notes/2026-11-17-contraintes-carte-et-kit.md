# Carte, mémoire, kit d'Alif et SDK du porteur : ce qu'on a vérifié

- **Auteur** : LeslieY0919
- **Date** : 2026-11-17, complétée le même jour avec le SDK du porteur
- **État** : [SOURCÉ] pour ce qui est lu dans le code du kit d'évaluation ML d'Alif, de SETOOLS et du SDK du
  porteur (`alif_aidge_sdk`, `eb35a81`). Les temps et la mémoire sont des estimations de Vela. **Rien n'est
  encore mesuré sur la carte** : aucun firmware, ni le nôtre ni celui du SDK, n'a été flashé.

Deux bases de firmware sont en jeu :

- le **kit d'Alif** : le dépôt public `alif_ml-embedded-evaluation-kit` (TFLM + Ethos-U), compilé par
  `make board-firmware` ;
- le **SDK du porteur**, que CLAUDE.md §3 désigne comme chaîne outil. Il reprend les briques du kit d'Alif
  (caméra, LCD, TFLM, scripts d'édition de liens) et y ajoute l'export Aidge, la génération des sources du modèle,
  un runtime C++ et le flash. Je l'ai lu et j'ai essayé son export
  ([note sur l'export](2026-11-17-aidge-vers-npu.md)), mais je ne l'ai ni compilé ni flashé.

## En bref

1. Trois valeurs [POSÉ] de CLAUDE.md §6 se retrouvent dans le code du kit d'Alif comme dans celui du SDK :
   caméra 1280×720, LCD 480×800, console applicative à 115 200 bauds sur UART2.
2. Notre carte n'expose qu'**un** port série ; ce sont les cavaliers J15 qui choisissent entre l'UART du Secure
   Enclave et la console applicative.
3. Les 8 Mo de SRAM et 4,5 Mo de MRAM correspondent à **un** cœur. Le SDK, comme le kit d'Alif, place les
   images HP et HE à la même adresse et flashe **une seule image à la fois** : HP + HE en même temps n'est pas
   prévu tel quel.
4. Le SDK règle deux limites des exemples du kit d'Alif : l'inférence tourne en continu et l'entrée peut dépasser
   320×320. En revanche, il ne donne ni l'image en pleine résolution ni de décodeur YOLOv10.
5. Sur l'AppKit, le SDK suppose par défaut une caméra ARX3A0 : il faut choisir le MT9M114 à la compilation.

## Valeurs [POSÉ] de CLAUDE.md §6

| | Valeur supposée | Kit d'Alif | SDK du porteur |
|---|---|---|---|
| `REQ-107` | caméra 1280×720 | MT9M114 en 1280×720, RGB565 (`RTE_Device.h`, configuration d'image n° 2) | même configuration n° 2, si l'on choisit le MT9M114 |
| `REQ-112` | LCD 480×800 | dalle ILI9806E, 480×800 (`RTE_Device.h`) | 480×800 (`runtime/board.hpp` : « 480 x 800 on the AppKits ») |
| `REQ-113` | console à 115 200 bauds | UART2, 115 200 bauds (`CONSOLE_UART`, `uart_tracelib.c`) | idem (« 2 = UARTA on the AppKits ») |

Ces valeurs viennent du code, pas d'une observation sur la carte.

## Le port série

- [MESURÉ] Sur notre carte (AppKit avec J-Link embarquée), macOS ne voit qu'un seul port `/dev/cu.usbmodem*`, et
  SETOOLS dialogue avec la puce par ce port (`maintenance`, lecture de la révision). Avec les cavaliers dans leur
  position d'origine, ce port est donc l'UART du Secure Enclave.
- [SOURCÉ] D'après le guide de la carte (Alif AUGD0020), les cavaliers **J15** relient ce port soit à SEUART,
  soit à UART2. Pour lire la console de l'application, il faut les déplacer sur UART2, puis les remettre sur
  SEUART avant de flasher.
- [SOURCÉ] SETOOLS utilise 55 000 bauds pour notre révision B4 (`utils/featuresDB.db`). Le script de flash du SDK
  prend aussi 55 000 bauds par défaut (« SETOOLS' own default of 57600 does not reach it »).
- [SOURCÉ] Les règles udev du SDK (`tools/99-alif-devkit.rules`) décrivent la même chose : le port série de la
  J-Link est l'UART du Secure Enclave, et sur la DevKit la console de l'application passe par un second câble
  (FTDI).

Cela rejoint la mesure de `connect.py --scan` (fenêtre 55 000–60 000, flux qui n'est pas une trace
d'application) : ce port serait l'UART du Secure Enclave, et non la console applicative. **À vérifier** en
déplaçant J15 sur UART2 avec un firmware qui écrit sur sa console ; le firmware de démonstration compilé par
`make board-firmware` le fait, à 115 200 bauds.

## Mémoire

[SOURCÉ] SETOOLS annonce pour la puce « 5.5 MRAM / 13.5 SRAM ». Avec le script d'édition de liens du cœur HP
(`RTSS-HP.ld`, identique dans le kit d'Alif et dans le SDK), une application voit :

| Zone | Taille | Usage dans le kit d'Alif et le SDK |
|---|---|---|
| MRAM | 4,46 Mio | programme + poids des modèles |
| SRAM0 | 4 Mio | image caméra, image RGB, tampon LCD |
| SRAM1 | 2,5 Mio | arène des tenseurs (2 Mio par défaut dans le SDK, `ACTIVATION_BUF_SZ`) |
| DTCM / ITCM | 1 Mio / 256 Kio | données et code rapides (DTCM de 256 Kio sur le HE) |

Soit environ 7,75 Mio de SRAM et 4,5 Mio de MRAM : les budgets de CLAUDE.md §2.

**HP + HE.** Les scripts HP et HE partent de la **même adresse MRAM** (`0x80008000`) et utilisent les **mêmes**
SRAM0 et SRAM1, dans le kit d'Alif comme dans le SDK. Le script de flash du SDK le dit :

> The image is written at the same MRAM address either way and the TOC tells the Secure Enclave which core boots
> it; one image at a time, the TOC replaces the previous one. (`tools/flash_mram.sh`)

`--cpu HP` puis `--cpu HE` ne donne donc pas deux binaires qui tournent ensemble : le second remplace le
premier. Pour le schéma de CLAUDE.md §2 (détecteur sur le HP, classifieur sur le HE), il faudra répartir MRAM et
SRAM entre les deux cœurs, écrire deux entrées dans la TOC et synchroniser les cœurs (travaux ouverts n° 2 et
n° 4 de CLAUDE.md §7). Question posée dans [questions-ouvertes.md](questions-ouvertes.md).

## Ce que fait le SDK du porteur

[SOURCÉ] Lu dans le code, `eb35a81`.

- **Chaîne** :
  - `alif-aidge-export` produit `model.tosa`, `output/model_vela.tflite` et `model_io.json` (cible `hp` ou `he`) ;
  - `alif-aidge-gen-model` génère `model_data.cc` et `model_io.hpp` ;
  - `alif_add_runner` (CMake) compile le firmware ;
  - `tools/flash_mram.sh --cpu HP|HE` le flashe.

  À la configuration, CMake vérifie que le modèle a été exporté pour le cœur compilé.
- **Caméra.**
  - Sur l'AppKit, le module par défaut est l'ARX3A0 (« The AppKit E7 ships with the ARX3A0 »). Notre carte porte
    un MT9M114 (AGENTS.md) : il faut donc passer `-DALIF_CAMERA_MODULE=MT9M114`, comme le font CLAUDE.md §3 et
    `alif_yolo_fastest`. À confirmer au premier essai caméra.
  - La puce n'a pas d'ISP matériel.
  - `board::camera_capture` livre une image RGB888 déjà recadrée au carré central et réduite à la taille du
    modèle. La taille maximale est `max(320, entrée du modèle)` : la limite de 320 disparaît.
  - La caméra peut exposer l'image suivante pendant qu'on traite la précédente.
- **Écran.** `LCD_ROTATE_180` et `LCD_MIRROR` sont actifs par défaut (dalle montée tête en bas ; « the MT9M114
  delivers it mirrored »).
- **Runtime.**
  - TFLM n'enregistre que l'opérateur Ethos-U (`MicroMutableOpResolver<1>` + `AddEthosU`) : un modèle doit passer
    **à 100 % sur le NPU** après Vela, sans aucun opérateur renvoyé au CPU.
  - Les compteurs du NPU (cycles, cycles actifs, accès mémoire) sont lisibles après chaque inférence.
- **Exemple MobileNetV2.** Il tourne en boucle continue, sans joystick. Il affiche le top-5, les images par
  seconde et le temps NPU sur l'écran, et écrit une ligne par image sur la console.
- **Détection.** Le SDK n'a pas de décodeur YOLOv10. `alif_yolo_fastest` a celui de YOLO-Fastest (ancres + NMS),
  avec une référence Python, un build hôte du C++ et un script qui compare les deux : c'est le schéma de
  CLAUDE.md §4.
- Un commentaire du code caméra cite un `alif_mdv6_runner`, que je n'ai trouvé dans aucun dépôt public.

## Temps et mémoire sur le NPU (estimations de Vela)

Mesures faites sur **YOLOv10n**, de la même famille que `MDV6-yolov10-c` (qui reste à profiler : il faut son
`.onnx` ou son `.pt`). Alif indique que les temps réels sur carte valent environ 2 fois l'estimation de Vela.
Tableau complet dans [AGENTS.md](../AGENTS.md#mesures-vela-sur-lethos-u55).

| Détecteur sur HP (U55-256) | 320×320 | 416×416 | 640×640 |
|---|---|---|---|
| YOLOv10n d'origine (SiLU + PSA), par TFLite | 48 ms, 5,3 Mio de SRAM | – | – |
| YOLOv10n d'origine (SiLU + PSA), par Aidge et le fork | 54 ms, 0,79 Mio | – | – |
| YOLOv10n avec ReLU, sans PSA, par TFLite | 16 ms, 0,7 Mio | 27 ms, 1,2 Mio | 63 ms, 2,5 Mio |

| Classifieur (par TFLite) | HP (U55-256) | HE (U55-128) | SRAM | MRAM |
|---|---|---|---|---|
| MobileNetV2-0.35, 128×128 | 1,6 ms | 4,8 ms | 0,25 Mio | 0,50 Mio |
| MobileNetV2-0.5, 160×160 | 2,8 ms | 9,2 ms | 0,39 Mio | 0,77 Mio |

Ce que j'en retiens :

- **Arène du détecteur.** Les 5,3 Mio de SRAM du YOLOv10n d'origine en 320 viennent de la conversion TFLite. Par
  Aidge et le fork, le chemin que demande le défi, le pic est de 0,79 Mio : cela tient dans l'arène de 2,5 Mo
  (REQ-102).
- **SiLU et PSA.** Par TFLite, SiLU prend environ la moitié du temps NPU (48 ms → 23 ms en 320 en passant à ReLU :
  SiLU s'exécute comme une table suivie d'une multiplication, alors que ReLU se fusionne dans la convolution).
  Le PSA prend environ 30 % du reste (23 ms → 16 ms). Par Aidge, le modèle d'origine prend 54 ms : le coût de SiLU
  demeure.
  - Retirer SiLU ou le PSA change le modèle et demande de le réentraîner, ce que CLAUDE.md §10 exclut.
  - `scripts/detector/finetune.py` neutralise déjà l'attention du PSA.
  - Je pose la question dans [questions-ouvertes.md](questions-ouvertes.md).
- **Classifieur.** Un MobileNetV2 réduit tient largement dans 1 Mo de MRAM et coûte quelques millisecondes, même
  sur le HE. Je n'ai pas mesuré en 96×96.
- **HP et HE.** Le même modèle est 3 à 4 fois plus lent sur le HE, mais pour le classifieur l'écart (environ 3 ms
  par animal) est négligeable devant le détecteur.

## Limites des exemples du kit d'Alif, et ce qu'en fait le SDK

[SOURCÉ] Lu dans `alif_img_class` et `alif_object_detection` (kit d'Alif), puis dans le SDK.

| Limite du kit d'Alif | Dans le SDK |
|---|---|
| Image 1280×720 recadrée au carré central 720×720 puis réduite, dans son propre tampon : l'image d'origine est écrasée | inchangé : l'application ne reçoit que l'image à la taille du modèle. Pour découper les boîtes en pleine résolution (classifieur), il faudra passer sous `board::camera_capture` |
| Entrée du modèle limitée à 320×320 | limite levée : `max(320, entrée du modèle)` |
| Seul décodeur de détection : YOLO-Fastest | idem (`alif_yolo_fastest`) ; le décodeur YOLOv10 est à écrire |
| Inférence à l'appui sur le joystick | boucle continue |
| Cas d'usage hors du dossier du kit possible (`USE_CASES_SEARCH_DIR_LIST`) | le projet inclut le SDK comme sous-module et appelle `alif_add_runner` |

## Reproduire

```sh
.venv-yolo/bin/python tools/u55/profile_yolov10_u55.py --imgsz 320 416     # détecteur, par TFLite
.venv-yolo/bin/python tools/u55/profile_classifier_u55.py                  # classifieurs candidats, par TFLite
```

Environnement et versions dans [AGENTS.md](../AGENTS.md#environnement-de-la-carte) ; résultats et rapports par
couche dans `artifacts/u55_profile/`. Le chiffre par Aidge se reproduit comme indiqué dans
[la note sur l'export](2026-11-17-aidge-vers-npu.md#reproduire).
