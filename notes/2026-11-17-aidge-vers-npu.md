# Aidge → TOSA → Vela : exporter un classifieur pour l'Ethos-U55

- **Auteur** : LeslieY0919
- **Date** : 2026-11-17
- **État** : [MESURÉ] sur PC, dans l'image `docker/aidge/` (Aidge 0.10.1, `aidge_tosa` **officiel** à un commit
  figé, Vela 5.1.0). Les temps et la mémoire sont des estimations de Vela. **Rien n'a tourné sur la carte**, et
  je n'ai essayé ni le fork `aidge-tosa-fork` ni le SDK du porteur.

## En bref

1. La chaîne Aidge (PTQ int8) → `aidge_tosa` → Vela **fonctionne** : un MobileNetV2 complet passe à 100 % sur le
   NPU, sans opérateur renvoyé au CPU.
2. Mais avec l'`aidge_tosa` officiel, le résultat est **inutilisable tel quel** : 77 ms et 9,2 Mio de SRAM pour
   un MobileNetV2 en 224, contre 10,7 ms et 1,4 Mio pour le même réseau passé par TFLite. Le graphe reste en
   NCHW et chaque convolution est encadrée de deux Transpose.
3. Exporter directement en NHWC supprime ces Transpose (4 fois plus rapide sur un petit réseau), mais échoue sur
   MobileNetV2 : le remplissage des convolutions en profondeur est appliqué sur les mauvais axes.
4. Prochaine étape : refaire la même mesure avec le fork du porteur (branche `ethos_u55_support_branch`).

## Ce qui a été essayé

Script : `scripts/npu/export_classifier.py`. Il charge un ONNX, le quantifie en int8 avec la PTQ d'Aidge, puis
appelle `aidge_export_arm_ethos`, qui écrit le fichier TOSA et lance Vela avec la configuration du cœur HP
(U55-256, poids en MRAM, activations en SRAM). Il compare aussi la classe prédite par le graphe int8 à celle
du modèle flottant.

Modèles : un petit réseau de test (quelques convolutions, 32×32) et `mobilenetv2-7.onnx` (MobileNetV2-1.0 en
224, 1000 classes, publié par Aidge sur Hugging Face). Calibration sur 10 images seulement : assez pour faire tourner la
chaîne, pas pour juger la précision.

## Résultats

MobileNetV2-1.0 en 224, cœur HP (U55-256) :

| Chemin | Opérateurs NPU / CPU | Temps | Pic SRAM | MRAM |
|---|---|---|---|---|
| TFLite → Vela (modèle fourni avec le kit d'Alif) | – | 10,7 ms | 1,44 Mio | 3,29 Mio |
| Aidge → TOSA → Vela, entrée NCHW | 472 / 0 | 77,0 ms | 9,19 Mio | 2,98 Mio |
| Aidge → TOSA → Vela, entrée NHWC | échec à la validation de Vela | – | – | – |

Les deux MobileNetV2 ne sont pas exactement le même fichier (0,30 et 0,43 GMAC), mais l'écart vient surtout des
Transpose : 472 opérateurs au lieu d'une centaine.

Petit réseau de test :

| Entrée | Opérateurs NPU | Temps | Pic SRAM |
|---|---|---|---|
| NCHW | 31 | 0,09 ms | 32 Kio |
| NHWC | 8 | 0,02 ms | 8,6 Kio |

Précision : sur 10 images, le graphe int8 donne la même classe que le flottant 8 fois sur 10 pour MobileNetV2
(10 sur 10 pour le petit réseau). À refaire avec un vrai jeu de calibration et de validation.

## Ce qu'il faut savoir

- **NCHW ou NHWC.** Le tenseur d'exemple passé à l'export fixe la disposition. En NCHW (ce que fait le tutoriel
  `mobilenetv2_alif_hp.ipynb`), le graphe reste en NCHW et `aidge_tosa` ajoute des Transpose autour de chaque
  convolution. En NHWC, `aidge_tosa` convertit tout le graphe, et l'entrée du modèle devient une image
  entrelacée, ce qu'attend le code caméra.
- **L'échec en NHWC.** Aidge signale `Wrong input channel size ([1, 112, 114, 34]) for ConvDepthWise` alors que
  la bonne forme est `[1, 114, 114, 32]` : le remplissage est ajouté sur la largeur et les canaux au lieu de la
  hauteur et la largeur. Vela refuse ensuite le fichier (53 erreurs : formes de sortie et de biais incohérentes). Les
  convolutions ordinaires passent, d'où le succès du petit réseau. Si le fork ne corrige pas ce point, c'est une
  contribution possible à Aidge.
- **Entrée du modèle.** La PTQ d'Aidge suppose une entrée flottante dans [-1, 1], quantifiée en `q = 127·x`.
  Le classifieur, lui, est entraîné avec la normalisation ImageNet. Le script replie cette normalisation dans la
  première convolution : le modèle exporté prend directement `q = pixel − 128`, ce que fournit le kit d'Alif.
  Seul le bord de l'image change légèrement (le remplissage vaut 128 au lieu de la moyenne).
- **Pas de paramètres de quantification dans le `.tflite`.** Le fichier produit par Vela n'a ni échelle ni
  zéro sur son entrée et sa sortie. Un code qui les lit pour convertir les scores (comme les exemples TFLM)
  risque de mal les convertir ; l'ordre des classes, lui, ne dépend pas de l'échelle. À vérifier sur la carte.
- **Deux pièges d'installation.** `import aidge_core` doit précéder `import aidge_quantization`, sinon l'import
  échoue (`unknown base type Aidge::OperatorTensor`). Et `aidge_tosa.export` appelé directement prend par
  défaut une version de TOSA qui ne correspond pas à la bibliothèque de sérialisation ; passer par
  `aidge_export_arm_ethos.export`, qui donne la bonne.
- **Opérateurs.** En int8, l'`aidge_tosa` officiel gère Conv, DWConv, pooling, FC, ReLU, Add, Reshape et
  Transpose. Il lui manque Concat, Slice, Resize, Pad, Sigmoid/Mul, Softmax et MatMul : le détecteur ne passe
  pas. D'après CLAUDE.md §3, c'est ce que le fork ajoute.

## Ce que ça change

- Pour le classifieur, la chaîne Aidge → NPU est praticable, à condition de se débarrasser des Transpose : 9 Mio
  de SRAM ne tiennent dans aucune arène de la carte.
- Avant d'aller plus loin avec l'`aidge_tosa` officiel, il faut essayer le fork et le SDK du porteur : ils
  règlent peut-être déjà la disposition NHWC et les opérateurs du détecteur.

## À vérifier

- Mêmes mesures avec `aidge-tosa-fork` (branche `ethos_u55_support_branch`) et avec `alif-aidge-export`.
- Le vrai classifieur du projet, dès que son `.onnx` est partagé.
- L'exécution sur la carte : le `.tflite` exporté n'a encore jamais été flashé.

## Reproduire

Dans l'environnement de `docker/aidge/Dockerfile`, depuis la racine du dépôt :

```sh
python scripts/npu/export_classifier.py --onnx models/mobilenetv2-7.onnx --size 224                       # NCHW
python scripts/npu/export_classifier.py --onnx models/mobilenetv2-7.onnx --size 224 --input-format nhwc   # échoue
```

Sur une machine sans wheel Aidge (macOS), préfixer par `scripts/npu/aidge_docker.sh run`. Le modèle vient de
<https://huggingface.co/EclipseAidge/mobilenet_v2>. Résultats dans `exports/npu/<modèle>/` : `model.tosa`,
`output/model_vela.tflite` et `result.json`.
