# `data/samples/` — images de référence

Le seul dossier de données versionné. Quelques centaines de kilo-octets, que
les contrôles de `tests/` et les outils de `tools/` utilisent comme entrée
fixe. Une entrée stable est ce qui rend comparables deux exécutions séparées
par un changement de code.

| Image | Rôle |
|---|---|
| `test_wildlife_1mp.jpg` | image 1 Mpx de référence. `mountain.paths.SAMPLE_IMAGE` y pointe, et presque tous les contrôles s'en servent. 1 Mpx en RGB, c'est 3 Mo — la moitié de la SRAM de l'E7 d'un seul coup, ce qui en fait un cas de dimensionnement réaliste. |
| `ecureuil-roux.jpg` | gros plan d'écureuil roux, pour contrôler le classifieur seul sur une imagette déjà cadrée. |

Les sorties annotées produites à partir de ces images vont dans `artifacts/`,
qui n'est pas versionné.
