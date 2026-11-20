# `cpp/wildlife_pipeline/` — instantané de référence

**Ne pas éditer ces fichiers.** Ils sont **générés** par
`scripts/pipeline/build_standalone_cpp.py`, qui construit `main.cpp`,
`CMakeLists.txt` et les deux scripts de version par concaténation de chaînes,
en y interpolant les noms de classes et les facteurs d'échelle int8 relevés
dans l'export du détecteur.

Ce qui est versionné ici est un **instantané de la dernière sortie connue**, à
deux fins :

- **relire le résultat.** Le générateur porte son C++ dans une f-string de
  ~260 lignes : relire la sortie est bien plus simple que relire le gabarit ;
- **diffuser les changements.** Quand le générateur évolue, le diff de cet
  instantané montre ce qui change réellement dans le C++ produit.

La vraie cible de build est **`exports/wildlife_pipeline/`**, non versionnée.
Le générateur commence par en faire un `rmtree` — raison pour laquelle il
n'écrit pas ici.

```bash
make quantize     # produit exports/detector_int8/ et exports/classifier_int8/
make build-cpp    # assemble et compile exports/wildlife_pipeline/
exports/wildlife_pipeline/build/wildlife_pipeline data/samples/test_wildlife_1mp.jpg
```

Pour modifier le C++ du pipeline, éditer la f-string de
`scripts/pipeline/build_standalone_cpp.py`, relancer `make build-cpp`, puis
recopier la sortie ici pour tenir l'instantané à jour.

> À terme, ce C++ devrait vivre dans un fichier gabarit plutôt que dans une
> f-string, et les facteurs d'échelle int8 devraient être écrits en JSON par
> l'étape de quantification au lieu d'être relus par expression régulière.
> Voir « Ce que cette structure n'a pas changé » dans
> [docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md).
