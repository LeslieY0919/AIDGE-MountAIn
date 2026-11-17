# Notes d'échange

Ce dossier sert à **se tenir au courant entre membres du groupe** : ce que chacun a essayé, mesuré ou
découvert, les pièges rencontrés et les questions à trancher ensemble.

Ce n'est pas de la documentation de référence. Le contexte du défi et les décisions sont dans
[CLAUDE.md](../CLAUDE.md), l'organisation du dépôt dans [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md),
l'outillage de la carte dans [AGENTS.md](../AGENTS.md). Ici, ce sont des notes de travail : elles peuvent être
incomplètes ou fausses, et elles sont faites pour être lues, discutées et corrigées.

## Comment s'en servir

- **Une note par sujet**, nommée `AAAA-MM-JJ-sujet.md`.
- Commencer par trois lignes : **auteur**, **date**, **état**. Pour l'état, reprendre les marques de CLAUDE.md
  et dire à quoi elles s'appliquent : **[MESURÉ]** (observé en exécutant quelque chose), **[SOURCÉ]** (lu dans
  une documentation ou dans du code), **[POSÉ]** (hypothèse, à confirmer).
- Rester court : ce qu'on a essayé, ce qu'on a observé, ce que ça change pour le projet, ce qui reste à
  vérifier, et comment reproduire.
- **Pour réagir** : compléter la note en signant son ajout, ou ouvrir une issue ou une PR. Une question
  posée dans [questions-ouvertes.md](questions-ouvertes.md) reste dans la liste jusqu'à ce que quelqu'un
  y réponde.
- Une note ne décide rien. Quand une conclusion est acquise, la reporter dans CLAUDE.md, le README ou
  AGENTS.md, et le signaler dans la note.

## Notes

| Date | Note | État |
|---|---|---|
| 2026-11-17 | [Aidge → TOSA → Vela : exporter les modèles pour l'Ethos-U55](2026-11-17-aidge-vers-npu.md) | mesuré sur PC avec les outils du porteur, rien sur la carte |
| 2026-11-17 | [Carte, mémoire, kit d'Alif et SDK du porteur : ce qu'on a vérifié](2026-11-17-contraintes-carte-et-kit.md) | lu dans le code du kit et du SDK, estimations Vela |
| en continu | [Questions ouvertes](questions-ouvertes.md) | questions |
