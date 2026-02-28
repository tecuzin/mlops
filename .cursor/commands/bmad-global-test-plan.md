---
description: Genere un plan de test global orienté risque pour consolidation multi-branches.
---

# BMAD Global Test Plan

A partir des branches candidates au merge, construis un plan de test global orienté risque.

Inclure au minimum:
- smoke tests
- tests fonctionnels par domaine (API/UI/DB/workers)
- tests d'integration inter-services
- tests de non-regression
- tests securite/perf de base si applicables

Format par test (obligatoire):
- Objectif
- Prerequis
- Etapes
- Resultat attendu
- Critere Go/No-Go

Sortie attendue:
1. Priorisation des tests (P0/P1/P2)
2. Checklist executable (copier-coller)
3. Risques residuels et limites de couverture
