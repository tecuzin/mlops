---
description: Produit un runbook de merge vers une branche unique avec checkpoints QA.
---

# BMAD Merge Playbook

Tu dois produire un playbook de merge concret pour converger vers UNE seule branche cible.

Contraintes Git:
- pas de reset destructif
- pas de force-push
- pas de modification git config
- pas de commit implicite

Attendus:
1. Nom recommande pour la branche cible d'integration
2. Ordre optimal des merges
3. Commandes Git exactes, etape par etape
4. Checkpoints obligatoires apres chaque merge:
   - validations lint/tests/build disponibles
   - decision STOP/GO explicite
5. Strategie de resolution de conflit:
   - API/schemas d'abord
   - DB ensuite
   - Docker/workers
   - UI en dernier
6. Plan de rollback simple en cas d'echec

Format de sortie:
- A) Resume executif
- B) Procedure numerotee
- C) Checkpoints QA
- D) Rollback
- E) Questions bloquantes
