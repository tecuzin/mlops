---
description: Crée un commit Git avec un message riche basé sur la conversation et les changements.
---

# Git Commit (rich)

Crée un commit Git de qualité, avec un message riche qui reflète **le pourquoi**, **le quoi**, et **les validations**.

## Objectif

- Produire un commit clair, lisible, et utile pour l'historique.
- S'appuyer sur la conversation en cours + le diff réel.
- Éviter les commits bruités, incomplets, ou risqués.

## Procédure obligatoire

1. **Analyser l'état Git**
   - Exécuter en parallèle:
     - `git status --short --branch`
     - `git diff --stat`
     - `git diff`
     - `git log --oneline -n 10`

2. **Filtrer les fichiers à committer**
   - Inclure uniquement les changements pertinents pour l'objectif courant.
   - Exclure les secrets et artefacts temporaires (`.env`, credentials, dumps, caches, `__pycache__`, etc.).
   - Si des changements semblent hors périmètre, les signaler avant commit.

3. **Construire un message de commit riche**
   - Sujet court (impératif, précis), puis corps structuré.
   - Le corps doit couvrir:
     - **Contexte**: pourquoi ce changement est nécessaire.
     - **Changements clés**: principaux axes modifiés (pas une liste brute de fichiers).
     - **Validation**: tests/lints/commands exécutés.
     - **Impact**: compatibilité, migration, risques résiduels si pertinents.

4. **Template du message**
   - Utiliser ce format:

```text
<type>: <résumé court orienté intention>

Contexte:
- ...

Changements:
- ...
- ...

Validation:
- ...

Impact:
- ...
```

Types recommandés: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.

5. **Créer le commit**
   - Stage des fichiers ciblés.
   - Commit avec HEREDOC:

```bash
git add <fichiers>
git commit -m "$(cat <<'EOF'
<message riche>
EOF
)"
```

6. **Contrôle final**
   - Exécuter `git status --short --branch`.
   - Afficher un récapitulatif:
     - hash du commit
     - sujet
     - fichiers inclus
     - validations exécutées

## Règles de qualité

- Ne jamais inventer des validations non exécutées.
- Ne pas utiliser de message générique type "update files".
- Préférer 1 commit cohérent plutôt que plusieurs commits bruités.
- Si le diff est trop large, proposer un split en commits logiques avant d'exécuter.
