---
name: audit
description: >
  Full senior-dev audit of SmartSplit codebase. Covers clean code (SOLID, DRY, KISS, YAGNI),
  tech debt, Python best practices (type hints, ruff, complexity), security (OWASP, secrets, injection),
  dead code, performance, architecture, and testing. Use when the user says "audit", "code review",
  "clean code", "tech debt", "code quality", "code health", "refactor check", or "is the code clean".
---

# SmartSplit — Audit Complet Senior Dev

Audit exhaustif du codebase en 9 dimensions. Chaque section produit des findings classés par sévérité.
Ne propose PAS de fix — liste les problèmes avec fichier, ligne, sévérité, et justification.

## Sévérités

- **CRITICAL** — bug en prod, faille sécu, perte de données
- **HIGH** — code incorrect, violation d'archi, dette bloquante
- **MEDIUM** — code smell, complexité excessive, convention violée
- **LOW** — nit, style, amélioration mineure
- **PRAISE** — code exemplaire à souligner

---

## Phase 0 — Snapshot du projet

Avant tout, établir le contexte :

1. Lire `CLAUDE.md`, `pyproject.toml` / `setup.cfg` / `Makefile` pour comprendre le stack, les conventions, les outils de lint/test
2. `git log --oneline -20` pour le contexte récent
3. `git diff --stat main...HEAD` si on est sur une branche
4. Lire `.gitignore` pour identifier les répertoires à exclure (`.venv/`, `dist/`, etc.)
5. Lister les fichiers Python **source uniquement** — Glob `**/*.py` dans `smartsplit/`, `tests/`, `scripts/` (JAMAIS à la racine du projet, ça inclut `.venv/`)
6. Compter les lignes par module (identifier les god files > 400 lignes)

Produire un résumé : nb fichiers, nb lignes total, modules les plus gros, stack détecté.

---

## Phase 1 — Clean Code (SOLID / DRY / KISS / YAGNI)

Scanner chaque module pour :

### Single Responsibility (S)
- Fonctions qui font plus d'une chose (indice : "and" dans la description)
- Modules fourre-tout ("utils", "helpers", "misc")
- Classes avec trop de responsabilités (> 5 méthodes publiques non cohésives)

### Open/Closed (O)
- Switch/if-chains qui grandissent à chaque ajout (vérifier les `if isinstance`, `if type ==`)
- Absence de registre/dispatch table là où le pattern est approprié
- Code qui force la modification du noyau pour ajouter un comportement

### Liskov Substitution (L)
- Sous-classes qui lèvent `NotImplementedError` sur des méthodes héritées
- `isinstance` / `type()` checks après avoir accepté un type générique

### Interface Segregation (I)
- ABCs avec trop de méthodes abstraites (consommateurs forcés d'implémenter ce qu'ils n'utilisent pas)
- Imports massifs pour n'utiliser qu'une seule fonction

### Dependency Inversion (D)
- Business logic qui instancie directement des clients HTTP, DB, ou SDKs externes
- Impossibilité de tester sans infra réelle

### DRY
- Logique métier dupliquée (même règle dans 2+ endroits)
- Magic strings / magic numbers répétés sans constante
- Structures parallèles maintenues manuellement

### KISS
- Nesting > 3 niveaux (aplatir avec guard clauses)
- One-liners cryptiques ou comprehensions imbriquées illisibles
- Over-engineering : abstraction pour un seul usage
- Fonctions > 50 lignes

### YAGNI
- Code spéculatif (features non utilisées, paramètres "au cas où")
- Abstractions prématurées sans second consommateur
- Configs pour des cas qui n'existent pas

---

## Phase 2 — Python Best Practices

### Type Hints
- Fonctions sans annotations de type (paramètres ET retour)
- Usage de `Any` (interdit par CLAUDE.md)
- `# type: ignore` (interdit par CLAUDE.md)
- Absence de `from __future__ import annotations` dans un module

### Style & Idiomes
- Non-respect PEP 8 (snake_case fonctions/variables, PascalCase classes, UPPER_CASE constantes)
- `str.format()` avec input utilisateur (interdit — utiliser concaténation)
- Bare `except:` ou `except Exception` trop large
- `print()` au lieu de `logging`
- Mutable default arguments (`def f(x=[])`)
- String concatenation dans des boucles (utiliser join ou list)
- `os.path` au lieu de `pathlib.Path`

### Idiomes Pythoniques
- Boucles `for i in range(len(x))` au lieu de `enumerate` / itération directe
- `if x == True` / `if x == None` au lieu de `if x` / `if x is None`
- Context managers manquants pour les ressources (fichiers, connections)
- Dict/list comprehension là où un `map`/`filter` serait plus clair (ou l'inverse)

### Ruff / Linting
- Exécuter mentalement les règles ruff : imports non triés, imports inutilisés, variables non utilisées
- Vérifier que les f-strings sont préférées aux % formatting et .format()

---

## Phase 3 — Dette Technique

### Catégorisation (formule de priorisation : Priority = (Impact + Risk) × (6 - Effort))
- **Code Debt** — logique dupliquée, abstractions cassées, nommage trompeur
- **Architecture Debt** — couplage excessif, violations de boundaries entre modules
- **Test Debt** — couverture faible, tests flaky, assertions insuffisantes
- **Dependency Debt** — packages outdated, dépendances avec CVE connues
- **Documentation Debt** — docstrings manquantes sur l'API publique, README décalé

### Dead Code
- Imports non utilisés
- Fonctions/classes jamais appelées (chercher les références avec Grep)
- Variables assignées mais jamais lues
- Branches conditionnelles impossibles
- Code commenté (supprimer, git l'a)
- Paramètres de fonctions jamais passés par les callers

### Complexité
- Complexité cyclomatique estimée (> 10 = HIGH, > 20 = CRITICAL)
- Nombre de paramètres par fonction (> 5 = flag)
- Profondeur d'héritage excessive
- Fan-out excessif (un module qui importe 10+ autres modules)

---

## Phase 4 — Sécurité

### OWASP Top 10 (adapté backend Python)
- **Injection** — SQL injection, command injection (`os.system`, `subprocess` avec shell=True), template injection
- **Broken Auth** — secrets en dur, tokens mal gérés
- **Sensitive Data Exposure** — clés API dans les logs, données sensibles non masquées
- **XXE / Deserialization** — `pickle.loads` sur données non fiables, `yaml.load` sans SafeLoader
- **Security Misconfiguration** — debug mode en prod, CORS trop permissif

### Secrets
- Grep pour patterns de secrets : `API_KEY`, `SECRET`, `PASSWORD`, `TOKEN` dans le code source (pas .env)
- Vérifier que `.env` est dans `.gitignore`
- Vérifier qu'aucun secret ne fuit dans les messages d'erreur ou les logs

### Input Validation
- Données utilisateur utilisées sans validation (vérifier les endpoints HTTP)
- Path traversal (vérifier les accès fichiers avec des chemins utilisateur — critique pour `tool_anticipator.py`)
- Taille des inputs non limitée

### Dependencies
- Vérifier `requirements.txt` / `pyproject.toml` pour des versions pinées
- Packages connus pour des CVE (vérifier les versions)

---

## Phase 5 — Architecture & Design

### Structure du Projet
- Séparation claire des responsabilités entre modules
- Respect du pattern Strategy pour les providers (conformité avec `base.py`)
- Cohérence du routing dual-mode (Agent vs API) dans `proxy.py`
- Circuit breaker correctement implémenté dans `registry.py`

### Couplage & Cohésion
- Modules trop couplés (imports circulaires, dépendances croisées excessives)
- Modules peu cohésifs (fonctions sans rapport groupées ensemble)
- God classes / god modules (> 400 lignes = suspect)

### Patterns & Consistency
- Pydantic models partout (zero raw dicts — CLAUDE.md)
- StrEnum au lieu de magic strings
- Exception hierarchy respectée (`exceptions.py`)
- Logging cohérent (même format, même niveau de détail)

### Invariants du Projet (depuis CLAUDE.md)
Vérifier explicitement :
- [ ] SAFE_TOOLS only — aucun tool d'écriture dans `intention_detector.py` ou `tool_anticipator.py`
- [ ] Tool passthrough — les tools du client sont forwarded intacts au brain
- [ ] Context injection — résultats pré-fetchés injectés en début de system message
- [ ] Confidence threshold >= 0.7
- [ ] Brain response jamais modifiée
- [ ] Circuit breaker : 5 fails / 2 min
- [ ] Pas de `str.format()` avec user input

---

## Phase 6 — Performance

### Algorithmes & Data Structures
- Boucles O(n²) cachées (nested loops sur les mêmes données)
- Recherches linéaires là où un dict/set suffirait
- Concaténation de strings dans des boucles

### Async & Concurrency
- Opérations bloquantes dans du code async (`time.sleep` dans un contexte asyncio)
- Awaits manquants (coroutines non awaitées)
- Race conditions sur des structures partagées
- Connection pooling mal configuré

### I/O & Network
- Requêtes HTTP sérielles qui pourraient être parallélisées (`asyncio.gather`)
- Absence de timeout sur les appels réseau
- Réponses volumineuses non compressées ou non streamées
- Cache manquant pour des données fréquemment accédées

### Memory
- Listes entières chargées en mémoire là où un générateur suffirait
- Références circulaires empêchant le GC
- Accumulation de données dans des structures globales

---

## Phase 7 — Tests

### Couverture
- Modules sans aucun test
- Fonctions publiques non testées
- Branches conditionnelles non couvertes (if/else, try/except)
- Edge cases manquants (None, vide, très grand, caractères spéciaux)

### Qualité des Tests
- Tests qui testent l'implémentation plutôt que le comportement
- Assertions faibles (`assert result` au lieu de `assert result == expected`)
- Tests interdépendants (ordre d'exécution importe)
- Mocks excessifs qui masquent des bugs réels
- Tests sans arrange/act/assert clair

### Tests Manquants Critiques
- Path traversal dans `tool_anticipator.py`
- Circuit breaker edge cases dans `registry.py`
- Concurrent requests dans `proxy.py`
- Provider fallback chains
- Stale pre-fetch detection

---

## Phase 8 — Rapport Final

### Format de sortie

```markdown
# Audit SmartSplit — {date}

## Résumé Exécutif
- Score global : X/100
- Findings : N critical, N high, N medium, N low
- Top 3 actions prioritaires

## Métriques
| Métrique | Valeur |
|---|---|
| Fichiers Python | N |
| Lignes de code | N |
| Plus gros module | fichier.py (N lignes) |
| Fonctions sans type hints | N/total |
| Complexité max | N (fichier.py:fonction) |
| Tests | N tests, N modules non couverts |

## Findings par Catégorie

### Clean Code
| # | Sévérité | Fichier:Ligne | Finding | Principe Violé |
|---|---|---|---|---|

### Python Best Practices
| # | Sévérité | Fichier:Ligne | Finding |
|---|---|---|---|

### Tech Debt
| # | Sévérité | Fichier:Ligne | Finding | Priority Score |
|---|---|---|---|---|

### Sécurité
| # | Sévérité | Fichier:Ligne | Finding | OWASP Cat |
|---|---|---|---|---|

### Architecture
| # | Sévérité | Fichier:Ligne | Finding |
|---|---|---|---|

### Performance
| # | Sévérité | Fichier:Ligne | Finding |
|---|---|---|---|

### Tests
| # | Sévérité | Fichier:Ligne | Finding |
|---|---|---|---|

### Praise
| # | Fichier | Ce qui est bien fait |
|---|---|---|

## Plan d'Action Priorisé
1. [CRITICAL] ...
2. [HIGH] ...
3. ...
```

### Scoring (sur 100)

Déductions :
- CRITICAL : -10 points chacun
- HIGH : -5 points chacun
- MEDIUM : -2 points chacun
- LOW : -0.5 point chacun
- Bonus PRAISE : +1 point chacun (max +10)

Score = max(0, 100 - déductions + bonus)

---

## Règles d'exécution

1. **Lire TOUT le code** — chaque fichier .py, pas de raccourci. Utiliser Read sur chaque module.
2. **Être factuel** — chaque finding cite fichier:ligne et le code incriminé
3. **Ne pas inventer** — si c'est correct, le dire. PRAISE le bon code.
4. **Respecter le contexte** — les conventions de CLAUDE.md sont la référence
5. **Pas de fix** — lister les problèmes, le dev décide quoi corriger
6. **Exécuter les outils** — `ruff check smartsplit/ tests/` et `python -m pytest tests/ -v` via le skill /test si disponible
7. **Prioriser** — les findings CRITICAL et HIGH d'abord, toujours
