# Towel Counter CV

Projet personnel de vision par ordinateur : détection et comptage de serviettes de bain pliées, par transfer learning sur YOLOv8.

## Objectif

Démontrer une chaîne complète de machine learning appliqué :

> collecte de données → annotation → entraînement par transfer learning → évaluation → inférence → tests automatisés → CI/CD

## Stack technique

| Composant | Rôle |
|-----------|------|
| Python 3.x | Langage principal |
| Ultralytics YOLOv8 | Transfer learning, détection d'objets |
| OpenCV | Traitement d'images |
| Pytest | Tests automatisés |
| GitHub Actions | CI/CD |

> **Note matérielle** : entraînement réalisé en CPU (driver GPU non compatible).

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Structure du projet

*(à compléter)*

## Statut

En cours de développement.

---

## Décisions techniques et difficultés rencontrées

### Tri du dataset par état de la serviette

Le dataset source (Roboflow "towel detection") contient des serviettes dans des états variés (pliées, roulées, chiffonnées, pendues), alors que le projet cible spécifiquement les serviettes pliées et roulées.

Une première tentative de tri manuel via un script Python (`cv2.imshow`) a échoué : l'environnement Linux local n'avait pas le support d'affichage GTK nécessaire. Plutôt que de corriger l'environnement pour un script à usage unique, le choix a été fait d'utiliser les fonctionnalités de tagging natives de Roboflow pour filtrer les images par état avant export. Ce choix illustre une priorité donnée à l'outil existant adapté à la tâche plutôt qu'à la réimplémentation d'une fonctionnalité déjà disponible.

### Limite identifiée : détection sur piles de serviettes empilées

Le modèle entraîné (**mAP50 = 0.995** sur le set de test) donne d'excellents résultats sur des serviettes isolées, similaires à celles du dataset d'entraînement. Un test sur une image réelle de pile de 4 serviettes a révélé une limite claire : le modèle détecte l'ensemble de la pile comme **une seule instance** (`towel 0.84`), sans distinguer les 4 unités individuelles pourtant séparées par des plis nets.

**Cause** : le dataset d'entraînement (209 images après tri) ne contient que des images à serviette unique et isolée. Le modèle n'a jamais été exposé à des instances empilées/contiguës, et n'a pas appris à distinguer les frontières entre plusieurs objets identiques adjacents.

**Piste d'amélioration** *(hors périmètre de ce POC)* : enrichir le dataset avec des images de piles réelles, annotées avec une bounding box par serviette individuelle.

Ce résultat, bien que négatif par rapport à l'objectif final, valide la pertinence de la méthodologie et illustre l'importance de la diversité du dataset par rapport aux cas d'usage réels visés.

### CI/CD (GitHub Actions)

Le pipeline CI (`.github/workflows/tests.yml`) régénère le dataset (téléchargement + split) à chaque push, puis exécute les tests de structure du dataset.

**Choix assumé** : les tests du pipeline d'inférence (`test_predict_and_count.py`) ne sont pas exécutés en CI, car ils dépendent des poids du modèle entraîné (~6 Mo, volontairement exclus du dépôt Git).

**Piste d'évolution** : entraînement automatisé en CI, ou récupération du modèle via GitHub Releases/artifacts.
