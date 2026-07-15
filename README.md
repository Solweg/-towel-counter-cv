# Towel Counter CV

Projet personnel de vision par ordinateur : détection et comptage de serviettes de bain pliées, par transfer learning sur YOLOv8.

## Décisions techniques et difficultés rencontrées

### Tri du dataset par état de la serviette

Le dataset source (Roboflow "towel detection") contient des serviettes dans des
états variés (pliées, roulées, chiffonnées, pendues), alors que le projet cible
spécifiquement les serviettes pliées et roulées. Une première tentative de tri
manuel via un script Python (OpenCV `cv2.imshow`) a échoué : l'environnement
Linux local n'avait pas le support d'affichage GTK nécessaire.

Plutôt que de corriger l'environnement pour un script à usage unique, le choix a
été fait d'utiliser les fonctionnalités de tagging natives de Roboflow
(plateforme déjà utilisée pour héberger le dataset) pour filtrer les images par
état, avant export d'une version réduite du dataset. Ce choix illustre une
priorité donnée à l'outil existant adapté à la tâche plutôt qu'à la
réimplémentation d'une fonctionnalité déjà disponible.

## Objectif

Démontrer une chaîne complète de machine learning appliqué :
collecte de données → annotation → entraînement par transfer learning →
évaluation → inférence → tests automatisés → CI/CD.

## Stack technique

- Python 3.x
- Ultralytics YOLOv8 (transfer learning, détection d'objets)
- OpenCV
- Pytest
- GitHub Actions (CI/CD)
- Entraînement réalisé en CPU (contrainte matérielle : driver GPU non compatible)

## Structure du projet

(à compléter)

## Installation

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

## Statut du projet

En cours de développement.
