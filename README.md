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
### Limite identifiée : détection sur piles de serviettes empilées

Le modèle entraîné (mAP50 = 0.995 sur le set de test) donne d'excellents
résultats sur des serviettes isolées, similaires à celles du dataset
d'entraînement. Un test sur une image réelle de pile de 4 serviettes
empilées a révélé une limite claire : le modèle détecte l'ensemble de la
pile comme **une seule instance** (`towel 0.84`), sans distinguer les
4 unités individuelles pourtant visuellement séparées par des plis nets.

**Cause identifiée** : le dataset d'entraînement (209 images, après tri
manuel) ne contient que des images à serviette unique, isolée. Le modèle
n'a donc jamais été exposé à des exemples d'instances empilées/contiguës
pendant l'apprentissage, et n'a pas appris à distinguer les frontières
entre plusieurs objets identiques adjacents.

**Piste d'amélioration identifiée (hors périmètre de ce POC)** : enrichir
le dataset avec des images de piles réelles, annotées avec une bounding
box par serviette individuelle, pour permettre au modèle d'apprendre la
notion de séparation d'instances empilées.

Ce résultat, bien que négatif par rapport à l'objectif final visé, valide
la pertinence de la méthodologie (transfer learning, pipeline complet) et
illustre l'importance de la diversité du dataset d'entraînement par
rapport aux cas d'usage réels visés.