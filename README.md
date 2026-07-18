# Towel Counter CV

Projet personnel de vision par ordinateur : détection et comptage de serviettes de bain pliées, par transfer learning sur YOLOv8.

## Objectif

Démontrer une chaîne complète de machine learning appliqué :

> collecte de données → annotation → entraînement par transfer learning → évaluation → inférence → tests automatisés → CI/CD

## Résultats

Modèle entraîné sur 209 images (146 train / 41 val / 22 test), 50 époques, base YOLOv8n.

| Métrique | Validation | Test (22 images) |
|----------|-----------|-----------------|
| Precision | 0.972 | 0.997 |
| Recall | 0.976 | 1.000 |
| mAP50 | 0.974 | **0.995** |
| mAP50-95 | 0.794 | 0.876 |

> Limite identifiée : le modèle détecte une pile de serviettes empilées comme une seule instance. Voir [Décisions techniques](#décisions-techniques-et-difficultés-rencontrées) pour l'analyse.

## Stack technique

| Composant | Rôle |
|-----------|------|
| Python 3.x | Langage principal |
| Ultralytics YOLOv8 | Transfer learning, détection d'objets |
| OpenCV | Traitement d'images |
| Pytest | Tests automatisés |
| GitHub Actions | CI/CD |

> **Note matérielle** : entraînement réalisé en CPU (driver GPU non compatible).

## CI/CD

Le pipeline CI (`.github/workflows/tests.yml`) se déclenche à chaque push :
1. Téléchargement du dataset filtré depuis Roboflow
2. Split reproductible train/val/test
3. Exécution des tests de structure du dataset (`test_dataset_structure.py`)

Les tests d'inférence (`test_predict_and_count.py`) sont exclus de la CI : les poids du modèle entraîné (~6 Mo) ne sont pas versionnés dans le dépôt Git.

**Piste d'évolution** : récupération automatique du modèle via GitHub Releases ou artifacts, ou entraînement en CI.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

### 1. Télécharger le dataset

Requiert une clé Roboflow dans un fichier `.env` (`ROBOFLOW_API_KEY=...`).

```bash
python src/download_dataset.py
```

### 2. Préparer le dataset (split train/val/test)

```bash
python src/prepare_dataset.py
```

### 3. Entraîner le modèle

```bash
python src/train.py
```

Les poids sont sauvegardés dans `runs/detect/runs/train/towel_detector_v1/weights/best.pt`.

### 4. Évaluer sur le set de test

```bash
python src/evaluate.py
```

### 5. Compter les serviettes sur une image

```bash
python src/predict_and_count.py chemin/vers/image.jpg
```

L'image annotée (bounding boxes) est sauvegardée dans `runs/detect/predict/`.

### 6. Lancer les tests

```bash
pytest tests/
```

## Structure du projet

```
towel-counter-cv/
├── data/
│   ├── raw/          # dataset brut téléchargé depuis Roboflow
│   ├── images/       # split train/val/test (généré par prepare_dataset.py)
│   ├── labels/       # annotations YOLO (générées par prepare_dataset.py)
│   └── data.yaml     # config Ultralytics (généré par prepare_dataset.py)
├── src/
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict_and_count.py
├── tests/
│   ├── test_dataset_structure.py
│   └── test_predict_and_count.py
├── runs/             # sorties d'entraînement et d'inférence (non versionné)
└── .github/workflows/tests.yml
```

## Statut

POC fonctionnel.

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
