"""
Tests du pipeline d'inférence + comptage : vérifie que le script s'exécute
sans erreur et retourne un résultat cohérent, sur une image connue du set
de test (donc jamais vue à l'entraînement).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict_and_count import count_towels

MODEL_PATH = "runs/detect/runs/train/towel_detector_v1/weights/best.pt"
TEST_IMAGES_DIR = "data/images/test"


@pytest.fixture
def sample_test_image():
    """Récupère la première image disponible du set de test."""
    images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    assert images, "Aucune image trouvée dans le set de test"
    return os.path.join(TEST_IMAGES_DIR, images[0])


def test_model_weights_exist():
    assert os.path.exists(MODEL_PATH), (
        f"Poids du modèle introuvables : {MODEL_PATH}. "
        "Lancez d'abord src/train.py."
    )


def test_count_towels_returns_positive_integer(sample_test_image):
    count = count_towels(sample_test_image, save_output=False)

    assert isinstance(count, int), "Le comptage doit retourner un entier"
    assert count >= 0, "Le comptage ne peut pas être négatif"


def test_count_towels_detects_at_least_one_on_known_positive_image(sample_test_image):
    """
    Sur une image du set de test (qui contient toujours au moins une
    serviette par construction du dataset), on s'attend à au moins 1 détection.
    """
    count = count_towels(sample_test_image, save_output=False)
    assert count >= 1, "Aucune serviette détectée sur une image censée en contenir"
