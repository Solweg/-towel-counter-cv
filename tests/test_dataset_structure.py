"""
Tests de structure du dataset : vérifie que chaque split (train/val/test)
a bien des images ET des labels correspondants, et que la config data.yaml
est valide.
"""

import os
import yaml
import pytest

DATA_YAML_PATH = "data/data.yaml"
SPLITS = ["train", "val", "test"]


def test_data_yaml_exists():
    assert os.path.exists(DATA_YAML_PATH), "data.yaml introuvable"


def test_data_yaml_valid_content():
    with open(DATA_YAML_PATH) as f:
        config = yaml.safe_load(f)

    assert "names" in config
    assert "nc" in config
    assert config["nc"] == len(config["names"]), "nc ne correspond pas au nombre de classes déclarées"
    assert "towel" in config["names"]


@pytest.mark.parametrize("split", SPLITS)
def test_split_has_images_and_labels(split):
    images_dir = f"data/images/{split}"
    labels_dir = f"data/labels/{split}"

    assert os.path.isdir(images_dir), f"Dossier images manquant : {images_dir}"
    assert os.path.isdir(labels_dir), f"Dossier labels manquant : {labels_dir}"

    images = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    labels = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith(".txt")}

    assert len(images) > 0, f"Aucune image trouvée dans {images_dir}"

    missing_labels = images - labels
    assert not missing_labels, f"Images sans label dans {split}: {missing_labels}"

    orphan_labels = labels - images
    assert not orphan_labels, f"Labels sans image correspondante dans {split}: {orphan_labels}"


@pytest.mark.parametrize("split", SPLITS)
def test_no_duplicate_images_across_splits(split):
    """Vérifie qu'une même image n'apparaît pas dans plusieurs splits à la fois."""
    other_splits = [s for s in SPLITS if s != split]
    current_images = set(os.listdir(f"data/images/{split}"))

    for other in other_splits:
        other_images = set(os.listdir(f"data/images/{other}"))
        overlap = current_images & other_images
        assert not overlap, f"Images dupliquées entre {split} et {other}: {overlap}"
