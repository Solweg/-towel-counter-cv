"""
Prépare le dataset final pour l'entraînement YOLOv8 :
- Lit les 209 images/labels du dataset filtré (Roboflow, post-tri manuel)
- Effectue un split reproductible 70% train / 20% val / 10% test
- Copie vers l'arborescence data/images/{train,val,test} et data/labels/{...}
- Génère un data.yaml propre à la racine du projet

Seed fixée à 42 pour garantir la reproductibilité du split.
"""

import os
import random
import shutil
import yaml

SEED = 42
SOURCE_IMAGES = "data/raw/towel-detection-filtered/train/images"
SOURCE_LABELS = "data/raw/towel-detection-filtered/train/labels"

DEST_IMAGES = "data/images"
DEST_LABELS = "data/labels"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
# le reste (0.10) va au test

CLASS_NAME = "towel"


def get_image_label_pairs():
    """Associe chaque image à son fichier label correspondant."""
    images = sorted(f for f in os.listdir(SOURCE_IMAGES) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    pairs = []
    missing_labels = []

    for img in images:
        label_name = os.path.splitext(img)[0] + ".txt"
        label_path = os.path.join(SOURCE_LABELS, label_name)
        if os.path.exists(label_path):
            pairs.append((img, label_name))
        else:
            missing_labels.append(img)

    if missing_labels:
        print(f"ATTENTION : {len(missing_labels)} image(s) sans label associé, ignorée(s) :")
        for m in missing_labels[:10]:
            print(f"  - {m}")

    return pairs


def split_dataset(pairs):
    """Split reproductible train/val/test."""
    random.seed(SEED)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train:n_train + n_val]
    test_set = shuffled[n_train + n_val:]

    return {"train": train_set, "val": val_set, "test": test_set}


def copy_split(split_name, pairs):
    """Copie les fichiers d'un split vers l'arborescence finale."""
    img_dest_dir = os.path.join(DEST_IMAGES, split_name)
    label_dest_dir = os.path.join(DEST_LABELS, split_name)

    for img_name, label_name in pairs:
        shutil.copy2(
            os.path.join(SOURCE_IMAGES, img_name),
            os.path.join(img_dest_dir, img_name),
        )
        shutil.copy2(
            os.path.join(SOURCE_LABELS, label_name),
            os.path.join(label_dest_dir, label_name),
        )


def write_data_yaml():
    """Génère le data.yaml final utilisé par Ultralytics."""
    data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": [CLASS_NAME],
    }
    with open("data/data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def main():
    pairs = get_image_label_pairs()
    print(f"Total paires image/label valides : {len(pairs)}")

    splits = split_dataset(pairs)

    for split_name, split_pairs in splits.items():
        print(f"  {split_name}: {len(split_pairs)} images")
        copy_split(split_name, split_pairs)

    write_data_yaml()
    print("\nSplit terminé. Fichier de config généré : data/data.yaml")


if __name__ == "__main__":
    main()
