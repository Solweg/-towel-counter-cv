"""
Entraînement YOLOv8 par transfer learning sur le dataset de serviettes pliées/roulées.

Modèle de base : YOLOv8n (nano), pré-entraîné sur COCO.
Choix motivé par : rapidité d'entraînement en CPU, adapté à un petit dataset (146 images train).
"""

from ultralytics import YOLO

DATA_YAML = "data/data.yaml"
BASE_MODEL = "yolov8n.pt"  # nano, pré-entraîné COCO
EPOCHS = 50
IMG_SIZE = 640
PROJECT_NAME = "runs/train"
RUN_NAME = "towel_detector_v1"


def main():
    model = YOLO(BASE_MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        device="cpu",
        patience=15,  # arrêt anticipé si pas d'amélioration après 15 époques
        verbose=True,
    )

    print("Entraînement terminé.")
    print(f"Résultats et poids sauvegardés dans : {PROJECT_NAME}/{RUN_NAME}")


if __name__ == "__main__":
    main()
