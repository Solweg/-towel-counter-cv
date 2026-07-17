"""
Inférence + comptage de serviettes sur une image donnée.

Le comptage est simplement le nombre de détections retournées par le modèle
au-dessus du seuil de confiance défini (pas d'algorithme de comptage séparé :
YOLO fait la détection, on compte les boîtes obtenues).
"""

import argparse
from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/train/towel_detector_v1/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.25  # seuil de confiance minimum pour valider une détection


def count_towels(image_path, save_output=True):
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=image_path,
        conf=CONFIDENCE_THRESHOLD,
        device="cpu",
        save=save_output,  # sauvegarde l'image annotée avec les boîtes
    )

    result = results[0]
    count = len(result.boxes)

    print(f"\nImage : {image_path}")
    print(f"Nombre de serviettes détectées : {count}")

    for i, box in enumerate(result.boxes, start=1):
        conf = float(box.conf[0])
        print(f"  Détection {i} : confiance = {conf:.2f}")

    if save_output:
        print(f"\nImage annotée sauvegardée dans : {result.save_dir}")

    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compte les serviettes sur une image")
    parser.add_argument("image", help="Chemin vers l'image à analyser")
    args = parser.parse_args()

    count_towels(args.image)
