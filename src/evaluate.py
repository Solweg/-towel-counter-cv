"""
Évaluation du modèle entraîné sur le set de test (22 images jamais vues
ni pendant l'entraînement, ni pendant la validation).

Objectif : obtenir une estimation non biaisée des performances du modèle,
distincte du mAP de validation obtenu pendant l'entraînement.
"""

from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/train/towel_detector_v1/weights/best.pt"
DATA_YAML = "data/data.yaml"


def main():
    model = YOLO(MODEL_PATH)

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        device="cpu",
    )

    print("\n=== Résultats sur le set de test ===")
    print(f"mAP50    : {metrics.box.map50:.3f}")
    print(f"mAP50-95 : {metrics.box.map:.3f}")
    print(f"Precision moyenne : {metrics.box.mp:.3f}")
    print(f"Recall moyen      : {metrics.box.mr:.3f}")


if __name__ == "__main__":
    main()
