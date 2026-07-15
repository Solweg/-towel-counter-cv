"""
Script de téléchargement du dataset "towel detection" (version filtrée)
depuis le workspace Roboflow personnel.

Source originale : https://universe.roboflow.com/zezamii/towel-detection-1xa8w
Fork personnel après tri manuel (conservation des serviettes pliées/roulées
uniquement, 209 images sur 307) : workspace hlnes-workspace-zlfgw
Licence : CC BY 4.0
"""

import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise ValueError(
        "ROBOFLOW_API_KEY introuvable. "
        "Vérifie que le fichier .env existe et contient bien la clé."
    )

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("hlnes-workspace-zlfgw").project("towel-detection-1xa8w-ofnju")
version = project.version(1)
dataset = version.download("yolov8", location="data/raw/towel-detection-filtered")

print(f"Dataset filtré téléchargé dans : {dataset.location}")