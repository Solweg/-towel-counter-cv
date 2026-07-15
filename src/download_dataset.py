"""
Script de téléchargement du dataset "towel detection" depuis Roboflow Universe.
Source : https://universe.roboflow.com/zezamii/towel-detection-1xa8w
Licence : CC BY 4.0
"""

import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()  # charge les variables depuis le fichier .env

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise ValueError(
        "ROBOFLOW_API_KEY introuvable. "
        "Vérifie que le fichier .env existe et contient bien la clé."
    )

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("zezamii").project("towel-detection-1xa8w")
version = project.version(1)
dataset = version.download("yolov8", location="data/raw/towel-detection")

print(f"Dataset téléchargé dans : {dataset.location}")

