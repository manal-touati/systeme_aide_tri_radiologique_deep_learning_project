Système d'Aide au Tri Radiologique - Deep Learning
Projet de classification / détection d'anomalies sur radiographies thoraciques (ChestMNIST, NIH subset) avec démonstrateur Streamlit et suivi MLflow.

## Installation
```bash
pip install -r requirements.txt
```

## Commandes rapides
### Dataset 1 — ChestMNIST (classification)
```bash
# Préparer les données
python scripts\01_prepare_data.py

# Entraîner (choisir un modèle)
python scripts\02_train_supervised.py --model cnn_simple --epochs 5
python scripts\02_train_supervised.py --model resnet50 --epochs 5
python scripts\02_train_supervised.py --model efficientnet_b0 --epochs 5
```

### Dataset 2 — NIH (anomaly + multimodal)
```bash
# Préparer un échantillon réel (ex. 4 999 images)
python scripts\05_prepare_nih_sample.py --num_samples 4999 --target_size 64 --csv_path .\data\raw\Data_Entry_2017.csv --images_dir .\data\raw --data_dir .\data

# Autoencoder (anomaly detection)
python scripts\06_train_autoencoder.py --epochs 10

# Multimodal (image + texte)
python scripts\07_train_multimodal.py --epochs 5 --fusion late
```

### MLflow
```bash
mlflow ui --port 5000
# puis ouvrir http://localhost:5000
```

### Démonstrateur Streamlit
```bash
streamlit run app\streamlit_app.py
# puis ouvrir http://localhost:8501
```

## Architecture
systeme_aide_tri_radiologique_dp_project/
├── scripts/                # 01..07 pipeline entraînement/évaluation
├── src/                    # models, training, preprocessing, utils
├── app/                    # streamlit_app.py (démo)
├── data/processed/         # npz preprocessés
├── models/                 # checkpoints entraînés
├── mlruns/                 # expériences MLflow
├── config.yaml             # configuration
└── requirements.txt        # dépendances

## Composants
- ChestMNIST : 14 pathologies, 64x64, modèles CNN simple / ResNet50 / EfficientNet-B0.
- NIH subset : autoencoder pour anomalie, modèle multimodal (image + texte, fusion tardive).
- MLflow : tracking hyperparamètres, métriques, artefacts.

## Métriques
- Classification : ROC-AUC (micro/macro/weighted), F1, Precision, Recall, Hamming, Exact Match.
- Anomalie : erreur de reconstruction (MSE), seuil à calibrer (ex. p95).

## Configuration
Adapter `config.yaml` pour les hyperparamètres, chemins de données, choix de modèle.

## Ressources
- PyTorch : https://pytorch.org
- MedMNIST : https://medmnist.com
- MLflow : https://mlflow.org

Dernière mise à jour : Mars 2026