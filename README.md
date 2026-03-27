# Système d'Aide au Tri Radiologique - Deep Learning

Projet de classification de pathologies thoraciques sur radiographies avec Deep Learning.

## Commandes Rapides

### Installation
```bash
pip install -r requirements.txt
```

### Dataset 1 - ChestMNIST (Classification Supervisée)
```bash
# 1. Préparer les données
python scripts\01_prepare_data.py

# 2. Entraîner les modèles (choisir un modèle)
python scripts\02_train_supervised.py --model cnn_simple --epochs 5
python scripts\02_train_supervised.py --model resnet50 --epochs 5
python scripts\02_train_supervised.py --model efficientnet_b0 --epochs 5
```

### Dataset 2 - NIH (Anomaly + Multimodal)
```bash
# 1. Préparer échantillon NIH
python scripts\05_prepare_nih_sample.py --num_samples 1000

# 2. Entraîner autoencoder (anomaly detection)
python scripts\06_train_autoencoder.py --epochs 10

# 3. Entraîner multimodal (image + texte)
python scripts\07_train_multimodal.py --epochs 5 --fusion late
```

### MLflow (Visualisation des expériences)
```bash
mlflow ui --port 5000
# Ouvrir http://localhost:5000
```

### Application Streamlit
```bash
streamlit run app\streamlit_app.py
# Ouvrir http://localhost:8501
```

---

## Architecture

```
systeme_aide_tri_radiologique_dp_project/
├── scripts/
│   ├── 01_prepare_data.py          # Préparation ChestMNIST
│   ├── 02_train_supervised.py      # Entraînement supervisé (3 modèles)
│   ├── 05_prepare_nih_sample.py    # Préparation NIH
│   ├── 06_train_autoencoder.py     # Anomaly detection
│   └── 07_train_multimodal.py      # Modèle multimodal
├── src/
│   ├── models/                      # Architectures CNN, ResNet, EfficientNet
│   ├── training/                    # Trainer + MLflow utils
│   ├── preprocessing/               # DataLoaders
│   └── utils/                       # Métriques, visualisation
├── app/
│   └── streamlit_app.py             # Interface web
├── data/
│   └── processed/                   # Données preprocessées
├── models/                          # Modèles entraînés
├── mlruns/                          # Expériences MLflow
├── config.yaml                      # Configuration
└── requirements.txt                 # Dépendances
```

---

## Composants

### Dataset 1 - ChestMNIST
- Classification multi-label (14 pathologies)
- 3 modèles : CNN Simple, ResNet50, EfficientNet-B0
- Resolution : 64x64
- Dataset : 78k images train, 11k val, 22k test

### Dataset 2 - NIH Sample
- Echantillon de 1000 images (vs 112k complet)
- Autoencoder pour anomaly detection
- Modèle multimodal (image + texte)
- Fusion early ou late

### MLflow
- Tracking des hyperparamètres
- Métriques : ROC-AUC, F1-score, Precision, Recall
- Sauvegarde automatique des meilleurs modèles

---

## Métriques

Multi-label classification :
- ROC-AUC (micro, macro, weighted)
- F1-Score, Precision, Recall
- Hamming Loss, Exact Match Ratio

Anomaly detection :
- Erreur de reconstruction (MSE)
- Seuil 95e percentile

---

## Configuration

Modifier `config.yaml` pour ajuster :
- Hyperparamètres (learning rate, batch size, epochs)
- Architecture des modèles
- Chemins des données

---

## Ressources

- PyTorch : https://pytorch.org
- MedMNIST : https://medmnist.com
- MLflow : https://mlflow.org

---

Dernière mise à jour : Mars 2026
