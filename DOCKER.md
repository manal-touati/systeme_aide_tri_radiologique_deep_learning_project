# 🐳 Guide Docker Complet

## Table des matières

1. [Installation](#installation)
2. [Démarrage rapide](#démarrage-rapide)
3. [Services disponibles](#services-disponibles)
4. [Configuration avancée](#configuration-avancée)
5. [Dépannage](#dépannage)
6. [Personnalisation](#personnalisation)

---

## Installation

### Prérequis

#### Windows
1. Télécharger [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
2. Installer avec WSL 2 backend
3. Redémarrer l'ordinateur

#### macOS
1. Télécharger [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
2. Intel ou Apple Silicon (M1/M2/M3) supporté
3. Installer et lancer

#### Linux
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo usermod -aG docker $USER
newgrp docker

# Vérifier
docker --version
docker-compose --version
```

### Vérifier l'installation

```bash
docker ps
docker-compose --version
```

---

## Démarrage rapide

### Méthode 1 : Scripts interactifs (recommandé)

**Windows** (PowerShell ou CMD) :
```powershell
.\docker_start.bat
```

**Linux/macOS** (Terminal) :
```bash
bash docker_start.sh
```

Puis sélectionner l'option désirée dans le menu.

### Méthode 2 : Commandes Docker Compose

**Démarrer l'app Streamlit seule** :
```bash
docker-compose up streamlit
```
✅ Accéder à http://localhost:8501

**Démarrer Streamlit + MLflow** :
```bash
docker-compose up streamlit mlflow
```
- App Streamlit : http://localhost:8501
- MLflow UI : http://localhost:5000

**Lancer une expérience d'entraînement** :
```bash
docker-compose --profile training up trainer
```
Exécute `quickstart.py` dans un conteneur.

**Afficher les logs** :
```bash
docker-compose logs -f streamlit
docker-compose logs -f mlflow
```

**Arrêter tout** :
```bash
docker-compose down
```

**Arrêter et supprimer les données** :
```bash
docker-compose down -v
```

---

## Services disponibles

### 1. Streamlit (Application Web)

```yaml
Container: radiological_app
Port: 8501
```

**Fonctionnalités** :
- Upload d'images radiologiques
- Prédiction en temps réel (multi-modèles)
- Visualisation des scores
- Anomaly detection
- Fusion multimodale (si données disponibles)

**Accès** : http://localhost:8501

### 2. MLflow (Experiment Tracking)

```yaml
Container: mlflow_server
Port: 5000
Backend: fichier local (mlruns/)
```

**Fonctionnalités** :
- Historique de tous les entraînements
- Comparaison des modèles
- Artéfacts sauvegardés
- Métriques et hyperparamètres

**Accès** : http://localhost:5000

**Utiliser MLflow depuis Streamlit** :
```python
from src.training.mlflow_utils import MLFlowTracker

tracker = MLFlowTracker(
    tracking_uri="http://mlflow:5000",  # Référence intra-Docker
    experiment_name="streaming_exp"
)
```

### 3. Trainer (Entraînement - Optionnel)

```yaml
Container: radiological_trainer
Profile: training
```

**Utilisation** :
```bash
# Démarrer l'entraînement
docker-compose --profile training up trainer

# Voir les logs
docker-compose logs trainer

# Entraîner un modèle spécifique
docker-compose run --rm trainer python train.py --model transfer_learning
```

---

## Configuration avancée

### Variables d'environnement

Créer un fichier `.env` à la racine du projet :

```env
# Ports
STREAMLIT_PORT=8501
MLFLOW_PORT=5000

# GPU (si disponible)
GPU_ENABLED=false

# Logging
PYTHONUNBUFFERED=1
LOG_LEVEL=INFO

# MLflow
MLFLOW_BACKEND_STORE=/app/mlruns

# Modèle par défaut
DEFAULT_MODEL=transfer_learning
DEFAULT_BACKBONE=resnet50
```

Puis charger dans docker-compose.yml :
```yaml
env_file:
  - .env
```

### GPU Support (NVIDIA)

#### Installation (si vous avez une GPU NVIDIA)

1. Installer [NVIDIA Container Runtime](https://github.com/NVIDIA/nvidia-docker)

2. Modifier `docker-compose.yml` :
```yaml
services:
  streamlit:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

3. Démarrer :
```bash
docker-compose up streamlit
```

Vérifier GPU dans le conteneur :
```bash
docker exec radiological_app python -c "import torch; print(torch.cuda.is_available())"
```

### Volumes personnalisés

Ajouter des volumes dans docker-compose.yml :

```yaml
services:
  streamlit:
    volumes:
      - ./data:/app/data              # Données
      - ./models:/app/models          # Modèles
      - ./notebooks:/app/notebooks    # Notebooks
      - ./custom_data:/app/external   # Source externe
```

### Build personnalisé

Créer un `Dockerfile.custom` :

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Dépendances système supplémentaires
RUN apt-get update && apt-get install -y \
    graphviz \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Packages supplémentaires
RUN pip install --no-cache-dir \
    torch-geometric \
    wandb

COPY . .

EXPOSE 8501 5000

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Utiliser dans docker-compose.yml :
```yaml
services:
  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.custom
```

---

## Dépannage

### Problème : "Cannot connect to Docker daemon"

**Linux** :
```bash
sudo systemctl start docker
sudo usermod -aG docker $USER
```

**Windows/macOS** : Vérifier que Docker Desktop est lancé

### Problème : Port déjà utilisé

```bash
# Voir quels processus utilisent les ports
# Windows
netstat -ano | findstr :8501

# Linux/macOS
lsof -i :8501

# Solution : Modifier docker-compose.yml
# ports: ["8502:8501"]  # Utiliser 8502 à la place
```

### Problème : "Out of memory"

```bash
# Augmenter la RAM de Docker
# Windows/Mac: Settings > Resources > Memory → 8GB+ recommandé

# Linux: Limiter dans docker-compose.yml
services:
  streamlit:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

### Problème : Logs incompréhensibles

```bash
# Voir les logs détaillés
docker-compose logs streamlit --details

# Avec timestamps
docker-compose logs -f --timestamps streamlit

# Seulement les erreurs
docker-compose logs streamlit | grep ERROR
```

### Problème : Les changements de code ne se reflètent pas

**Solution** : Rebuildez l'image

```bash
# Sans cache pour forcer rebuild
docker-compose build --no-cache

# Puis relancer
docker-compose up streamlit
```

Ou utiliser **volumes pour développement** :

```yaml
services:
  streamlit:
    volumes:
      - .:/app  # Monte le répertoire local en live
```

Avec cette approche, les changements .py sont immédiatement visibles.

### Problème : "Module not found"

```bash
# Entrer dans le conteneur et tester
docker exec -it radiological_app python -c "import torch; print(torch.__version__)"

# Réinstaller les requirements
docker-compose build --no-cache
docker-compose up --force-recreate
```

### Problème : Données non persistantes

Vérifier que les volumes sont montés correctement :

```bash
# Lister les volumes
docker volume ls

# Inspecter un volume
docker volume inspect <volume_name>

# Vérifier le mapping dans docker-compose.yml
docker-compose config | grep -A 5 volumes
```

---

## Personnalisation

### Ajouter un nouveau service

Exemple : Ajouter Jupyter Notebook

```yaml
jupyter:
  build: .
  container_name: radiological_jupyter
  ports:
    - "8888:8888"
  volumes:
    - ./notebooks:/app/notebooks
    - ./data:/app/data
    - ./models:/app/models
  environment:
    - PYTHONUNBUFFERED=1
  command: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Démarrer :
```bash
docker-compose up jupyter
```

Token Jupyter s'affiche dans les logs :
```bash
docker-compose logs jupyter | grep token
```

### Multi-stage build (optionnel)

Pour réduire la taille de l'image :

```dockerfile
# Stage 1 : Build
FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2 : Runtime
FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8501 5000
CMD ["streamlit", "run", "app/streamlit_app.py"]
```

### Docker Network (si vous avez plusieurs projets interconnectés)

```yaml
networks:
  radiological_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

services:
  streamlit:
    networks:
      - radiological_network
    environment:
      - MLFLOW_URI=http://mlflow:5000
```

---

## Tips & Best Practices

### 1. Utiliser .dockerignore

L'avoir (`✅ inclus dans le projet`) évite de copier inutilement :
- `__pycache__`
- `.git`
- `*.pyc`
- Datasets volumineux
- Résultats/modèles

### 2. Images parallèles pour dev et prod

```yaml
# dev
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# prod
docker-compose -f docker-compose.prod.yml up
```

### 3. Health checks

```yaml
services:
  streamlit:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### 4. Sauvegarder/charger les données

```bash
# Sauvegarder les volumes
docker run --rm -v mlruns:/data -v ${PWD}:/backup ubuntu tar czf /backup/mlruns.tar.gz -C /data .

# Restaurer
docker run --rm -v mlruns:/data -v ${PWD}:/backup ubuntu tar xzf /backup/mlruns.tar.gz -C /data
```

### 5. Utiliser Compose Overrides pour développement local

Créer `docker-compose.override.yml` (auto-chargé par défaut) :

```yaml
# Montage live du code, ports alternatifs, etc.
services:
  streamlit:
    volumes:
      - .:/app
    ports:
      - "8502:8501"
    environment:
      - DEBUG=true
```

---

## Ressources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Streamlit + Docker](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
- [MLflow Deployment](https://mlflow.org/docs/latest/deployments/index.html)

---

**Dernière mise à jour** : Mars 2026
