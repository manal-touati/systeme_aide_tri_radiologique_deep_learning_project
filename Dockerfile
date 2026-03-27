# Dockerfile pour Système d'Aide Radiologique

FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt


# Copier le code du projet
COPY . .

# Exposer les ports
# 8501 : Streamlit
# 5000 : MLflow
EXPOSE 8501 5000

# Commande par défaut : lancer Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
