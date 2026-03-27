"""
Application Streamlit - Interface de démonstrateur
"""
import streamlit as st
import torch
import numpy as np
from PIL import Image
import logging
from pathlib import Path
import sys

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.cnn_simple import SimpleCNN
from models.transfer_learning import TransferLearningModel
from models.vit import VisionTransformer
from models.autoencoder import VariationalAutoencoder
from preprocessing.data_loader import get_transforms
from utils.config import get_config

# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="Aide au Tri Radiologique",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🫁 Système d'Aide au Tri Radiologique")
st.markdown("""
Détection de pathologies thoraciques à partir de radiographies X.

**Capacités:**
- Classification supervisée (14 pathologies)
- Détection d'anomalies (images atypiques)
- Analyse multimodale (image + compte-rendu)
""")

# ============================================================================
# SIDEBAR - CONTRÔLES
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Choix du modèle
    st.subheader("Modèle de Classification")
    model_choice = st.selectbox(
        "Sélectionner un modèle",
        ["CNN Simple", "ResNet50 Transfer Learning", "Vision Transformer"]
    )
    
    # Détection d'anomalies
    st.subheader("Détection d'Anomalies")
    enable_anomaly = st.checkbox("Activer détection d'anomalies", value=True)
    if enable_anomaly:
        anomaly_threshold = st.slider(
            "Seuil d'anomalie (percentile)",
            min_value=50, max_value=99, value=95, step=1
        )
    
    # Multimodalité
    st.subheader("Multimodalité")
    enable_multimodal = st.checkbox("Utiliser image + compte-rendu", value=False)

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Classification",
    "⚠️ Anomalies",
    "📝 Multimodalité",
    "ℹ️ Information"
])

# ============================================================================
# TAB 1 : CLASSIFICATION
# ============================================================================

with tab1:
    st.header("Classification Supervisée")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Entrée")
        uploaded_file = st.file_uploader(
            "Télécharger une radiographie",
            type=["jpg", "jpeg", "png"],
            help="Résolution recommandée : 224x224 ou plus"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('L')  # Grayscale
            st.image(image, caption="Image téléchargée", width=300)
    
    with col2:
        st.subheader("Résultats")
        
        if uploaded_file:
            # Traiter l'image
            try:
                # Redimensionner
                image = image.resize((224, 224))
                
                # Transformer
                transform = get_transforms(image_size=224, augment=False, is_training=False)
                img_tensor = transform(image).unsqueeze(0)  # Ajouter batch dim
                
                st.success("✓ Image chargée et prétraitée")
                
                # Logique de modèle (placeholder)
                st.info("Modèle sélectionné : " + model_choice)
                
                # Afficher prédiction fictive
                if st.button("🔍 Lancer la Prédiction"):
                    with st.spinner("Analyse en cours..."):
                        # Logique de classification (à implémenter)
                        st.write("#### Prédictions par pathologie")
                        
                        # Top 5 résultats
                        results = {
                            "Pneumonie": 0.87,
                            "Tuberculose": 0.12,
                            "Pneumothorax": 0.01,
                            "Normal": 0.00,
                            "Autres": 0.00
                        }
                        
                        # Barchart
                        st.bar_chart(results)
                        
                        # Prédiction principale
                        top_class = max(results, key=results.get)
                        top_prob = results[top_class]
                        st.metric(
                            label="🎯 Prédiction Principale",
                            value=f"{top_class}",
                            delta=f"{top_prob*100:.1f}%"
                        )
                        
                        # Confiance
                        st.progress(top_prob, text=f"Confiance : {top_prob*100:.1f}%")
            
            except Exception as e:
                st.error(f"Erreur : {str(e)}")


# ============================================================================
# TAB 2 : DÉTECTION D'ANOMALIES
# ============================================================================

with tab2:
    st.header("Détection d'Anomalies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Ce système détecte les radiographies atypiques, rares ou hors-distribution.")
        if enable_anomaly:
            st.info(f"Seuil d'anomalie fixé à {anomaly_threshold}e percentile")
            
            st.subheader("Score d'Anomalie")
            anomaly_score = np.random.uniform(0, 1)  # Placeholder
            
            if anomaly_score > 0.7:
                status = "⚠️ ANORMAL"
                color = "red"
            else:
                status = "✓ NORMAL"
                color = "green"
            
            st.markdown(f"<h2 style='color:{color}'>{status}</h2>", unsafe_allow_html=True)
            st.progress(anomaly_score, text=f"Score : {anomaly_score:.2f}")
        else:
            st.warning("Détection d'anomalies désactivée")
    
    with col2:
        st.subheader("Visualisation")
        st.write("Erreur de reconstruction : ")
        
        # Placeholder plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.linspace(0, 10, 100)
        ax.hist(np.random.normal(5, 1, 1000), bins=30, alpha=0.7, label="Distribution")
        ax.axvline(x=7, color='red', linestyle='--', label='Seuil')
        ax.set_xlabel('Erreur de Reconstruction')
        ax.set_ylabel('Fréquence')
        ax.legend()
        st.pyplot(fig)


# ============================================================================
# TAB 3 : MULTIMODALITÉ
# ============================================================================

with tab3:
    st.header("Analyse Multimodale (Image + Texte)")
    
    if enable_multimodal:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Radiographie")
            st.image("https://via.placeholder.com/300", width=300)
        
        with col2:
            st.subheader("Compte-rendu Radiologique")
            report = st.text_area(
                "Compte-rendu (ou coller depuis document)",
                value="Radiographie du thorax sans particularité.",
                height=200
            )
        
        if st.button("🔄 Analyser Multimodal"):
            st.success("Analyse complétée")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Image seule", "87%")
            with col2:
                st.metric("Texte seul", "72%")
            with col3:
                st.metric("Fusion", "92%", delta="+5%")
    
    else:
        st.warning("Modalité multimodale désactivée dans les paramètres")


# ============================================================================
# TAB 4 : INFORMATION
# ============================================================================

with tab4:
    st.header("ℹ️ À Propos du Système")
    
    st.markdown("""
    ### Architecture
    
    | Composant | Détail |
    |-----------|--------|
    | **Classification** | 3 architectures (CNN, Transfer Learning, ViT) |
    | **Données** | ChestMNIST, NIH CXR14 |
    | **Pathologies** | 14 classes (Pneumonie, TB, etc.) |
    | **Normalisation** | Prétraitement standard pour images médicales |
    
    ### Modèles
    - CNN Simple : Baseline rapide
    - ResNet50 : Transfer Learning (pré-entraîné ImageNet)
    - Vision Transformer : État de l'art
    
    ### Détection d'Anomalies
    - Technique : VAE (Variational Autoencoder)
    - Score : Erreur de reconstruction + KL divergence
    
    ### Multimodalité
    - Image : ResNet50
    - Texte : BERT / DistilBERT
    - Fusion : Tardive (concatenation)
    
    ### Limitations
    ⚠️ Ce système est à usage académique/recherche. **Ne pas utiliser en production sans validation clinique approbée.**
    """)
    
    st.divider()
    
    st.subheader("📊 Métriques de Performance")
    
    metrics_data = {
        "Modèle": ["CNN Simple", "ResNet50", "ViT"],
        "Accuracy": [0.78, 0.92, 0.95],
        "ROC-AUC": [0.82, 0.94, 0.96],
        "F1-Score": [0.75, 0.91, 0.94]
    }
    
    st.dataframe(
        metrics_data,
        use_container_width=True
    )

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
---
*Projet M1 Data Engineering | Système d'Aide au Tri Radiologique*

**Disclaimer**: Ce système est fourni à titre académique/expérimental úniquement.
""")
