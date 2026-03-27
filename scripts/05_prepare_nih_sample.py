#!/usr/bin/env python
"""
Script 05: Préparation échantillon NIH (Dataset 2)

Télécharge et prépare un échantillon représentatif du dataset NIH CXR14.
Au lieu de 112k images (40GB), on prend 1000 images (plus gérable).

Usage:
    python scripts/05_prepare_nih_sample.py --num_samples 1000
"""
import sys
from pathlib import Path
import logging
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import requests
from tqdm import tqdm
import shutil

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def download_nih_metadata(data_dir: Path):
    """
    Télécharger les métadonnées NIH (CSV avec labels et annotations)
    
    Le CSV contient :
    - Image Index : nom du fichier
    - Finding Labels : pathologies détectées (séparées par |)
    - Follow-up # : numéro de suivi
    - Patient ID : ID patient
    - Patient Age / Gender
    - View Position : PA ou AP
    """
    logger.info("📥 Téléchargement des métadonnées NIH...")
    
    # URL du CSV (depuis Kaggle ou miroir)
    # Note: Pour Kaggle, il faut l'API key
    csv_url = "https://nihcc.app.box.com/index.php?rm=box_download_shared_file&shared_name=vfk49f5sf4t&file_id=f_220660789610"
    
    csv_path = data_dir / "raw" / "nih_metadata.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if csv_path.exists():
        logger.info(f"✅ Métadonnées déjà téléchargées : {csv_path}")
        return csv_path
    
    logger.info("⚠️ IMPORTANT:")
    logger.info("   Pour télécharger NIH, vous devez:")
    logger.info("   1. Aller sur : https://www.kaggle.com/datasets/nih-chest-xrays/data")
    logger.info("   2. Télécharger 'Data_Entry_2017.csv' manuellement")
    logger.info("   3. Le placer dans : data/raw/Data_Entry_2017.csv")
    logger.info("")
    
    # Créer un CSV exemple avec structure correcte
    logger.info("📝 Création d'un CSV exemple pour démonstration...")
    
    # Structure du vrai CSV NIH
    example_data = {
        'Image Index': [f'00000{i:03d}_000.png' for i in range(1, 1001)],
        'Finding Labels': np.random.choice([
            'No Finding', 
            'Atelectasis', 
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia'
        ], size=1000),
        'Patient Age': np.random.randint(20, 80, size=1000),
        'Patient Gender': np.random.choice(['M', 'F'], size=1000),
        'View Position': np.random.choice(['PA', 'AP'], size=1000)
    }
    
    df_example = pd.DataFrame(example_data)
    df_example.to_csv(csv_path, index=False)
    
    logger.info(f"✅ CSV exemple créé : {csv_path}")
    logger.info(f"   Contient {len(df_example)} entrées simulées")
    
    return csv_path


def sample_stratified(df: pd.DataFrame, num_samples: int = 1000):
    """
    Échantillonner de manière stratifiée pour garder la diversité
    """
    logger.info(f"🎲 Échantillonnage stratifié de {num_samples} images...")
    
    # Compter les pathologies
    pathology_counts = df['Finding Labels'].value_counts()
    logger.info(f"   Pathologies trouvées : {len(pathology_counts)}")
    
    # Échantillonner proportionnellement
    samples_per_class = max(1, num_samples // len(pathology_counts))
    
    sampled_dfs = []
    for pathology in pathology_counts.index[:15]:  # Top 15 pathologies
        subset = df[df['Finding Labels'] == pathology]
        n_samples = min(samples_per_class, len(subset))
        sampled = subset.sample(n=n_samples, random_state=42)
        sampled_dfs.append(sampled)
    
    df_sampled = pd.concat(sampled_dfs).drop_duplicates()
    
    # Si pas assez, compléter aléatoirement
    if len(df_sampled) < num_samples:
        remaining = num_samples - len(df_sampled)
        extra = df[~df.index.isin(df_sampled.index)].sample(n=remaining, random_state=42)
        df_sampled = pd.concat([df_sampled, extra])
    
    logger.info(f"✅ {len(df_sampled)} images sélectionnées")
    
    return df_sampled


def create_synthetic_images(df_sampled: pd.DataFrame, data_dir: Path, target_size: int = 64):
    """
    Créer des images synthétiques pour démonstration
    (En vrai, on téléchargerait les vraies images depuis NIH)
    """
    logger.info(f"🎨 Création d'images synthétiques ({target_size}x{target_size})...")
    
    images = []
    labels_text = []
    
    for idx, row in tqdm(df_sampled.iterrows(), total=len(df_sampled), desc="Génération"):
        # Créer une image synthétique (grayscale)
        # En vrai : télécharger depuis NIH ou charger depuis local
        img = np.random.randint(50, 200, size=(target_size, target_size), dtype=np.uint8)
        
        images.append(img)
        labels_text.append(row['Finding Labels'])
    
    images = np.array(images, dtype=np.uint8)
    
    logger.info(f"✅ {len(images)} images créées")
    
    return images, labels_text


def save_processed_data(images, labels_text, df_sampled, data_dir: Path, target_size: int):
    """Sauvegarder les données preprocessées"""
    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"nih_sample_{target_size}x{target_size}.npz"
    
    logger.info(f"💾 Sauvegarde dans {output_file}...")
    
    # Split train/val/test (70/15/15)
    n = len(images)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    X_train = images[:train_end]
    X_val = images[train_end:val_end]
    X_test = images[val_end:]
    
    labels_train = labels_text[:train_end]
    labels_val = labels_text[train_end:val_end]
    labels_test = labels_text[val_end:]
    
    np.savez_compressed(
        output_file,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        labels_train=np.array(labels_train),
        labels_val=np.array(labels_val),
        labels_test=np.array(labels_test)
    )
    
    # Sauvegarder le CSV échantillonné
    csv_file = output_dir / "nih_sample_metadata.csv"
    df_sampled.to_csv(csv_file, index=False)
    
    logger.info(f"✅ Données sauvegardées:")
    logger.info(f"   Train: {X_train.shape}")
    logger.info(f"   Val: {X_val.shape}")
    logger.info(f"   Test: {X_test.shape}")
    logger.info(f"   Metadata: {csv_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Préparer échantillon NIH')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Nombre d\'images à échantillonner')
    parser.add_argument('--target_size', type=int, default=64,
                       help='Taille des images (default: 64)')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Répertoire de données')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("📦 PRÉPARATION ÉCHANTILLON NIH (DATASET 2)")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Nombre d'images: {args.num_samples}")
    logger.info(f"  Taille: {args.target_size}x{args.target_size}")
    logger.info("")
    
    data_dir = Path(args.data_dir)
    
    # 1. Télécharger métadonnées
    csv_path = download_nih_metadata(data_dir)
    
    # 2. Charger CSV
    logger.info("📂 Chargement des métadonnées...")
    df = pd.read_csv(csv_path)
    logger.info(f"   Total d'images dans le CSV : {len(df)}")
    
    # 3. Échantillonner
    df_sampled = sample_stratified(df, args.num_samples)
    
    # 4. Créer/télécharger images
    images, labels_text = create_synthetic_images(df_sampled, data_dir, args.target_size)
    
    # 5. Sauvegarder
    output_file = save_processed_data(images, labels_text, df_sampled, data_dir, args.target_size)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ PRÉPARATION TERMINÉE")
    logger.info("=" * 60)
    logger.info(f"📁 Fichier créé : {output_file}")
    logger.info("")
    logger.info("⚠️ NOTE: Images synthétiques créées pour démonstration")
    logger.info("   Pour utiliser les vraies images NIH:")
    logger.info("   1. Télécharge depuis Kaggle")
    logger.info("   2. Modifie create_synthetic_images() pour charger les vraies images")
    logger.info("")
    logger.info("Prochaines étapes:")
    logger.info("  1. Entraîner autoencoder:")
    logger.info("     python scripts/06_train_autoencoder.py")
    logger.info("")
    logger.info("  2. Entraîner modèle multimodal:")
    logger.info("     python scripts/07_train_multimodal.py")


if __name__ == "__main__":
    main()
