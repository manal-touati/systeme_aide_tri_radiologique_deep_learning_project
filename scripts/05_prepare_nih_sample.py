#!/usr/bin/env python
"""
Script 05: Préparation échantillon NIH (Dataset 2)

Prépare un sous-ensemble réel du dataset NIH CXR14 à partir des fichiers
téchargés manuellement (CSV + images). Pas de génération synthétique.

Usage:
    python scripts/05_prepare_nih_sample.py --num_samples 5000 \
        --csv_path ./data/raw/Data_Entry_2017.csv \
        --images_dir ./data/raw/images_001 \
        --target_size 64
"""
import sys
from pathlib import Path
import logging
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_nih_metadata(csv_path: Path) -> pd.DataFrame:
    """Charger le CSV NIH téléchargé manuellement (Data_Entry_2017.csv)."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV non trouvé: {csv_path}\n"
            "Place Data_Entry_2017.csv dans data/raw/ ou précise --csv_path"
        )
    df = pd.read_csv(csv_path)
    logger.info(f"Metadonnees chargees: {csv_path} ({len(df)} lignes)")
    return df


def sample_stratified(df: pd.DataFrame, num_samples: int = 1000):
    """Échantillonnage simple pour garder de la diversité de labels."""
    logger.info(f"Echantillonnage de {num_samples} lignes…")

    pathology_counts = df['Finding Labels'].value_counts()
    logger.info(f"   Pathologies uniques: {len(pathology_counts)}")

    samples_per_class = max(1, num_samples // len(pathology_counts))

    sampled_dfs = []
    for pathology in pathology_counts.index:
        subset = df[df['Finding Labels'] == pathology]
        n_samples = min(samples_per_class, len(subset))
        if n_samples > 0:
            sampled_dfs.append(subset.sample(n=n_samples, random_state=42))

    df_sampled = pd.concat(sampled_dfs).drop_duplicates()

    if len(df_sampled) > num_samples:
        df_sampled = df_sampled.sample(n=num_samples, random_state=42)
    elif len(df_sampled) < num_samples:
        remaining = num_samples - len(df_sampled)
        extra = df[~df.index.isin(df_sampled.index)].sample(n=remaining, random_state=42)
        df_sampled = pd.concat([df_sampled, extra])

    logger.info(f"{len(df_sampled)} images selectionnees")
    return df_sampled


def load_images(df_sampled: pd.DataFrame, images_dir: Path, target_size: int = 64):
    """Charger les vraies images NIH en recherchant récursivement dans images_dir."""
    images = []
    labels_text = []
    missing = 0
    failed = 0

    # Indexer tous les fichiers images (png/jpg/jpeg) récursivement
    all_files = list(images_dir.rglob('*.png')) + list(images_dir.rglob('*.jpg')) + list(images_dir.rglob('*.jpeg'))
    path_map = {p.name: p for p in all_files}
    logger.info(f"Indexation images dans {images_dir} (recursif) : {len(path_map)} fichiers trouves")

    for _, row in tqdm(df_sampled.iterrows(), total=len(df_sampled), desc="Chargement"):
        fname = row['Image Index']
        fpath = path_map.get(fname)
        if fpath is None:
            missing += 1
            continue
        try:
            img = Image.open(fpath).convert('L').resize((target_size, target_size))
        except Exception as e:
            logger.warning(f"Ignoré {fname}: {e}")
            failed += 1
            continue
        images.append(np.array(img, dtype=np.uint8))
        labels_text.append(row['Finding Labels'])

    images = np.array(images, dtype=np.uint8)
    logger.info(f"{len(images)} images chargees depuis {images_dir}")
    if missing:
        logger.info(f"{missing} fichiers manquants (nom absent de l'index)")
    if failed:
        logger.info(f"{failed} fichiers ignores (erreurs de lecture ou corruption)")
    return images, labels_text


def save_processed_data(images, labels_text, df_sampled, data_dir: Path, target_size: int):
    """Sauvegarder les données preprocessées"""
    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"nih_sample_{target_size}x{target_size}.npz"

    logger.info(f"Sauvegarde dans {output_file}...")

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

    csv_file = output_dir / "nih_sample_metadata.csv"
    df_sampled.to_csv(csv_file, index=False)

    logger.info(f"Donnees sauvegardees:")
    logger.info(f"   Train: {X_train.shape}")
    logger.info(f"   Val: {X_val.shape}")
    logger.info(f"   Test: {X_test.shape}")
    logger.info(f"   Metadata: {csv_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Préparer échantillon NIH (réel)')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Nombre d\'images à échantillonner')
    parser.add_argument('--target_size', type=int, default=64,
                       help='Taille des images (default: 64)')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Répertoire de données (contiendra processed/)')
    parser.add_argument('--csv_path', type=str, default='./data/raw/Data_Entry_2017.csv',
                       help='Chemin vers le CSV NIH téléchargé')
    parser.add_argument('--images_dir', type=str, default='./data/raw/images',
                       help='Dossier contenant les images NIH téléchargées')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("PREPARATION ECHANTILLON NIH (DATASET 2)")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Nombre d'images: {args.num_samples}")
    logger.info(f"  Taille: {args.target_size}x{args.target_size}")
    logger.info("")
    
    data_dir = Path(args.data_dir)
    
    csv_path = Path(args.csv_path)
    images_dir = Path(args.images_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Dossier images introuvable: {images_dir}")

    # 1. Charger métadonnées
    df = load_nih_metadata(csv_path)
    
    # 2. Échantillonner
    df_sampled = sample_stratified(df, args.num_samples)

    # 3. Charger les images réelles
    images, labels_text = load_images(df_sampled, images_dir, args.target_size)

    if len(images) == 0:
        raise RuntimeError("Aucune image chargée. Vérifie images_dir et noms d'images.")
    
    # 4. Sauvegarder
    output_file = save_processed_data(images, labels_text, df_sampled, data_dir, args.target_size)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("PREPARATION TERMINEE")
    logger.info("=" * 60)
    logger.info(f"Fichier cree : {output_file}")
    logger.info("")
    logger.info("Prochaines étapes:")
    logger.info("  1. Entraîner autoencoder:")
    logger.info("     python scripts/06_train_autoencoder.py")
    logger.info("")
    logger.info("  2. Entraîner modèle multimodal:")
    logger.info("     python scripts/07_train_multimodal.py")


if __name__ == "__main__":
    main()
