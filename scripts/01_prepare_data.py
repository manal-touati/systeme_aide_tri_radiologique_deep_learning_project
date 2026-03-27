#!/usr/bin/env python
"""
Script 1: Préparation des données ChestMNIST

Ce script télécharge et prétraite le dataset ChestMNIST à la résolution 64x64.
À exécuter UNE SEULE FOIS avant l'entraînement des modèles.

Usage:
    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py --force  # Pour re-télécharger
"""
import sys
from pathlib import Path
import logging

# Ajouter src au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.preprocess import preprocess_chestmnist

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Préparer les données ChestMNIST"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Préparer les données ChestMNIST')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Répertoire de données (default: ./data)')
    parser.add_argument('--target_size', type=int, default=64,
                        help='Taille des images (default: 64)')
    parser.add_argument('--force', action='store_true',
                        help='Forcer le re-téléchargement même si les données existent')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("📦 PRÉPARATION DES DONNÉES CHESTMNIST")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  - Répertoire: {args.data_dir}")
    logger.info(f"  - Taille: {args.target_size}x{args.target_size}")
    logger.info(f"  - Force: {args.force}")
    logger.info("")
    
    try:
        # Prétraiter les données
        output_file = preprocess_chestmnist(
            data_dir=args.data_dir,
            target_size=args.target_size,
            force=args.force
        )
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ PRÉPARATION TERMINÉE AVEC SUCCÈS")
        logger.info("=" * 60)
        logger.info(f"📁 Fichier de sortie: {output_file}")
        logger.info("")
        logger.info("Prochaines étapes:")
        logger.info("  1. Entraîner les modèles supervisés:")
        logger.info("     python scripts/02_train_supervised.py")
        logger.info("")
        logger.info("  2. Entraîner l'autoencoder:")
        logger.info("     python scripts/03_train_anomaly.py")
        logger.info("")
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ ERREUR LORS DE LA PRÉPARATION")
        logger.error("=" * 60)
        logger.error(f"Message: {e}")
        logger.error("")
        logger.error("Vérifications:")
        logger.error("  - Connexion internet active?")
        logger.error("  - medmnist installé? (pip install medmnist)")
        logger.error("  - Espace disque suffisant? (~100 MB)")
        sys.exit(1)


if __name__ == '__main__':
    main()
