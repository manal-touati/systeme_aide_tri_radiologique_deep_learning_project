#!/usr/bin/env python
"""
Script 3: Entraînement d'un autoencoder pour détection d'anomalies

Entraîne un autoencoder convolutionnel sur ChestMNIST pour:
- Apprentissage non-supervisé des features normales
- Calcul de scores d'anomalie basés sur l'erreur de reconstruction
- Détection d'images pathologiques rares

Architecture:
- Encoder: Conv2d -> BatchNorm -> ReLU -> MaxPool
- Latent Space: 64 dimensions
- Decoder: ConvTranspose2d -> BatchNorm -> ReLU -> Sigmoid

Loss: MSE (Mean Squared Error)
Score anomalie: Erreur de reconstruction moyenne

Usage:
    python scripts/03_train_anomaly.py
    python scripts/03_train_anomaly.py --epochs 100 --batch_size 64 --latent_dim 128
    python scripts/03_train_anomaly.py --use_vae  # Utiliser Variational AE
"""
import sys
from pathlib import Path
import logging
import argparse
import json
from datetime import datetime
import numpy as np

# Configuration du logging AVANT les imports torch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Ajouter src au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Imports du projet
from src.preprocessing.data_loader import load_chestmnist_data, create_dataloaders
from src.models.autoencoder import Autoencoder, VariationalAutoencoder
from src.training.mlflow_utils import MLFlowTracker


def compute_anomaly_scores(
    model: nn.Module,
    data_loader,
    device: torch.device,
    use_vae: bool = False
) -> np.ndarray:
    """
    Calculer les scores d'anomalie (erreur de reconstruction) pour un dataset
    
    Args:
        model: Autoencoder
        data_loader: DataLoader
        device: Device (CPU/GPU)
        use_vae: Si True, utiliser VAE pour les scores
        
    Returns:
        Array (N,) des scores d'anomalie
    """
    model.eval()
    anomaly_scores = []
    
    with torch.no_grad():
        for images, _ in data_loader:  # Labels non utilisés
            images = images.to(device)
            
            if use_vae:
                reconstruction, mu, logvar = model(images)
                # Score = reconstruction error + KL term
                mse = F.mse_loss(reconstruction, images, reduction='none')
                mse_score = mse.mean(dim=[1, 2, 3])
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                scores = mse_score + 0.0005 * kl  # kl_weight
            else:
                reconstruction, _ = model(images)
                mse = F.mse_loss(reconstruction, images, reduction='none')
                scores = mse.mean(dim=[1, 2, 3])
            
            anomaly_scores.extend(scores.cpu().numpy())
    
    return np.array(anomaly_scores)


def train_epoch_ae(
    model: nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_vae: bool = False,
    use_amp: bool = False
) -> float:
    """
    Entraîner une époque d'autoencoder
    
    Args:
        model: Autoencoder
        train_loader: DataLoader d'entraînement
        optimizer: Optimiseur
        device: Device (CPU/GPU)
        use_vae: Si True, utiliser VAE (avec KL divergence)
        use_amp: Utiliser Automatic Mixed Precision
        
    Returns:
        Loss moyen
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    scaler = GradScaler() if use_amp else None
    
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)
        
        optimizer.zero_grad()
        
        if use_vae:
            if use_amp:
                with autocast():
                    reconstruction, mu, logvar = model(images)
                    # Reconstruction loss
                    recon_loss = F.mse_loss(reconstruction, images, reduction='mean')
                    # KL divergence
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    # Total loss
                    loss = recon_loss + 0.0005 * kl_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                reconstruction, mu, logvar = model(images)
                recon_loss = F.mse_loss(reconstruction, images, reduction='mean')
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.0005 * kl_loss
                loss.backward()
                optimizer.step()
        else:
            # Standard AE
            if use_amp:
                with autocast():
                    reconstruction, _ = model(images)
                    loss = F.mse_loss(reconstruction, images, reduction='mean')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                reconstruction, _ = model(images)
                loss = F.mse_loss(reconstruction, images, reduction='mean')
                loss.backward()
                optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            logger.debug(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_ae(
    model: nn.Module,
    val_loader,
    device: torch.device,
    use_vae: bool = False
) -> tuple:
    """
    Valider l'autoencoder
    
    Args:
        model: Autoencoder
        val_loader: DataLoader de validation
        device: Device (CPU/GPU)
        use_vae: Si True, utiliser VAE
        
    Returns:
        (loss_moyen, scores_anomalie_moyens)
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            
            if use_vae:
                reconstruction, mu, logvar = model(images)
                recon_loss = F.mse_loss(reconstruction, images, reduction='mean')
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.0005 * kl_loss
            else:
                reconstruction, _ = model(images)
                loss = F.mse_loss(reconstruction, images, reduction='mean')
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def train_autoencoder(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    latent_dim: int = 64,
    data_dir: str = "./data",
    models_dir: str = "./models",
    use_vae: bool = False,
    use_mlflow: bool = True,
    use_amp: bool = False,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Entraîner un autoencoder pour détection d'anomalies
    
    Args:
        epochs: Nombre d'époques
        batch_size: Taille des batches
        learning_rate: Taux d'apprentissage
        latent_dim: Dimension de l'espace latent
        data_dir: Répertoire des données
        models_dir: Répertoire de sauvegarde des modèles
        use_vae: Utiliser Variational Autoencoder au lieu d'AE simple
        use_mlflow: Utiliser MLflow pour le tracking
        use_amp: Utiliser Automatic Mixed Precision
        device_name: Device ('cuda' ou 'cpu')
    """
    # Setup
    device = torch.device(device_name)
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    
    ae_type = "VAE" if use_vae else "Autoencoder"
    
    logger.info("=" * 70)
    logger.info(f"ENTRAINEMENT D'UN {ae_type} POUR DETECTION D'ANOMALIES")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Type: {ae_type}")
    logger.info(f"  Latent Dim: {latent_dim}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Learning Rate: {learning_rate}")
    logger.info(f"  Device: {device}")
    logger.info(f"  AMP: {use_amp}")
    
    # Charger les données
    logger.info("\nChargement des donnees...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_chestmnist_data(
        data_dir=data_dir,
        target_size=64
    )
    
    # Créer les dataloaders (num_workers=0 pour Windows)
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size,
        image_size=64,
        num_workers=0,  # Windows
        augment=False  # Pas d'augmentation pour AE
    )
    
    # Créer le modèle
    logger.info(f"\nCreation du {ae_type}...")
    if use_vae:
        model = VariationalAutoencoder(
            latent_dim=latent_dim,
            input_channels=1,
            kl_weight=0.0005
        )
    else:
        model = Autoencoder(
            latent_dim=latent_dim,
            input_channels=1
        )
    
    model = model.to(device)
    
    # Info du modèle
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{ae_type} cree:")
    logger.info(f"  Total params: {num_params:,}")
    logger.info(f"  Latent dim: {latent_dim}")
    
    # Optimiseur
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # MLflow tracking
    mlflow_tracker = None
    if use_mlflow:
        mlflow_tracker = MLFlowTracker(
            experiment_name="Anomaly Detection ChestMNIST"
        )
        mlflow_tracker.start_run(
            run_name=f"{ae_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={
                "model_type": "unsupervised",
                "task": "anomaly_detection",
                "dataset": "chestmnist_64x64"
            }
        )
        mlflow_tracker.log_params({
            "model_type": ae_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "latent_dim": latent_dim,
            "optimizer": "Adam",
            "use_amp": use_amp,
            "device": str(device)
        })
    
    # Entraînement
    logger.info("\nDemarrage de l'entrainement...")
    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    for epoch in range(1, epochs + 1):
        # Entraînement
        train_loss = train_epoch_ae(
            model, train_loader, optimizer, device, use_vae=use_vae, use_amp=use_amp
        )
        
        # Validation
        val_loss = validate_ae(model, val_loader, device, use_vae=use_vae)
        
        # Mise à jour du learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Sauvegarder l'historique
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['learning_rate'].append(current_lr)
        
        # Log
        if epoch % 1 == 0:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {current_lr:.2e}"
            )
        
        # MLflow logging
        if mlflow_tracker:
            mlflow_tracker.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr
            }, step=epoch)
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            model_path = models_path / f"{ae_type.lower()}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'latent_dim': latent_dim
            }, model_path)
            
            logger.info(f"  Meilleur modele sauvegarde: {model_path}")
    
    # Calculer les scores d'anomalie sur tous les datasets
    logger.info("\nCalcul des scores d'anomalie...")
    
    train_scores = compute_anomaly_scores(
        model, train_loader, device, use_vae=use_vae
    )
    val_scores = compute_anomaly_scores(
        model, val_loader, device, use_vae=use_vae
    )
    test_scores = compute_anomaly_scores(
        model, test_loader, device, use_vae=use_vae
    )
    
    logger.info(f"Scores d'anomalie calcules:")
    logger.info(f"  Train: mean={train_scores.mean():.4f}, std={train_scores.std():.4f}, "
                f"min={train_scores.min():.4f}, max={train_scores.max():.4f}")
    logger.info(f"  Val: mean={val_scores.mean():.4f}, std={val_scores.std():.4f}, "
                f"min={val_scores.min():.4f}, max={val_scores.max():.4f}")
    logger.info(f"  Test: mean={test_scores.mean():.4f}, std={test_scores.std():.4f}, "
                f"min={test_scores.min():.4f}, max={test_scores.max():.4f}")
    
    # Sauvegarder le modèle final
    model_path = models_path / f"{ae_type.lower()}_final.pt"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'latent_dim': latent_dim,
        'anomaly_scores': {
            'train_mean': float(train_scores.mean()),
            'train_std': float(train_scores.std()),
            'val_mean': float(val_scores.mean()),
            'val_std': float(val_scores.std()),
            'test_mean': float(test_scores.mean()),
            'test_std': float(test_scores.std())
        }
    }, model_path)
    logger.info(f"Modele final sauvegarde: {model_path}")
    
    # Sauvegarder les résultats en JSON
    results = {
        'model_type': ae_type,
        'latent_dim': latent_dim,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'device': str(device),
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(train_loss),
        'final_val_loss': float(val_loss),
        'anomaly_scores': {
            'train': {
                'mean': float(train_scores.mean()),
                'std': float(train_scores.std()),
                'min': float(train_scores.min()),
                'max': float(train_scores.max()),
                'median': float(np.median(train_scores))
            },
            'val': {
                'mean': float(val_scores.mean()),
                'std': float(val_scores.std()),
                'min': float(val_scores.min()),
                'max': float(val_scores.max()),
                'median': float(np.median(val_scores))
            },
            'test': {
                'mean': float(test_scores.mean()),
                'std': float(test_scores.std()),
                'min': float(test_scores.min()),
                'max': float(test_scores.max()),
                'median': float(np.median(test_scores))
            }
        },
        'training_history': {k: [float(v) for v in vals]
                            for k, vals in training_history.items()},
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = models_path / f"{ae_type.lower()}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Resultats sauvegardes: {results_path}")
    
    # Sauvegarder les scores d'anomalie
    scores_path = models_path / f"{ae_type.lower()}_scores.npz"
    np.savez(scores_path,
             train_scores=train_scores,
             val_scores=val_scores,
             test_scores=test_scores)
    logger.info(f"Scores d'anomalie sauvegardes: {scores_path}")
    
    # MLflow - finalisation
    if mlflow_tracker:
        mlflow_tracker.log_artifacts({
            'results': str(results_path),
            'scores': str(scores_path),
            'model': str(model_path)
        })
        mlflow_tracker.end_run()
    
    logger.info("\n" + "=" * 70)
    logger.info("Entrainement du modele d'anomalies termine!")
    logger.info("=" * 70)
    
    return model, results


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description='Entraîner un autoencoder pour détection d\'anomalies'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Nombre d\'époques'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Taille des batches'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Taux d\'apprentissage'
    )
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=64,
        help='Dimension de l\'espace latent'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Répertoire des données'
    )
    parser.add_argument(
        '--models_dir',
        type=str,
        default='./models',
        help='Répertoire de sauvegarde des modèles'
    )
    parser.add_argument(
        '--use_vae',
        action='store_true',
        help='Utiliser Variational Autoencoder au lieu d\'AE simple'
    )
    parser.add_argument(
        '--no_mlflow',
        action='store_true',
        help='Désactiver MLflow tracking'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Utiliser Automatic Mixed Precision'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Entraîner le modèle
    train_autoencoder(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        use_vae=args.use_vae,
        use_mlflow=not args.no_mlflow,
        use_amp=args.amp,
        device_name=args.device
    )


if __name__ == "__main__":
    main()
