#!/usr/bin/env python
"""
Script 06: Entraînement Autoencoder pour détection d'anomalies (Dataset 2)

Entraîne un autoencoder simple pour détecter des anomalies dans les radiographies NIH.
Score d'anomalie = erreur de reconstruction MSE.

Usage:
    python scripts/06_train_autoencoder.py --epochs 10
"""
import sys
from pathlib import Path
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.data_loader import get_transforms

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SIMPLE AUTOENCODER FOR 64x64 IMAGES
# ============================================================================

class SimpleAutoencoder(nn.Module):
    """Autoencoder simple pour images 64x64"""
    def __init__(self, latent_dim=64):
        super().__init__()
        
        # Encoder: 64 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
        )
        
        self.fc_encode = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 8 * 8)
        
        # Decoder: 8 -> 16 -> 32 -> 64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),  # 32 -> 64
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        z = self.fc_encode(z)
        
        # Decode
        x = self.fc_decode(z)
        x = x.view(x.size(0), 128, 8, 8)
        x = self.decoder(x)
        
        return x, z


def load_nih_data(data_dir: str = "./data", target_size: int = 64):
    """Charger les données NIH échantillonnées"""
    data_path = Path(data_dir) / "processed" / f"nih_sample_{target_size}x{target_size}.npz"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Fichier non trouve : {data_path}\n"
            f"   Lance d'abord: python scripts/05_prepare_nih_sample.py"
        )
    
    logger.info(f"Chargement depuis {data_path}")
    data = np.load(data_path, allow_pickle=True)
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    
    logger.info(f"Donnees chargees:")
    logger.info(f"   Train: {X_train.shape}")
    logger.info(f"   Val: {X_val.shape}")
    logger.info(f"   Test: {X_test.shape}")
    
    return X_train, X_val, X_test


def create_dataloaders(X_train, X_val, X_test, batch_size=32, image_size=64):
    """Créer les dataloaders (sans labels pour autoencoder)"""
    transform = get_transforms(image_size, augment=False, is_training=False)
    
    # Convertir en tensors
    def to_tensor_dataset(X):
        tensors = []
        for img in X:
            img_pil = Image.fromarray(img, mode='L')
            img_tensor = transform(img_pil)
            tensors.append(img_tensor)
        return torch.stack(tensors)
    
    from PIL import Image
    
    logger.info("Conversion en tensors...")
    X_train_t = to_tensor_dataset(X_train)
    X_val_t = to_tensor_dataset(X_val)
    X_test_t = to_tensor_dataset(X_test)
    
    train_loader = DataLoader(
        TensorDataset(X_train_t, X_train_t),  # Input = Output pour AE
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(X_val_t, X_val_t),
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        TensorDataset(X_test_t, X_test_t),
        batch_size=batch_size,
        shuffle=False
    )
    
    logger.info(f"Dataloaders crees:")
    logger.info(f"   Train: {len(train_loader)} batches")
    logger.info(f"   Val: {len(val_loader)} batches")
    logger.info(f"   Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entraîner une epoch"""
    model.train()
    total_loss = 0.0
    
    for images, _ in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        
        optimizer.zero_grad()
        reconstructed, _ = model(images)
        loss = criterion(reconstructed, images)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Valider le modèle"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            reconstructed, _ = model(images)
            loss = criterion(reconstructed, images)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def compute_anomaly_scores(model, test_loader, device):
    """Calculer les scores d'anomalie (MSE par image)"""
    model.eval()
    scores = []
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Computing anomaly scores"):
            images = images.to(device)
            reconstructed, _ = model(images)
            
            # MSE par image
            mse_per_image = ((images - reconstructed) ** 2).mean(dim=[1, 2, 3])
            scores.extend(mse_per_image.cpu().numpy())
    
    return np.array(scores)


def main():
    parser = argparse.ArgumentParser(description='Entraîner Autoencoder')
    parser.add_argument('--epochs', type=int, default=10, help='Nombre d\'epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Taille batch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=64, help='Dimension latente')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("ENTRAINEMENT AUTOENCODER - ANOMALY DETECTION")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Latent dim: {args.latent_dim}")
    logger.info(f"  Device: {args.device}")
    logger.info("")
    
    # 1. Load data
    X_train, X_val, X_test = load_nih_data()
    
    # 2. Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test,
        batch_size=args.batch_size
    )
    
    # 3. Create model
    logger.info("Creation de l'autoencoder...")
    model = SimpleAutoencoder(latent_dim=args.latent_dim)
    model = model.to(args.device)
    logger.info(f"   Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 5. Training
    logger.info("")
    logger.info("Debut de l'entrainement...")
    logger.info("")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss = validate(model, val_loader, criterion, args.device)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = "models/autoencoder_best.pt"
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"   Meilleur modele sauvegarde")
    
    # 6. Compute anomaly scores
    logger.info("")
    logger.info("Calcul des scores d'anomalie sur test set...")
    
    # Load best model
    model.load_state_dict(torch.load("models/autoencoder_best.pt"))
    
    anomaly_scores = compute_anomaly_scores(model, test_loader, args.device)
    
    # Statistiques
    logger.info("")
    logger.info("Scores d'anomalie:")
    logger.info(f"   Mean: {anomaly_scores.mean():.4f}")
    logger.info(f"   Std: {anomaly_scores.std():.4f}")
    logger.info(f"   Min: {anomaly_scores.min():.4f}")
    logger.info(f"   Max: {anomaly_scores.max():.4f}")
    logger.info(f"   Percentile 95: {np.percentile(anomaly_scores, 95):.4f}")
    
    # Sauvegarder résultats
    results = {
        'best_val_loss': float(best_val_loss),
        'anomaly_scores_mean': float(anomaly_scores.mean()),
        'anomaly_scores_std': float(anomaly_scores.std()),
        'anomaly_scores_95percentile': float(np.percentile(anomaly_scores, 95)),
        'test_scores': anomaly_scores.tolist()
    }
    
    results_path = "models/autoencoder_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("ENTRAINEMENT TERMINE")
    logger.info("=" * 60)
    logger.info(f"Modèle sauvegardé : models/autoencoder_best.pt")
    logger.info(f"Résultats sauvegardés : {results_path}")


if __name__ == "__main__":
    main()
