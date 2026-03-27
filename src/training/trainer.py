"""
Trainer pour classification multi-label avec BCEWithLogitsLoss
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class MultiLabelTrainer:
    """
    Trainer pour classification multi-label (ChestMNIST)
    Utilise BCEWithLogitsLoss
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()  # Multi-label
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Entraîner une epoch"""
        self.model.train()
        total_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Valider le modèle"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        early_stopping_patience: int = 5,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Entraîner le modèle
        
        Returns:
            Historique d'entraînement
        """
        logger.info(f"🚀 Starting training for {epochs} epochs")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Model: {self.model.__class__.__name__}")
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, save_path)
                    logger.info(f"   ✅ Best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"   ⚠️ Early stopping at epoch {epoch+1}")
                    break
        
        logger.info(f"✅ Training completed. Best val loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
