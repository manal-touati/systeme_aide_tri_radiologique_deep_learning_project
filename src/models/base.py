"""
Classe de base pour tous les modèles du projet
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """
    Classe de base abstraite pour tous les modèles
    
    Tous les modèles doivent:
    - Prendre images (batch, 1, H, W) en entrée
    - Retourner logits (batch, 14) SANS activation finale
    - BCEWithLogitsLoss appliquera sigmoid automatiquement
    """
    
    def __init__(self, num_classes: int = 14, image_size: int = 64):
        """
        Args:
            num_classes: Nombre de classes (14 pour ChestMNIST)
            image_size: Taille des images en entrée
        """
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, 1, H, W) pour grayscale
            
        Returns:
            Logits (batch, num_classes) SANS activation finale
        """
        pass
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Prédictions binaires multi-label
        
        Args:
            x: Input tensor (batch, 1, H, W)
            threshold: Seuil pour binarisation
            
        Returns:
            Prédictions binaires (batch, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
        return preds
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Probabilités multi-label
        
        Args:
            x: Input tensor (batch, 1, H, W)
            
        Returns:
            Probabilités (batch, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs
    
    def count_parameters(self) -> int:
        """Compter le nombre de paramètres entraînables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """Informations sur le modèle"""
        return {
            'model_name': self.__class__.__name__,
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'num_parameters': self.count_parameters()
        }
