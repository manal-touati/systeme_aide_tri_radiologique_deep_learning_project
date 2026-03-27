"""
Architecture CNN simple entraînée depuis zéro
"""
import torch
import torch.nn as nn
from typing import Tuple


class SimpleCNN(nn.Module):
    """
    CNN simple pour classification de radiographies thoraciques
    
    Architecture:
    - 3 blocs convolutionnels (Conv -> ReLU -> MaxPool)
    - 2 couches fully connected
    - Dropout pour régularisation
    """
    
    def __init__(
        self,
        num_classes: int = 14,
        input_channels: int = 1,  # Grayscale
        dropout_rate: float = 0.3,
        input_size: int = 128
    ):
        """
        Args:
            num_classes: Nombre de classes
            input_channels: Nombre de canaux d'entrée (1 pour grayscale)
            dropout_rate: Taux de dropout
            input_size: Taille de l'image d'entrée (128, 224, etc.)
        """
        super(SimpleCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        
        # Bloc 1 : Conv -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloc 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloc 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculer la taille après 3 pooling (chaque pooling divise par 2)
        # 128 -> 64 -> 32 -> 16 (pour input_size=128)
        # 224 -> 112 -> 56 -> 28 (pour input_size=224)
        final_size = input_size // (2 ** 3)  # 3 pooling
        fc_input_size = 128 * final_size * final_size
        
        # Fully Connected
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.relu_fc = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input (batch_size, channels, height, width)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Bloc 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        # Bloc 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        # Bloc 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_num_params(self) -> int:
        """Obtenir le nombre de paramètres"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
