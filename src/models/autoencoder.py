"""
Autoencoder et Variational Autoencoder pour détection d'anomalies dans images médicales

L'anomaly score est basé sur l'erreur de reconstruction (MSE)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class Autoencoder(nn.Module):
    """
    Autoencoder convolutionnel pour détection d'anomalies
    
    Utilise l'erreur de reconstruction comme score d'anomalie.
    Architecture: Encoder -> Latent Space -> Decoder
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        input_channels: int = 1
    ):
        """
        Args:
            latent_dim: Dimension de l'espace latent
            input_channels: Canaux d'entrée (1 pour grayscale)
        """
        super(Autoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder: 224 -> 112 -> 56 -> 28 -> 14
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Mapping vers latent space
        self.fc_encode = nn.Linear(256 * 14 * 14, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 14 * 14)
        
        # Decoder: 14 -> 28 -> 56 -> 112 -> 224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder vers l'espace latent
        
        Args:
            x: Image (batch_size, channels, height, width)
            
        Returns:
            Vecteur latent (batch_size, latent_dim)
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc_encode(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decoder de l'espace latent vers l'image
        
        Args:
            z: Vecteur latent (batch_size, latent_dim)
            
        Returns:
            Image reconstruite (batch_size, channels, height, width)
        """
        x = self.fc_decode(z)
        x = x.view(x.size(0), 256, 14, 14)
        x = self.decoder(x)
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Image d'entrée
            
        Returns:
            (image_reconstruite, vecteur_latent)
        """
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z
    
    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Score d'anomalie = erreur de reconstruction (MSE)
        
        Args:
            x: Image
            
        Returns:
            Score d'anomalie (batch_size,)
        """
        reconstruction, _ = self.forward(x)
        mse = F.mse_loss(reconstruction, x, reduction='none')
        score = mse.mean(dim=[1, 2, 3])
        return score


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder pour détection d'anomalies
    
    Amélioration de l'AE standard avec:
    - Espace latent: distribution gaussienne N(0, I)
    - Loss = Reconstruction + KL divergence
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        input_channels: int = 1,
        kl_weight: float = 0.0005
    ):
        """
        Args:
            latent_dim: Dimension de l'espace latent
            input_channels: Canaux d'entrée (1 pour grayscale)
            kl_weight: Poids du terme KL
        """
        super(VariationalAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Latent space: μ et log(σ²)
        self.fc_mu = nn.Linear(256 * 14 * 14, latent_dim)
        self.fc_logvar = nn.Linear(256 * 14 * 14, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, 256 * 14 * 14)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encoder avec paramètres gaussiens
        
        Returns:
            (mu, logvar)
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick
        
        z = μ + σ * ε, où ε ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decoder"""
        x = self.fc_decode(z)
        x = x.view(x.size(0), 256, 14, 14)
        x = self.decoder(x)
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Score d'anomalie = Reconstruction error + KL term
        
        Permet de combiner reconstruction et prior
        """
        reconstruction, mu, logvar = self.forward(x)
        
        # Reconstruction error
        mse = F.mse_loss(reconstruction, x, reduction='none').mean(dim=[1, 2, 3])
        
        # KL divergence
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # Combined score
        score = mse + self.kl_weight * kl
        return score


if __name__ == "__main__":
    pass
