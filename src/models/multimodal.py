"""
Modèles multimodaux combinant image et texte
"""
import torch
import torch.nn as nn
from typing import Tuple


class MultimodalEarlyFusion(nn.Module):
    """
    Fusion précoce : Concatène features image et texte avant classification
    """
    
    def __init__(
        self,
        image_feature_dim: int = 512,
        text_feature_dim: int = 256,
        num_classes: int = 14,
        dropout_rate: float = 0.3
    ):
        """
        Args:
            image_feature_dim: Dimension des features image
            text_feature_dim: Dimension des features texte
            num_classes: Nombre de classes
            dropout_rate: Taux de dropout
        """
        super(MultimodalEarlyFusion, self).__init__()
        
        # Combinaison des features
        combined_dim = image_feature_dim + text_feature_dim
        
        # Classifier fusionné
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            image_features: Features extraites de l'image (batch_size, image_dim)
            text_features: Features extraites du texte (batch_size, text_dim)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Concaténer les features
        combined = torch.cat([image_features, text_features], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        
        return logits


class MultimodalLateFusion(nn.Module):
    """
    Fusion tardive : Classifie image et texte séparément puis combine
    """
    
    def __init__(
        self,
        image_feature_dim: int = 512,
        text_feature_dim: int = 256,
        num_classes: int = 14,
        dropout_rate: float = 0.3
    ):
        """
        Args:
            image_feature_dim: Dimension des features image
            text_feature_dim: Dimension des features texte
            num_classes: Nombre de classes
            dropout_rate: Taux de dropout
        """
        super(MultimodalLateFusion, self).__init__()
        
        # Classifier image
        self.image_classifier = nn.Sequential(
            nn.Linear(image_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Classifier texte
        self.text_classifier = nn.Sequential(
            nn.Linear(text_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Fusion des prédictions
        self.fusion_layer = nn.Linear(num_classes * 2, num_classes)
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            image_features: Features extraites de l'image
            text_features: Features extraites du texte
            
        Returns:
            Logits fusionnés (batch_size, num_classes)
        """
        # Prédictions séparées
        image_logits = self.image_classifier(image_features)
        text_logits = self.text_classifier(text_features)
        
        # Concaténer les logits
        combined_logits = torch.cat([image_logits, text_logits], dim=1)
        
        # Fusion finale
        fused_logits = self.fusion_layer(combined_logits)
        
        return fused_logits


class ImageTextModel(nn.Module):
    """
    Modèle complet image + texte avec encodeurs intégrés
    """
    
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        fusion_type: str = 'late',
        num_classes: int = 14
    ):
        """
        Args:
            image_encoder: Encodeur pour images (ex: ResNet)
            text_encoder: Encodeur pour texte (ex: BERT)
            fusion_type: 'early' ou 'late'
            num_classes: Nombre de classes
        """
        super(ImageTextModel, self).__init__()
        
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion_type = fusion_type
        
        # Déterminer les dimensions des features
        # À adapter selon les encodeurs utilisés
        image_dim = 512
        text_dim = 256
        
        # Choisir le type de fusion
        if fusion_type == 'early':
            self.fusion = MultimodalEarlyFusion(
                image_feature_dim=image_dim,
                text_feature_dim=text_dim,
                num_classes=num_classes
            )
        else:  # late
            self.fusion = MultimodalLateFusion(
                image_feature_dim=image_dim,
                text_feature_dim=text_dim,
                num_classes=num_classes
            )
    
    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            images: Batch d'images
            texts: Batch de textes encodés
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Extraire les features
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)
        
        # Fusionner et classifier
        logits = self.fusion(image_features, text_features)
        
        return logits


if __name__ == "__main__":
    # Test
    batch_size = 4
    num_classes = 14
    
    # Simuler des features
    img_feats = torch.randn(batch_size, 512)
    text_feats = torch.randn(batch_size, 256)
    
    # Test Early Fusion
    model_early = MultimodalEarlyFusion(512, 256, num_classes)
    out_early = model_early(img_feats, text_feats)
    print(f"Early Fusion output: {out_early.shape}")
    
    # Test Late Fusion
    model_late = MultimodalLateFusion(512, 256, num_classes)
    out_late = model_late(img_feats, text_feats)
    print(f"Late Fusion output: {out_late.shape}")
