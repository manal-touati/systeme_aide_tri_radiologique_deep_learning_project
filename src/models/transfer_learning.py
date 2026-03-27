"""
Transfer Learning avec backbones pré-entraînés (ResNet, EfficientNet)
"""
import torch
import torch.nn as nn
import timm


class TransferLearningModel(nn.Module):
    """
    Modèle de transfer learning utilisant des backbones pré-entraînés
    """
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        num_classes: int = 14,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Args:
            model_name: Nom du modèle (resnet50, efficientnet_b0, etc.)
            num_classes: Nombre de classes de sortie
            pretrained: Utiliser poids pré-entraînés ImageNet
            freeze_backbone: Geler les poids du backbone
        """
        super(TransferLearningModel, self).__init__()
        
        self.model_name = model_name
        
        # Charger le modèle pré-entraîné via timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,  # Grayscale
            num_classes=0  # Pas de classifier (on ajoute le nôtre)
        )
        
        # Récupérer la dimension des features
        num_features = self.backbone.num_features
        
        # Geler le backbone si demandé
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classifier personnalisé pour multi-label
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Tensor d'images (batch_size, 1, H, W)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def unfreeze_backbone(self):
        """Dégeler le backbone pour fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Test
    model = TransferLearningModel(model_name='resnet50', num_classes=14)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
