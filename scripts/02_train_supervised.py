#!/usr/bin/env python
"""
Script 2: Entraînement des modèles supervisés

Entraîne les 3 modèles supervisés pour classification multi-label:
1. CNN Simple
2. ResNet50 (Transfer Learning)
3. EfficientNet-B0 (Transfer Learning)

Utilise BCEWithLogitsLoss pour multi-label classification sur ChestMNIST (64x64, 14 classes).

Usage:
    python scripts/02_train_supervised.py
    python scripts/02_train_supervised.py --model cnn_simple --epochs 50 --batch_size 64
    python scripts/02_train_supervised.py --model resnet50 --epochs 100 --lr 0.0001
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
from torch.cuda.amp import autocast, GradScaler

# Imports du projet
from src.preprocessing.data_loader import load_chestmnist_data, create_dataloaders
from src.models.cnn_simple import SimpleCNN
from src.models.transfer_learning import TransferLearningModel
from src.utils.metrics_new import compute_multilabel_metrics, predictions_from_logits, print_metrics_summary
from src.training.mlflow_utils import MLFlowTracker


def create_model(model_name: str, num_classes: int = 14, pretrained: bool = True) -> nn.Module:
    """
    Créer un modèle supervisé
    
    Args:
        model_name: Nom du modèle ('cnn_simple', 'resnet50', 'efficientnet_b0')
        num_classes: Nombre de classes
        pretrained: Utiliser poids pré-entraînés
        
    Returns:
        Modèle PyTorch
    """
    logger.info(f"🔧 Création du modèle: {model_name}")
    
    if model_name == "cnn_simple":
        model = SimpleCNN(
            num_classes=num_classes,
            input_size=64,
            dropout_rate=0.3
        )
    elif model_name == "resnet50":
        model = TransferLearningModel(
            model_name="resnet50",
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=False
        )
    elif model_name == "efficientnet_b0":
        # EfficientNet via timm (torchvision n'a pas EfficientNet dans les versions anciennes)
        try:
            import timm
            model_timm = timm.create_model('efficientnet_b0', pretrained=pretrained, in_chans=1)
            model_timm.classifier = nn.Linear(model_timm.num_features, num_classes)
            model = model_timm
            logger.info("✅ EfficientNet-B0 créé via timm")
        except ImportError:
            logger.warning("⚠️ timm non installé, utilisation de ResNet50 par défaut")
            model = TransferLearningModel(
                model_name="resnet50",
                num_classes=num_classes,
                pretrained=pretrained,
                freeze_backbone=False
            )
    else:
        raise ValueError(f"Modèle non supporté: {model_name}")
    
    return model


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_amp: bool = False
) -> float:
    """
    Entraîner une époque
    
    Args:
        model: Modèle à entraîner
        train_loader: DataLoader d'entraînement
        criterion: Fonction de perte
        optimizer: Optimiseur
        device: Device (CPU/GPU)
        use_amp: Utiliser Automatic Mixed Precision
        
    Returns:
        Loss moyen de l'époque
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    scaler = GradScaler() if use_amp else None
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            logger.debug(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """
    Valider le modèle
    
    Args:
        model: Modèle à valider
        val_loader: DataLoader de validation
        criterion: Fonction de perte
        device: Device (CPU/GPU)
        
    Returns:
        (loss_moyen, métriques)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Convertir logits en prédictions
            y_pred, y_scores = predictions_from_logits(logits.cpu(), threshold=0.5)
            
            all_preds.append(y_pred)
            all_scores.append(y_scores)
            all_labels.append(labels.cpu().numpy())
    
    # Concaténer tous les batches
    all_preds = np.vstack(all_preds)
    all_scores = np.vstack(all_scores)
    all_labels = np.vstack(all_labels)
    
    avg_loss = total_loss / len(val_loader)
    metrics = compute_multilabel_metrics(all_labels, all_preds, all_scores, threshold=0.5)
    
    return avg_loss, metrics


def train_model(
    model_name: str = "cnn_simple",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    data_dir: str = "./data",
    models_dir: str = "./models",
    use_mlflow: bool = True,
    use_amp: bool = False,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Entraîner un modèle supervisé
    
    Args:
        model_name: Nom du modèle
        epochs: Nombre d'époques
        batch_size: Taille des batches
        learning_rate: Taux d'apprentissage
        weight_decay: Régularisation L2
        data_dir: Répertoire des données
        models_dir: Répertoire de sauvegarde des modèles
        use_mlflow: Utiliser MLflow pour le tracking
        use_amp: Utiliser Automatic Mixed Precision
        device_name: Device ('cuda' ou 'cpu')
    """
    # Setup
    device = torch.device(device_name)
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("🚀 ENTRAÎNEMENT DU MODÈLE SUPERVISÉ")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Learning Rate: {learning_rate}")
    logger.info(f"  Device: {device}")
    logger.info(f"  AMP: {use_amp}")
    
    # Charger les données
    logger.info("\n📂 Chargement des données...")
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
        augment=True
    )
    
    # Créer le modèle
    model = create_model(model_name, num_classes=14, pretrained=True)
    model = model.to(device)
    
    # Afficher info du modèle
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\n🧠 Modèle créé:")
    logger.info(f"  Total params: {num_params:,}")
    logger.info(f"  Trainable params: {num_trainable:,}")
    
    # Perte (BCEWithLogitsLoss pour multi-label)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    
    # Optimiseur avec scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # MLflow tracking
    mlflow_tracker = None
    if use_mlflow:
        mlflow_tracker = MLFlowTracker(
            experiment_name="Supervised ChestMNIST Classification"
        )
        mlflow_tracker.start_run(
            run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={
                "model_type": "supervised",
                "dataset": "chestmnist_64x64",
                "task": "multi_label_classification"
            }
        )
        mlflow_tracker.log_params({
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": "Adam",
            "loss_function": "BCEWithLogitsLoss",
            "use_amp": use_amp,
            "device": str(device)
        })
    
    # Entraînement
    logger.info("\n⏱️ Démarrage de l'entraînement...")
    best_f1 = 0.0
    best_epoch = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'learning_rate': []
    }
    
    for epoch in range(1, epochs + 1):
        # Entraînement
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, use_amp=use_amp
        )
        
        # Validation
        val_loss, metrics = validate(model, val_loader, criterion, device)
        
        # Mise à jour du learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Sauvegarder l'historique
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_f1'].append(metrics['f1_weighted'])
        training_history['val_precision'].append(metrics['precision_weighted'])
        training_history['val_recall'].append(metrics['recall_weighted'])
        training_history['learning_rate'].append(current_lr)
        
        # Log
        if epoch % 1 == 0:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"F1 (weighted): {metrics['f1_weighted']:.4f} | "
                f"Precision: {metrics['precision_weighted']:.4f} | "
                f"Recall: {metrics['recall_weighted']:.4f} | "
                f"LR: {current_lr:.2e}"
            )
        
        # MLflow logging
        if mlflow_tracker:
            mlflow_tracker.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_f1_weighted': metrics['f1_weighted'],
                'val_precision_weighted': metrics['precision_weighted'],
                'val_recall_weighted': metrics['recall_weighted'],
                'learning_rate': current_lr
            }, step=epoch)
        
        # Sauvegarder le meilleur modèle
        if metrics['f1_weighted'] > best_f1:
            best_f1 = metrics['f1_weighted']
            best_epoch = epoch
            
            model_path = models_path / f"{model_name}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'metrics': metrics
            }, model_path)
            
            logger.info(f"  ✅ Meilleur modèle sauvegardé: {model_path}")
    
    # Évaluation finale sur test set
    logger.info("\n📊 Évaluation finale sur le test set...")
    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    
    logger.info("\n✅ Résultats du test set:")
    print_metrics_summary(test_metrics)
    
    # Sauvegarder le modèle final
    model_path = models_path / f"{model_name}_final.pt"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_metrics': test_metrics,
        'training_history': training_history
    }, model_path)
    logger.info(f"✅ Modèle final sauvegardé: {model_path}")
    
    # Sauvegarder les résultats en JSON
    results = {
        'model_name': model_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'device': str(device),
        'best_epoch': best_epoch,
        'best_val_f1': best_f1,
        'test_loss': float(test_loss),
        'test_metrics': {k: float(v) if isinstance(v, np.floating) else v 
                         for k, v in test_metrics.items()},
        'training_history': {k: [float(v) if isinstance(v, np.floating) else v 
                                 for v in vals]
                            for k, vals in training_history.items()},
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = models_path / f"{model_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✅ Résultats sauvegardés: {results_path}")
    
    # MLflow - finalisation
    if mlflow_tracker:
        mlflow_tracker.log_metrics(test_metrics, step=epochs)
        mlflow_tracker.log_artifacts({
            'results': str(results_path),
            'model': str(model_path)
        })
        mlflow_tracker.end_run()
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Entraînement terminé!")
    logger.info("=" * 70)
    
    return model, results


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description='Entraîner les modèles supervisés pour ChestMNIST'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='cnn_simple',
        choices=['cnn_simple', 'resnet50', 'efficientnet_b0'],
        help='Modèle à entraîner'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
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
        '--weight_decay',
        type=float,
        default=1e-5,
        help='Régularisation L2'
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
    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        use_mlflow=not args.no_mlflow,
        use_amp=args.amp,
        device_name=args.device
    )


if __name__ == "__main__":
    main()
