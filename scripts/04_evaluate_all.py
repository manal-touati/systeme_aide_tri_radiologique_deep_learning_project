#!/usr/bin/env python
"""
Script 4: Évaluation complète de tous les modèles entraînés

Évalue les modèles suivants sur le test set:
1. CNN Simple
2. ResNet50
3. EfficientNet-B0
4. Autoencoder
5. Variational Autoencoder

Génère un rapport d'évaluation complet avec:
- Métriques de classification multi-label (Precision, Recall, F1, ROC-AUC)
- Métriques par classe (pathologie)
- Comparaison des modèles
- Visualisations (optionnel)

Usage:
    python scripts/04_evaluate_all.py
    python scripts/04_evaluate_all.py --models_dir ./models --results_dir ./results
    python scripts/04_evaluate_all.py --data_dir ./data --batch_size 64
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

# Imports du projet
from src.preprocessing.data_loader import load_chestmnist_data, create_dataloaders
from src.models.cnn_simple import SimpleCNN
from src.models.transfer_learning import TransferLearningModel
from src.models.autoencoder import Autoencoder, VariationalAutoencoder
from src.utils.metrics_new import (
    compute_multilabel_metrics, compute_per_class_metrics,
    predictions_from_logits, print_metrics_summary, CHEST_PATHOLOGIES
)


def load_supervised_model(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
    num_classes: int = 14
) -> nn.Module:
    """
    Charger un modèle supervisé depuis checkpoint
    
    Args:
        model_name: Nom du modèle
        checkpoint_path: Chemin vers le fichier .pt
        device: Device (CPU/GPU)
        num_classes: Nombre de classes
        
    Returns:
        Modèle chargé
    """
    logger.info(f"📂 Chargement du modèle: {model_name} depuis {checkpoint_path}")
    
    # Créer le modèle
    if model_name == "cnn_simple":
        model = SimpleCNN(num_classes=num_classes, input_size=64, dropout_rate=0.3)
    elif model_name == "resnet50":
        model = TransferLearningModel(
            model_name="resnet50",
            num_classes=num_classes,
            pretrained=False,
            freeze_backbone=False
        )
    elif model_name == "efficientnet_b0":
        try:
            import timm
            model = timm.create_model('efficientnet_b0', pretrained=False, in_chans=1)
            model.classifier = nn.Linear(model.num_features, num_classes)
        except ImportError:
            logger.warning("⚠️ timm non installé, utilisation de ResNet50 par défaut")
            model = TransferLearningModel(
                model_name="resnet50",
                num_classes=num_classes,
                pretrained=False,
                freeze_backbone=False
            )
    else:
        raise ValueError(f"Modèle supervisé non reconnu: {model_name}")
    
    # Charger les poids
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✅ Modèle chargé depuis {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    return model


def load_anomaly_model(
    model_type: str,
    checkpoint_path: Path,
    device: torch.device
) -> nn.Module:
    """
    Charger un modèle d'anomalies depuis checkpoint
    
    Args:
        model_type: Type d'autoencoder ('autoencoder' ou 'vae')
        checkpoint_path: Chemin vers le fichier .pt
        device: Device (CPU/GPU)
        
    Returns:
        Modèle chargé
    """
    logger.info(f"📂 Chargement du modèle: {model_type} depuis {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    latent_dim = checkpoint.get('latent_dim', 64)
    
    # Créer le modèle
    if model_type == "autoencoder":
        model = Autoencoder(latent_dim=latent_dim, input_channels=1)
    elif model_type == "vae":
        model = VariationalAutoencoder(latent_dim=latent_dim, input_channels=1)
    else:
        raise ValueError(f"Type d'autoencoder non reconnu: {model_type}")
    
    # Charger les poids
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✅ Modèle chargé depuis {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_supervised_model(
    model: nn.Module,
    test_loader,
    model_name: str,
    device: torch.device
) -> dict:
    """
    Évaluer un modèle supervisé
    
    Args:
        model: Modèle supervisé
        test_loader: DataLoader de test
        model_name: Nom du modèle (pour logs)
        device: Device (CPU/GPU)
        
    Returns:
        Dictionnaire des résultats d'évaluation
    """
    logger.info(f"\n📊 Évaluation du modèle: {model_name}")
    
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    # Concaténer
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = np.vstack(all_labels)
    
    # Convertir logits en prédictions et probabilités
    y_pred, y_scores = predictions_from_logits(
        torch.from_numpy(all_logits).float(),
        threshold=0.5
    )
    
    # Métriques globales
    global_metrics = compute_multilabel_metrics(
        all_labels, y_pred, y_scores, threshold=0.5
    )
    
    # Métriques par classe
    per_class_metrics = compute_per_class_metrics(
        all_labels, y_pred, y_scores, class_names=CHEST_PATHOLOGIES
    )
    
    results = {
        'model_name': model_name,
        'num_samples': len(all_labels),
        'global_metrics': global_metrics,
        'per_class_metrics': per_class_metrics,
        'predictions': {
            'y_pred': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
            'y_scores': y_scores.tolist() if isinstance(y_scores, np.ndarray) else y_scores,
            'y_true': all_labels.tolist()
        }
    }
    
    # Afficher résumé
    logger.info(f"✅ Résultats pour {model_name}:")
    logger.info(f"  F1 (weighted): {global_metrics['f1_weighted']:.4f}")
    logger.info(f"  Precision (weighted): {global_metrics['precision_weighted']:.4f}")
    logger.info(f"  Recall (weighted): {global_metrics['recall_weighted']:.4f}")
    logger.info(f"  ROC-AUC (macro): {global_metrics.get('roc_auc_macro', 0):.4f}")
    logger.info(f"  Hamming Loss: {global_metrics['hamming_loss']:.4f}")
    
    return results


def evaluate_anomaly_model(
    model: nn.Module,
    test_loader,
    model_name: str,
    device: torch.device,
    use_vae: bool = False
) -> dict:
    """
    Évaluer un modèle d'anomalies
    
    Args:
        model: Modèle d'anomalies
        test_loader: DataLoader de test
        model_name: Nom du modèle
        device: Device (CPU/GPU)
        use_vae: Si True, c'est un VAE
        
    Returns:
        Dictionnaire des résultats d'évaluation
    """
    logger.info(f"\n📊 Évaluation du modèle: {model_name}")
    
    model.eval()
    anomaly_scores = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            
            if use_vae:
                reconstruction, mu, logvar = model(images)
                # Score = reconstruction error + KL term
                mse = torch.nn.functional.mse_loss(
                    reconstruction, images, reduction='none'
                )
                mse_score = mse.mean(dim=[1, 2, 3])
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                scores = mse_score + 0.0005 * kl
            else:
                reconstruction, _ = model(images)
                mse = torch.nn.functional.mse_loss(
                    reconstruction, images, reduction='none'
                )
                scores = mse.mean(dim=[1, 2, 3])
            
            anomaly_scores.extend(scores.cpu().numpy())
    
    anomaly_scores = np.array(anomaly_scores)
    
    results = {
        'model_name': model_name,
        'model_type': 'vae' if use_vae else 'autoencoder',
        'num_samples': len(anomaly_scores),
        'anomaly_scores_stats': {
            'mean': float(np.mean(anomaly_scores)),
            'std': float(np.std(anomaly_scores)),
            'min': float(np.min(anomaly_scores)),
            'max': float(np.max(anomaly_scores)),
            'median': float(np.median(anomaly_scores)),
            'q25': float(np.percentile(anomaly_scores, 25)),
            'q75': float(np.percentile(anomaly_scores, 75))
        },
        'anomaly_scores': anomaly_scores.tolist()
    }
    
    # Afficher résumé
    logger.info(f"✅ Résultats pour {model_name}:")
    logger.info(f"  Anomaly Score Mean: {np.mean(anomaly_scores):.6f}")
    logger.info(f"  Anomaly Score Std: {np.std(anomaly_scores):.6f}")
    logger.info(f"  Anomaly Score Range: [{np.min(anomaly_scores):.6f}, {np.max(anomaly_scores):.6f}]")
    
    return results


def evaluate_all_models(
    data_dir: str = "./data",
    models_dir: str = "./models",
    results_dir: str = "./results",
    batch_size: int = 32,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Évaluer tous les modèles disponibles
    
    Args:
        data_dir: Répertoire des données
        models_dir: Répertoire des modèles
        results_dir: Répertoire de sauvegarde des résultats
        batch_size: Taille des batches
        device_name: Device ('cuda' ou 'cpu')
    """
    device = torch.device(device_name)
    models_path = Path(models_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("📊 ÉVALUATION COMPLÈTE DE TOUS LES MODÈLES")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Data dir: {data_dir}")
    logger.info(f"  Models dir: {models_dir}")
    logger.info(f"  Results dir: {results_dir}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Device: {device}")
    
    # Charger les données
    logger.info("\n📂 Chargement des données...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_chestmnist_data(
        data_dir=data_dir,
        target_size=64
    )
    
    # Créer le test loader
    _, _, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size,
        image_size=64,
        num_workers=0,
        augment=False
    )
    
    # Résultats consolidés
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'test_samples': len(y_test),
        'supervised_models': {},
        'anomaly_models': {},
        'summary': {}
    }
    
    # Modèles supervisés
    supervised_models = ['cnn_simple', 'resnet50', 'efficientnet_b0']
    supervised_results = {}
    
    for model_name in supervised_models:
        checkpoint_path = models_path / f"{model_name}_best.pt"
        
        if not checkpoint_path.exists():
            logger.warning(f"⚠️ Checkpoint non trouvé: {checkpoint_path}")
            continue
        
        try:
            model = load_supervised_model(model_name, checkpoint_path, device)
            results = evaluate_supervised_model(model, test_loader, model_name, device)
            all_results['supervised_models'][model_name] = results
            supervised_results[model_name] = results['global_metrics']
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'évaluation de {model_name}: {e}")
    
    # Modèles d'anomalies
    anomaly_models = [
        ('autoencoder', 'autoencoder_best.pt'),
        ('vae', 'vae_best.pt')
    ]
    
    for model_type, checkpoint_name in anomaly_models:
        checkpoint_path = models_path / checkpoint_name
        
        if not checkpoint_path.exists():
            logger.warning(f"⚠️ Checkpoint non trouvé: {checkpoint_path}")
            continue
        
        try:
            model = load_anomaly_model(model_type, checkpoint_path, device)
            use_vae = (model_type == 'vae')
            model_name = f"{model_type}" if model_type != 'vae' else "variational_autoencoder"
            results = evaluate_anomaly_model(
                model, test_loader, model_name, device, use_vae=use_vae
            )
            all_results['anomaly_models'][model_type] = results
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'évaluation de {model_type}: {e}")
    
    # Résumé comparatif
    if supervised_results:
        logger.info("\n" + "=" * 70)
        logger.info("📊 RÉSUMÉ COMPARATIF DES MODÈLES SUPERVISÉS")
        logger.info("=" * 70)
        
        comparison_data = []
        for model_name, metrics in supervised_results.items():
            comparison_data.append({
                'Model': model_name,
                'F1 (weighted)': metrics['f1_weighted'],
                'Precision': metrics['precision_weighted'],
                'Recall': metrics['recall_weighted'],
                'ROC-AUC': metrics.get('roc_auc_macro', 0),
                'Hamming Loss': metrics['hamming_loss']
            })
        
        # Afficher tableau
        for row in comparison_data:
            logger.info(
                f"{row['Model']:20} | "
                f"F1: {row['F1 (weighted)']:.4f} | "
                f"Prec: {row['Precision']:.4f} | "
                f"Rec: {row['Recall']:.4f} | "
                f"AUC: {row['ROC-AUC']:.4f} | "
                f"Ham: {row['Hamming Loss']:.4f}"
            )
        
        all_results['summary']['model_comparison'] = comparison_data
    
    # Sauvegarder les résultats complets
    results_json_path = results_path / "evaluation_results.json"
    
    # Convertir les arrays en listes pour JSON
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj
    
    all_results = convert_to_json_serializable(all_results)
    
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n✅ Résultats d'évaluation sauvegardés: {results_json_path}")
    
    # Sauvegarder un rapport texte
    report_path = results_path / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("📊 RAPPORT D'ÉVALUATION DES MODÈLES\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {all_results['timestamp']}\n")
        f.write(f"Device: {all_results['device']}\n")
        f.write(f"Test samples: {all_results['test_samples']}\n\n")
        
        # Modèles supervisés
        if all_results.get('supervised_models'):
            f.write("\n🧠 MODÈLES SUPERVISÉS\n")
            f.write("-" * 70 + "\n")
            for model_name, results in all_results['supervised_models'].items():
                metrics = results['global_metrics']
                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  F1 Score (weighted): {metrics['f1_weighted']:.4f}\n")
                f.write(f"  Precision (weighted): {metrics['precision_weighted']:.4f}\n")
                f.write(f"  Recall (weighted): {metrics['recall_weighted']:.4f}\n")
                f.write(f"  ROC-AUC (macro): {metrics.get('roc_auc_macro', 0):.4f}\n")
                f.write(f"  Hamming Loss: {metrics['hamming_loss']:.4f}\n")
                f.write(f"  Exact Match: {metrics['exact_match']:.4f}\n")
        
        # Modèles d'anomalies
        if all_results.get('anomaly_models'):
            f.write("\n\n🔍 MODÈLES D'ANOMALIES\n")
            f.write("-" * 70 + "\n")
            for model_type, results in all_results['anomaly_models'].items():
                stats = results['anomaly_scores_stats']
                f.write(f"\n{model_type.upper()}:\n")
                f.write(f"  Mean Score: {stats['mean']:.6f}\n")
                f.write(f"  Std Dev: {stats['std']:.6f}\n")
                f.write(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
                f.write(f"  Median: {stats['median']:.6f}\n")
                f.write(f"  IQR: [{stats['q25']:.6f}, {stats['q75']:.6f}]\n")
    
    logger.info(f"✅ Rapport texte sauvegardé: {report_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Évaluation complète terminée!")
    logger.info("=" * 70)
    
    return all_results


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description='Évaluer tous les modèles entraînés'
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
        help='Répertoire des modèles'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        help='Répertoire de sauvegarde des résultats'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Taille des batches'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Évaluer tous les modèles
    evaluate_all_models(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        device_name=args.device
    )


if __name__ == "__main__":
    main()
