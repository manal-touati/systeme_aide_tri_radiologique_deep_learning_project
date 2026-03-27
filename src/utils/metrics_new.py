"""
Métriques d'évaluation pour classification multi-label (ChestMNIST)
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, hamming_loss,
    multilabel_confusion_matrix
)
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Noms des 14 pathologies ChestMNIST
CHEST_PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculer les métriques pour classification multi-label
    
    Args:
        y_true: Labels vrais (N, 14) binaire
        y_pred: Prédictions binaires (N, 14)
        y_scores: Probabilités prédites (N, 14), optionnel pour ROC-AUC
        threshold: Seuil pour convertir probas en prédictions binaires
        
    Returns:
        Dictionnaire des métriques
    """
    metrics = {}
    
    # Métriques globales
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    metrics['exact_match'] = accuracy_score(y_true, y_pred)  # Toutes les labels correctes
    
    # Métriques micro/macro/weighted
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC et PR-AUC (nécessitent les probabilités)
    if y_scores is not None:
        try:
            metrics['roc_auc_micro'] = roc_auc_score(y_true, y_scores, average='micro')
            metrics['roc_auc_macro'] = roc_auc_score(y_true, y_scores, average='macro')
            metrics['roc_auc_weighted'] = roc_auc_score(y_true, y_scores, average='weighted')
            
            metrics['pr_auc_micro'] = average_precision_score(y_true, y_scores, average='micro')
            metrics['pr_auc_macro'] = average_precision_score(y_true, y_scores, average='macro')
            metrics['pr_auc_weighted'] = average_precision_score(y_true, y_scores, average='weighted')
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics['roc_auc_macro'] = 0.0
            metrics['pr_auc_macro'] = 0.0
    
    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    class_names: Optional[list] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculer les métriques par classe (pathologie)
    
    Args:
        y_true: Labels vrais (N, 14)
        y_pred: Prédictions (N, 14)
        y_scores: Probabilités (N, 14), optionnel
        class_names: Noms des classes, sinon utilise CHEST_PATHOLOGIES
        
    Returns:
        Dict avec métriques par classe
    """
    if class_names is None:
        class_names = CHEST_PATHOLOGIES
    
    n_classes = y_true.shape[1]
    per_class_metrics = {}
    
    for i in range(n_classes):
        class_name = class_names[i] if i < len(class_names) else f"class_{i}"
        
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        
        metrics = {
            'precision': precision_score(y_true_class, y_pred_class, zero_division=0),
            'recall': recall_score(y_true_class, y_pred_class, zero_division=0),
            'f1': f1_score(y_true_class, y_pred_class, zero_division=0),
            'support': int(y_true_class.sum())
        }
        
        # ROC-AUC par classe
        if y_scores is not None:
            try:
                y_scores_class = y_scores[:, i]
                metrics['roc_auc'] = roc_auc_score(y_true_class, y_scores_class)
                metrics['pr_auc'] = average_precision_score(y_true_class, y_scores_class)
            except:
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        
        per_class_metrics[class_name] = metrics
    
    return per_class_metrics


def print_metrics_summary(metrics: Dict[str, float], per_class: Optional[Dict] = None):
    """
    Afficher un résumé des métriques
    
    Args:
        metrics: Métriques globales
        per_class: Métriques par classe (optionnel)
    """
    logger.info("=" * 60)
    logger.info("📊 MÉTRIQUES D'ÉVALUATION")
    logger.info("=" * 60)
    
    # Métriques globales
    logger.info("\nMétriques Globales:")
    logger.info(f"  Exact Match Ratio: {metrics.get('exact_match', 0):.4f}")
    logger.info(f"  Hamming Loss: {metrics.get('hamming_loss', 0):.4f}")
    
    logger.info("\nPrécision:")
    logger.info(f"  Micro: {metrics.get('precision_micro', 0):.4f}")
    logger.info(f"  Macro: {metrics.get('precision_macro', 0):.4f}")
    logger.info(f"  Weighted: {metrics.get('precision_weighted', 0):.4f}")
    
    logger.info("\nRappel:")
    logger.info(f"  Micro: {metrics.get('recall_micro', 0):.4f}")
    logger.info(f"  Macro: {metrics.get('recall_macro', 0):.4f}")
    logger.info(f"  Weighted: {metrics.get('recall_weighted', 0):.4f}")
    
    logger.info("\nF1-Score:")
    logger.info(f"  Micro: {metrics.get('f1_micro', 0):.4f}")
    logger.info(f"  Macro: {metrics.get('f1_macro', 0):.4f}")
    logger.info(f"  Weighted: {metrics.get('f1_weighted', 0):.4f}")
    
    if 'roc_auc_macro' in metrics:
        logger.info("\nROC-AUC:")
        logger.info(f"  Micro: {metrics.get('roc_auc_micro', 0):.4f}")
        logger.info(f"  Macro: {metrics.get('roc_auc_macro', 0):.4f}")
        logger.info(f"  Weighted: {metrics.get('roc_auc_weighted', 0):.4f}")
    
    # Métriques par classe (top 5)
    if per_class:
        logger.info("\nTop 5 Classes (par F1-Score):")
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1]['f1'], reverse=True)
        for class_name, class_metrics in sorted_classes[:5]:
            logger.info(f"  {class_name}:")
            logger.info(f"    F1: {class_metrics['f1']:.3f}, "
                       f"ROC-AUC: {class_metrics.get('roc_auc', 0):.3f}, "
                       f"Support: {class_metrics['support']}")
    
    logger.info("=" * 60)


def predictions_from_logits(logits: torch.Tensor, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convertir logits en prédictions binaires et probabilités
    
    Args:
        logits: Tensor (N, 14) sortie du modèle
        threshold: Seuil pour classification binaire
        
    Returns:
        (y_pred, y_scores)
        - y_pred: Prédictions binaires (N, 14)
        - y_scores: Probabilités (N, 14)
    """
    # Appliquer sigmoid pour obtenir probabilités
    y_scores = torch.sigmoid(logits).cpu().numpy()
    
    # Appliquer seuil pour obtenir prédictions binaires
    y_pred = (y_scores >= threshold).astype(np.float32)
    
    return y_pred, y_scores
