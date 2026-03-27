"""
Fonctions de visualisation pour l'analyse des résultats
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch


def plot_training_history(history: dict, save_path: str = None):
    """
    Tracer les courbes de loss et métriques pendant l'entraînement
    
    Args:
        history: Dict avec 'train_loss', 'val_loss', 'train_metrics', 'val_metrics'
        save_path: Chemin pour sauvegarder la figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Metrics (F1 par exemple)
    if 'train_f1' in history and 'val_f1' in history:
        axes[1].plot(history['train_f1'], label='Train F1')
        axes[1].plot(history['val_f1'], label='Val F1')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Training and Validation F1')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Tracer la matrice de confusion
    
    Args:
        y_true: Labels vrais
        y_pred: Labels prédits
        class_names: Noms des classes
        save_path: Chemin pour sauvegarder
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curves(y_true, y_scores, class_names=None, save_path=None):
    """
    Tracer les courbes ROC pour classification multi-label
    
    Args:
        y_true: Labels vrais (N, num_classes)
        y_scores: Scores prédits (N, num_classes)
        class_names: Noms des classes
        save_path: Chemin pour sauvegarder
    """
    num_classes = y_true.shape[1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        
        label = f'{class_names[i]} (AUC={roc_auc:.2f})' if class_names else f'Class {i} (AUC={roc_auc:.2f})'
        ax.plot(fpr, tpr, label=label)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Multi-Label Classification')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_samples(images, labels=None, predictions=None, num_samples=8, save_path=None):
    """
    Visualiser des échantillons d'images avec labels/prédictions
    
    Args:
        images: Tensor ou array d'images
        labels: Labels vrais
        predictions: Prédictions
        num_samples: Nombre d'échantillons à afficher
        save_path: Chemin pour sauvegarder
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    
    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
    axes = axes.flatten() if rows > 1 else [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Afficher l'image
        if images[i].shape[0] == 1:  # Grayscale
            ax.imshow(images[i][0], cmap='gray')
        else:
            ax.imshow(images[i].transpose(1, 2, 0))
        
        # Titre avec label et prédiction
        title = f"Sample {i+1}"
        if labels is not None:
            title += f"\nTrue: {labels[i]}"
        if predictions is not None:
            title += f"\nPred: {predictions[i]}"
        
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    
    # Cacher les axes vides
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Test
    pass
