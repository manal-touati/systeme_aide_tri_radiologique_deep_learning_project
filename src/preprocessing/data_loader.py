"""
Chargement de données et pipelines de preprocessing
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import logging
import shutil
from pathlib import Path
from typing import Tuple, List, Optional
from medmnist import ChestMNIST

logger = logging.getLogger(__name__)


class ChestRadiographyDataset(Dataset):
    """
    Dataset personnalisé pour radiographies thoraciques
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224
    ):
        """
        Args:
            image_paths: Liste des chemins vers les images
            labels: Liste des labels (indices de classe)
            transform: Transformations à appliquer
            image_size: Taille des images en sortie
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        
        assert len(image_paths) == len(labels), \
            f"Nombre d'images ({len(image_paths)}) ≠ nombre de labels ({len(labels)})"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Charger une image et son label
        
        Returns:
            (image_tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Charger image
        try:
            image = Image.open(img_path).convert('L')  # Grayscale
        except Exception as e:
            logger.error(f"Erreur chargement image {img_path}: {e}")
            raise
        
        # Appliquer transformations
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, label


def get_transforms(
    image_size: int = 224,
    augment: bool = False,
    is_training: bool = True
) -> transforms.Compose:
    """
    Créer pipeline de transformations
    
    Args:
        image_size: Taille cible
        augment: Appliquer augmentation de données
        is_training: Si True, appliquer augmentations
        
    Returns:
        Pipeline transforms.Compose
    """
    normalize = transforms.Normalize(
        mean=[0.485],  # Pour images grayscale
        std=[0.229]
    )
    
    if is_training and augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])


def create_dataloaders(
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    test_paths: List[str],
    test_labels: List[int],
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Créer les dataloaders train/val/test
    
    Args:
        train_paths, train_labels: Données d'entraînement
        val_paths, val_labels: Données de validation
        test_paths, test_labels: Données de test
        batch_size: Taille des batches
        image_size: Taille des images
        num_workers: Nombre de workers pour chargement
        augment: Activer augmentation
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Transformations
    train_transform = get_transforms(image_size, augment=augment, is_training=True)
    val_transform = get_transforms(image_size, augment=False, is_training=False)
    test_transform = get_transforms(image_size, augment=False, is_training=False)
    
    # Datasets
    train_dataset = ChestRadiographyDataset(train_paths, train_labels, train_transform, image_size)
    val_dataset = ChestRadiographyDataset(val_paths, val_labels, val_transform, image_size)
    test_dataset = ChestRadiographyDataset(test_paths, test_labels, test_transform, image_size)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Dataloaders créés :")
    logger.info(f"  Train: {len(train_dataset)} images")
    logger.info(f"  Val: {len(val_dataset)} images")
    logger.info(f"  Test: {len(test_dataset)} images")
    
    return train_loader, val_loader, test_loader


def load_chestmnist(data_dir: str = "data/raw", normalize: bool = True, target_size: int = 128) -> Tuple:
    """
    Charger ChestMNIST depuis les fichiers structurés avec redimensionnement optionnel
    
    Args:
        data_dir: Chemin vers le dossier contenant les données (train/val/test)
        normalize: Si True, normalise entre 0 et 1
        target_size: Taille cible pour redimensionner les images
        
    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    from PIL import Image
    
    data_path = Path(data_dir)
    
    def load_split(split_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Charger un split spécifique"""
        split_dir = data_path / split_name
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Dossier {split_dir} non trouvé")
        
        # Charger métadonnées
        metadata_file = data_path / f'{split_name}_metadata.csv'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Fichier {metadata_file} non trouvé")
        
        metadata_df = pd.read_csv(metadata_file)
        
        images = []
        labels = []
        
        for idx, row in metadata_df.iterrows():
            img_path = split_dir / row['filename']
            
            if not img_path.exists():
                logger.warning(f"Image manquante: {img_path}")
                continue
            
            # Charger image
            img = Image.open(img_path).convert('L')  # Grayscale
            
            # Redimensionner si nécessaire
            if img.size != (target_size, target_size):
                img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
            
            img_array = np.array(img, dtype=np.float32)
            
            # Normaliser
            if normalize:
                img_array = img_array / 255.0
            
            images.append(img_array)
            labels.append(int(row['label']))
        
        return np.array(images), np.array(labels)
    
    logger.info(f"Chargement ChestMNIST depuis {data_path} (target_size={target_size})...")
    
    X_train, y_train = load_split('train')
    X_val, y_val = load_split('val')
    X_test, y_test = load_split('test')
    
    # Ajouter dimension canal pour grayscale
    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    
    logger.info(f"✅ Données chargées:")
    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val: {X_val.shape}")
    logger.info(f"  Test: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_chestmnist_processed(data_dir: str = "data", target_size: int = 28) -> Tuple:
    """
    Charger ChestMNIST depuis fichier .npz preprocessé (data/processed/).
    Si le fichier n'existe pas, télécharge et redimensionne automatiquement.
    
    Args:
        data_dir: Chemin parent (doit contenir processed/)
        target_size: Taille cible pour les images
        
    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test) - normalisées 0-1
    """
    import shutil
    from medmnist import ChestMNIST
    
    data_path = Path(data_dir)
    processed_dir = data_path / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    processed_file = processed_dir / f'chestmnist_{target_size}x{target_size}.npz'
    
    # Si fichier existe, charger directement
    if processed_file.exists():
        logger.info(f"Chargement depuis {processed_file}...")
        data = np.load(processed_file)
        X_train = data['X_train'].astype(np.float32) / 255.0
        y_train = data['y_train'].astype(np.int64)
        X_val = data['X_val'].astype(np.float32) / 255.0
        y_val = data['y_val'].astype(np.int64)
        X_test = data['X_test'].astype(np.float32) / 255.0
        y_test = data['y_test'].astype(np.int64)
    else:
        # Télécharger et redimensionner
        logger.info(f"📥 Téléchargement ChestMNIST...")
        raw_dir = data_path / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        train_dataset = ChestMNIST(split='train', download=True, root=str(raw_dir))
        val_dataset = ChestMNIST(split='val', download=True, root=str(raw_dir))
        test_dataset = ChestMNIST(split='test', download=True, root=str(raw_dir))
        
        logger.info(f"📦 Redimensionnement à {target_size}x{target_size}...")
        train_images = []
        for img, _ in train_dataset:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img_resized = img.resize((target_size, target_size))
            train_images.append(np.array(img_resized))
        
        train_labels = [label for _, label in train_dataset]
        
        val_images = []
        for img, _ in val_dataset:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img_resized = img.resize((target_size, target_size))
            val_images.append(np.array(img_resized))
        
        val_labels = [label for _, label in val_dataset]
        
        test_images = []
        for img, _ in test_dataset:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img_resized = img.resize((target_size, target_size))
            test_images.append(np.array(img_resized))
        
        test_labels = [label for _, label in test_dataset]
        
        X_train = np.array(train_images, dtype=np.uint8)
        y_train = np.array(train_labels, dtype=np.int64)
        X_val = np.array(val_images, dtype=np.uint8)
        y_val = np.array(val_labels, dtype=np.int64)
        X_test = np.array(test_images, dtype=np.uint8)
        y_test = np.array(test_labels, dtype=np.int64)
        
        logger.info(f"💾 Sauvegarde vers {processed_file}...")
        np.savez(processed_file,
                 X_train=X_train, y_train=y_train,
                 X_val=X_val, y_val=y_val,
                 X_test=X_test, y_test=y_test)
        
        # Supprimer raw après traitement
        if raw_dir.exists():
            logger.info(f"🧹 Nettoyage {raw_dir}...")
            shutil.rmtree(raw_dir)
        
        # Normaliser
        X_train = X_train.astype(np.float32) / 255.0
        X_val = X_val.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
    
    # Ajouter dimension canal (grayscale)
    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    
    logger.info(f"✅ Données prêtes:")
    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val: {X_val.shape}")
    logger.info(f"  Test: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Module data_loader prêt")
