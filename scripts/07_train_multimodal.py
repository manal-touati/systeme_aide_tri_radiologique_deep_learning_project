#!/usr/bin/env python
"""
Script 07: Entraînement modèle multimodal (Dataset 2)

Entraîne 3 modèles pour comparer:
1. Image seule (CNN/ResNet)
2. Texte seul (encodeur simple)
3. Multimodal fusionné (early ou late fusion)

Usage:
    python scripts/07_train_multimodal.py --epochs 5 --fusion late
"""
import sys
from pathlib import Path
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import pandas as pd

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.data_loader import get_transforms
from PIL import Image
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODÈLES SIMPLIFIÉS
# ============================================================================

class ImageOnlyModel(nn.Module):
    """Modèle image seule (CNN simple)"""
    def __init__(self, num_classes=15):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class TextOnlyModel(nn.Module):
    """Modèle texte seul (encodeur bag-of-words simple)"""
    def __init__(self, vocab_size=100, num_classes=15):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, 128, mode='mean')
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, offsets=None):
        x = self.embedding(x, offsets)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class MultimodalFusionModel(nn.Module):
    """Modèle multimodal avec fusion"""
    def __init__(self, num_classes=15, fusion_type='late'):
        super().__init__()
        self.fusion_type = fusion_type
        
        # Image branch
        self.img_conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.img_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.img_pool = nn.MaxPool2d(2, 2)
        self.img_fc = nn.Linear(64 * 16 * 16, 128)
        
        # Text branch
        self.text_embedding = nn.EmbeddingBag(100, 128, mode='mean')
        self.text_fc = nn.Linear(128, 64)
        
        # Fusion
        if fusion_type == 'early':
            # Concatenate features
            self.fusion_fc = nn.Linear(128 + 64, 256)
        else:  # late fusion
            # Separate then combine
            self.img_classifier = nn.Linear(128, num_classes)
            self.text_classifier = nn.Linear(64, num_classes)
            self.fusion_fc = nn.Linear(num_classes * 2, num_classes)
        
        self.final_fc = nn.Linear(256 if fusion_type == 'early' else num_classes, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, img, text, offsets=None):
        # Image branch
        x_img = self.img_pool(torch.relu(self.img_conv1(img)))
        x_img = self.img_pool(torch.relu(self.img_conv2(x_img)))
        x_img = x_img.view(x_img.size(0), -1)
        x_img = torch.relu(self.img_fc(x_img))
        
        # Text branch
        x_text = self.text_embedding(text, offsets)
        x_text = torch.relu(self.text_fc(x_text))
        
        # Fusion
        if self.fusion_type == 'early':
            # Concatenate and fuse
            x = torch.cat([x_img, x_text], dim=1)
            x = self.dropout(torch.relu(self.fusion_fc(x)))
            x = self.final_fc(x)
        else:  # late fusion
            # Classify separately then fuse
            out_img = self.img_classifier(x_img)
            out_text = self.text_classifier(x_text)
            x = torch.cat([out_img, out_text], dim=1)
            x = self.fusion_fc(x)
        
        return x


# ============================================================================
# DATASET
# ============================================================================

class MultimodalDataset(Dataset):
    """Dataset multimodal (image + texte)"""
    def __init__(self, images, texts, labels, transform=None, vocab=None):
        self.images = images
        self.texts = texts
        self.labels = labels
        self.transform = transform
        self.vocab = vocab or self._build_vocab(texts)
    
    def _build_vocab(self, texts):
        """Construire vocabulaire simple"""
        all_words = set()
        for text in texts:
            words = text.lower().split('|')  # Séparées par |
            all_words.update(words)
        return {word: idx for idx, word in enumerate(sorted(all_words))}
    
    def _text_to_indices(self, text):
        """Convertir texte en indices"""
        words = text.lower().split('|')
        indices = [self.vocab.get(word, 0) for word in words if word in self.vocab]
        return torch.tensor(indices if indices else [0], dtype=torch.long)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Image
        img_pil = Image.fromarray(img, mode='L')
        if self.transform:
            img = self.transform(img_pil)
        else:
            img = transforms.ToTensor()(img_pil)
        
        # Text
        text_indices = self._text_to_indices(text)
        
        # Label (convert to multi-hot encoding)
        label_vec = torch.zeros(15, dtype=torch.float32)
        if label != 'No Finding':
            pathologies = label.split('|')
            for p in pathologies:
                if p.strip() in self.vocab:
                    idx_p = list(self.vocab.keys()).index(p.strip()) % 15
                    label_vec[idx_p] = 1.0
        
        return img, text_indices, label_vec


def load_nih_multimodal_data(data_dir="./data"):
    """Charger données NIH avec texte"""
    npz_path = Path(data_dir) / "processed" / "nih_sample_64x64.npz"
    csv_path = Path(data_dir) / "processed" / "nih_sample_metadata.csv"
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Fichier non trouve : {npz_path}")
    
    logger.info(f"Chargement depuis {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    df = pd.read_csv(csv_path)
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    
    labels_train = data['labels_train']
    labels_val = data['labels_val']
    labels_test = data['labels_test']
    
    logger.info(f"Donnees chargees:")
    logger.info(f"   Train: {len(X_train)} images")
    logger.info(f"   Val: {len(X_val)} images")
    logger.info(f"   Test: {len(X_test)} images")
    
    return (X_train, labels_train), (X_val, labels_val), (X_test, labels_test)


def collate_multimodal(batch):
    """Collate function pour gérer les longueurs variables de texte"""
    images, texts, labels = zip(*batch)
    
    images = torch.stack(images)
    labels = torch.stack(labels)
    
    # Flatten text indices et créer offsets
    text_list = []
    offsets = [0]
    for text in texts:
        text_list.extend(text.tolist())
        offsets.append(len(text_list))
    
    texts_tensor = torch.tensor(text_list, dtype=torch.long)
    offsets_tensor = torch.tensor(offsets[:-1], dtype=torch.long)
    
    return images, texts_tensor, offsets_tensor, labels


def train_model(model, train_loader, val_loader, epochs, device, model_name):
    """Entraîner un modèle"""
    logger.info(f"\nEntrainement : {model_name}")
    logger.info(f"   Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            if len(batch) == 4:
                images, texts, offsets, labels = batch
                images = images.to(device)
                texts = texts.to(device)
                offsets = offsets.to(device)
                labels = labels.to(device)

                if isinstance(model, TextOnlyModel):
                    outputs = model(texts, offsets)
                elif isinstance(model, MultimodalFusionModel):
                    outputs = model(images, texts, offsets)
                else:
                    outputs = model(images)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
            
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    images, texts, offsets, labels = batch
                    images = images.to(device)
                    texts = texts.to(device)
                    offsets = offsets.to(device)
                    labels = labels.to(device)

                    if isinstance(model, TextOnlyModel):
                        outputs = model(texts, offsets)
                    elif isinstance(model, MultimodalFusionModel):
                        outputs = model(images, texts, offsets)
                    else:
                        outputs = model(images)
                else:
                    images, labels = batch
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        logger.info(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"models/{model_name}_best.pt"
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), save_path)
    
    logger.info(f"Meilleur Val Loss: {best_val_loss:.4f}")
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Entraîner modèle multimodal')
    parser.add_argument('--epochs', type=int, default=5, help='Nombre d\'epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Taille batch')
    parser.add_argument('--fusion', choices=['early', 'late'], default='late', help='Type de fusion')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("ENTRAINEMENT MULTIMODAL (IMAGE + TEXTE)")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Fusion: {args.fusion}")
    logger.info(f"  Device: {args.device}")
    logger.info("")
    
    # Load data
    (X_train, labels_train), (X_val, labels_val), (X_test, labels_test) = load_nih_multimodal_data()
    
    # Create datasets
    transform = get_transforms(64, augment=False, is_training=False)
    
    train_dataset = MultimodalDataset(X_train, labels_train, labels_train, transform=transform)
    val_dataset = MultimodalDataset(X_val, labels_val, labels_val, transform=transform, vocab=train_dataset.vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multimodal)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multimodal)
    
    # Train 3 models
    results = {}
    
    # 1. Image only
    model_img = ImageOnlyModel(num_classes=15).to(args.device)
    results['image_only'] = train_model(model_img, train_loader, val_loader, args.epochs, args.device, 'multimodal_image_only')

    # 2. Texte only
    model_txt = TextOnlyModel(num_classes=15).to(args.device)
    results['text_only'] = train_model(model_txt, train_loader, val_loader, args.epochs, args.device, 'multimodal_text_only')

    # 3. Multimodal
    model_multi = MultimodalFusionModel(num_classes=15, fusion_type=args.fusion).to(args.device)
    results['multimodal'] = train_model(model_multi, train_loader, val_loader, args.epochs, args.device, f'multimodal_{args.fusion}_fusion')
    
    # Save results
    results_path = "models/multimodal_comparison.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPARAISON DES MODELES")
    logger.info("=" * 60)
    logger.info(f"Image seule : Val Loss = {results['image_only']:.4f}")
    logger.info(f"Multimodal  : Val Loss = {results['multimodal']:.4f}")
    logger.info("")
    logger.info(f"Résultats sauvegardés : {results_path}")


if __name__ == "__main__":
    main()
