"""
Preprocessing script to download ChestMNIST and resize to 28x28.
Run once to prepare data, then notebooks load from processed/.
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image
import medmnist
from medmnist import ChestMNIST

def preprocess_chestmnist(data_dir='../../data', target_size=28, force=False):
    """
    Download ChestMNIST and resize to target_size.
    Saves to data/processed/ as .npz files.
    
    Args:
        data_dir: parent directory for data/
        target_size: image size (default 28x28)
        force: re-download even if processed/ exists
    """
    processed_dir = Path(data_dir) / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    output_file = processed_dir / f'chestmnist_{target_size}x{target_size}.npz'
    if output_file.exists() and not force:
        print(f"✅ Already preprocessed: {output_file}")
        return
    
    print(f"Downloading ChestMNIST...")
    download_dir = Path(data_dir) / 'raw'
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Download all splits
    train_dataset = ChestMNIST(split='train', download=True, root=str(download_dir))
    val_dataset = ChestMNIST(split='val', download=True, root=str(download_dir))
    test_dataset = ChestMNIST(split='test', download=True, root=str(download_dir))
    
    # Process each split
    print(f"Resizing to {target_size}x{target_size}...")
    
    train_images = [np.array(Image.fromarray(img).resize((target_size, target_size))) 
                    for img, _ in train_dataset]
    train_labels = [label for _, label in train_dataset]
    
    val_images = [np.array(Image.fromarray(img).resize((target_size, target_size))) 
                  for img, _ in val_dataset]
    val_labels = [label for _, label in val_dataset]
    
    test_images = [np.array(Image.fromarray(img).resize((target_size, target_size))) 
                   for img, _ in test_dataset]
    test_labels = [label for _, label in test_dataset]
    
    X_train = np.array(train_images, dtype=np.uint8)
    y_train = np.array(train_labels, dtype=np.int64)
    X_val = np.array(val_images, dtype=np.uint8)
    y_val = np.array(val_labels, dtype=np.int64)
    X_test = np.array(test_images, dtype=np.uint8)
    y_test = np.array(test_labels, dtype=np.int64)
    
    # Save
    print(f"Saving to {output_file}...")
    np.savez(output_file,
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)
    
    print(f"✅ Done! Shapes:")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"\nOutput: {output_file}")

if __name__ == '__main__':
    preprocess_chestmnist(data_dir='../../data', target_size=28, force=False)
