"""
Utilitaires pour charger et gérer la configuration
"""
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Charger le fichier de configuration YAML
    
    Args:
        config_path: Chemin vers le fichier config.yaml
        
    Returns:
        Dictionnaire de configuration
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Fichier de config non trouvé : {config_path}")
        return get_default_config()
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration chargée depuis {config_path}")
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Configuration par défaut
    
    Returns:
        Dictionnaire de configuration par défaut
    """
    return {
        'data': {
            'dataset': 'chestmnist',
            'image_size': 64,
            'num_classes': 14,
            'batch_size': 32,
            'num_workers': 4
        },
        'model': {
            'name': 'cnn_simple',
            'dropout_rate': 0.3,
            'latent_dim': 64
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'epochs': 10,
            'early_stopping_patience': 5,
            'device': 'cuda'
        },
        'mlflow': {
            'enabled': True,
            'experiment_name': 'ChestMNIST Classification',
            'tracking_uri': 'http://localhost:5000'
        }
    }


def save_config(config: Dict[str, Any], config_path: str = "config.yaml") -> None:
    """
    Sauvegarder la configuration dans un fichier YAML
    
    Args:
        config: Dictionnaire de configuration
        config_path: Chemin où sauvegarder
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Configuration sauvegardée dans {config_path}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusionner deux configurations (override écrase base)
    
    Args:
        base_config: Configuration de base
        override_config: Configuration à appliquer par dessus
        
    Returns:
        Configuration fusionnée
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


if __name__ == "__main__":
    # Test
    config = get_default_config()
    print(config)
    save_config(config, "test_config.yaml")
    loaded = load_config("test_config.yaml")
    print(loaded)
