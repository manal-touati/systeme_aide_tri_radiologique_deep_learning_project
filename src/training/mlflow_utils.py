"""
Utilitaires MLflow pour tracking des expériences
"""
import mlflow
import mlflow.pytorch
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def setup_mlflow(
    experiment_name: str = "ChestMNIST Classification",
    tracking_uri: str = "http://localhost:5000"
) -> None:
    """
    Configure MLflow
    
    Args:
        experiment_name: Nom de l'expérience
        tracking_uri: URI du serveur MLflow
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow initialisé : {experiment_name} @ {tracking_uri}")


def start_run(run_name: Optional[str] = None) -> mlflow.ActiveRun:
    """
    Démarrer un run MLflow
    
    Args:
        run_name: Nom du run
        
    Returns:
        Active run context
    """
    run = mlflow.start_run(run_name=run_name)
    logger.info(f"Run démarré : {run_name or run.info.run_id}")
    return run


def log_params(params: Dict[str, Any]) -> None:
    """
    Logger les hyperparamètres
    
    Args:
        params: Dictionnaire des paramètres
    """
    mlflow.log_params(params)
    logger.info(f"Paramètres loggés : {len(params)} items")


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Logger les métriques
    
    Args:
        metrics: Dictionnaire des métriques
        step: Step/epoch optionnel
    """
    mlflow.log_metrics(metrics, step=step)
    logger.info(f"Métriques loggées (step {step})")


def log_artifact(artifact_path: str, artifact_name: str = None) -> None:
    """
    Logger un artefact (fichier)
    
    Args:
        artifact_path: Chemin vers le fichier
        artifact_name: Nom de l'artefact dans MLflow
    """
    if Path(artifact_path).exists():
        mlflow.log_artifact(artifact_path, artifact_name)
        logger.info(f"Artefact loggé : {artifact_name or artifact_path}")
    else:
        logger.warning(f"Artefact non trouvé : {artifact_path}")


def log_model(model: torch.nn.Module, artifact_path: str = "model") -> None:
    """
    Logger un modèle PyTorch
    
    Args:
        model: Modèle PyTorch
        artifact_path: Chemin de l'artefact dans MLflow
    """
    mlflow.pytorch.log_model(model, artifact_path)
    logger.info(f"Modèle loggé : {artifact_path}")


def end_run() -> None:
    """Terminer le run MLflow"""
    mlflow.end_run()
    logger.info("Run terminé")


class MLflowLogger:
    """
    Contexte manager pour MLflow
    
    Usage:
        with MLflowLogger("experiment", "run_name") as logger:
            logger.log_params({"lr": 0.001})
            logger.log_metrics({"loss": 0.5})
    """
    
    def __init__(
        self,
        experiment_name: str = "Default",
        run_name: Optional[str] = None,
        tracking_uri: str = "http://localhost:5000"
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.run = None
    
    def __enter__(self):
        setup_mlflow(self.experiment_name, self.tracking_uri)
        self.run = start_run(self.run_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_run()
    
    def log_params(self, params: Dict[str, Any]):
        """Logger paramètres"""
        log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Logger métriques"""
        log_metrics(metrics, step)
    
    def log_artifact(self, path: str, name: str = None):
        """Logger artefact"""
        log_artifact(path, name)
    
    def log_model(self, model: torch.nn.Module, path: str = "model"):
        """Logger modèle"""
        log_model(model, path)


if __name__ == "__main__":
    # Test
    with MLflowLogger("test_experiment", "test_run") as ml:
        ml.log_params({"learning_rate": 0.001, "batch_size": 32})
        ml.log_metrics({"train_loss": 0.5, "val_loss": 0.6}, step=1)
