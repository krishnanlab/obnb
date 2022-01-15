"""Unified framework for training different types of models."""
from .label_propagation import LabelPropagationTrainer
from .supervised_learning import MultiSupervisedLearningTrainer
from .supervised_learning import SupervisedLearningTrainer

__all__ = [
    "LabelPropagationTrainer",
    "MultiSupervisedLearningTrainer",
    "SupervisedLearningTrainer",
]
