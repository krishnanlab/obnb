"""Unified framework for training different types of models."""
from obnb.model_trainer.label_propagation import LabelPropagationTrainer
from obnb.model_trainer.supervised_learning import (
    MultiSupervisedLearningTrainer,
    SupervisedLearningTrainer,
)

__all__ = [
    "LabelPropagationTrainer",
    "MultiSupervisedLearningTrainer",
    "SupervisedLearningTrainer",
]
