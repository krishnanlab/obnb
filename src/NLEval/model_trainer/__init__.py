"""Unified framework for training different types of models."""
from NLEval.model_trainer.label_propagation import LabelPropagationTrainer
from NLEval.model_trainer.supervised_learning import (
    MultiSupervisedLearningTrainer,
    SupervisedLearningTrainer,
)

__all__ = [
    "LabelPropagationTrainer",
    "MultiSupervisedLearningTrainer",
    "SupervisedLearningTrainer",
]
