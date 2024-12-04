import numpy as np

from obnb.alltypes import Callable, Dict, LogLevel, Optional
from obnb.model_trainer.base import StandardTrainer


class LabelPropagationTrainer(StandardTrainer):
    """Label propagation trainer."""

    def __init__(
        self,
        metrics: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], float]]] = None,
        train_on="train",
        log_level: LogLevel = "WARNING",
        log_path: Optional[str] = None,
    ):
        """Initialize LabelPropagationTrainer."""
        super().__init__(
            metrics,
            train_on=train_on,
            log_level=log_level,
            log_path=log_path,
        )

    @staticmethod
    def _model_predict(model, x, mask):
        return model.predictions[mask]

    @staticmethod
    def _model_train(model, g, x, y, mask):
        model.fit(g, y * mask)
