import warnings

import numpy as np
from NLEval.graph.BaseGraph import BaseGraph
from NLEval.util.exceptions import NotConvergedWarning


class IterativePropagation:
    """Iteratively propagate seed node information.

    This class of method propagates the initial (or seed) node information
    across the network iteratively until either convergence or the max allowed
    iteration is reached. It relies mainly on the ``propagate`` method from
    the graph objects to perform a single hop propagation.

    """

    def __init__(
        self,
        tol: float = 1e-6,
        max_iter: int = 200,
        warn: bool = True,
    ):
        """Initialize the IterativePropagation.

        Args:
            tol (float): Error tolerance for convergence (default: ``1e-6``)
            max_iter (int): Max iteration for propagation (default: ``200``)
            warn (bool): If set, print warning message if the propagated
                vector did not converge when the max number of iteration is
                reached (default: ``True``)

        """
        self.tol = tol  # TODO: check non neg
        self.max_iter = max_iter  # TODO: check >= 1
        self.warn = warn

    def __call__(self, graph: BaseGraph, seed: np.ndarray) -> np.ndarray:
        y_pred = graph.propagate(seed)
        y_pred_prev = y_pred.copy()
        converged = False
        for _ in range(self.max_iter - 1):
            y_pred = graph.propagate(y_pred_prev)
            norm = np.norm(y_pred - y_pred_prev)
            y_pred_prev, y_pred = y_pred, y_pred_prev  # switch pointers
            if norm < self.tol:
                converged = True
                break

        if not converged and self.warn:
            warnings.warn(
                f"Failed to converge within {self.max_iter} steps.",
                NotConvergedWarning,
            )

        return y_pred


class KHopPropagation(IterativePropagation):
    """Propagate seed node information exactly k times."""

    def __init__(self, k: int):
        """Initialize the KHopPropagation.

        Args:
            k (int): Number of hops to propagate.

        """
        super().__init__(tol=np.inf, max_iter=k, warn=False)


class OneHopPropagation(KHopPropagation):
    """Propagate seed node information exactly 1 time."""

    def __init__(self):
        """Initialize the OneHopPropagation."""
        super().__init__(1)
