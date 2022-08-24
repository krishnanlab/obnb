import warnings

import numpy as np

from NLEval.graph.base import BaseGraph
from NLEval.util.checkers import checkValueNonnegative, checkValuePositive
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

        Raises:
            ValueError: If :attr:`tol` is not non-negative, or :attr:`max_iter`
                is not positive.

        """
        self.tol = tol
        self.max_iter = max_iter
        self.warn = warn

    @property
    def tol(self) -> float:
        """Error tolerance."""
        return self._tol

    @tol.setter
    def tol(self, val: float):
        """Setter for :attr:`tol`."""
        checkValueNonnegative("tol", val)
        self._tol = val

    @property
    def max_iter(self) -> int:
        """Max iteration."""
        return self._max_iter

    @max_iter.setter
    def max_iter(self, val: int):
        """Setter for :attr:`max_iter`."""
        checkValuePositive("max_iter", val)
        self._max_iter = val

    def __call__(self, graph: BaseGraph, seed: np.ndarray) -> np.ndarray:
        """Propagate the seed information over the network.

        Args:
            graph (:obj:`BaseGraph`): The underlying graph object used to
                propagate the seed node information.
            seed (`obj:`np.ndarray`): A 1-dimensional numpy array with the
                size of the number of nodes in the graph.

        """
        y_pred_new = y_pred = graph.propagate(seed)
        converged = False
        # TODO: restart
        for _ in range(self.max_iter - 1):
            y_pred_new = graph.propagate(y_pred)
            norm = np.linalg.norm(y_pred_new - y_pred)
            y_pred_new, y_pred = y_pred, y_pred_new  # switch pointers
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
        super().__init__(tol=0.0, max_iter=k, warn=False)


class OneHopPropagation(KHopPropagation):
    """Propagate seed node information exactly 1 time."""

    def __init__(self):
        """Initialize the OneHopPropagation."""
        super().__init__(1)
