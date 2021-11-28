from NLEval.util import IDHandler
from NLEval.util import checkers


class BaseGraph:
    """Base Graph object that contains basic graph operations."""

    def __init__(self):
        super().__init__()
        self.idmap = IDHandler.IDmap()

    @property
    def idmap(self):
        return self._idmap

    @idmap.setter
    def idmap(self, idmap):
        checkers.checkType("idmap", IDHandler.IDmap, idmap)
        self._idmap = idmap

    def __contains__(self, graph):
        """Return true if contains the input graph."""
        for ID in graph.idmap:  # check if containes all IDs in input graph
            if ID not in self.idmap:
                return False
        for ID1 in graph.idmap:  # check if all connections match
            for ID2 in graph.idmap:
                if self.get_edge(ID1, ID2) != graph.get_edge(ID1, ID2):
                    return False
        return True

    def __eq__(self, graph):
        """Return true if input graph is identical to self.

        For example, same set of IDs with same connections.

        """
        if self.idmap == graph.idmap:
            return graph in self
        return False

    @property
    def size(self):
        """int: number of nodes in graph."""
        return self.idmap.size

    def isempty(self):
        """bool: true if graph is empty, indicated by empty idmap."""
        return not self.size
