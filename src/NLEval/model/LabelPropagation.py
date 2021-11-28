import numpy as np
from NLEval.model.BaseModel import BaseModel

__all__ = ["LP"]


class LP(BaseModel):
    def __init__(self, graph):
        super(LP, self).__init__(graph)

    def train(self, ID_ary, y):
        pos_idx_ary = self.graph.idmap[ID_ary][y]
        hotvecs_ary = np.zeros(self.graph.size)
        hotvecs_ary[pos_idx_ary] = True
        self.coef_ = np.matmul(self.graph.mat, hotvecs_ary)

    def decision(self, ID_ary):
        idx_ary = self.graph.idmap[ID_ary]
        decision_ary = self.coef_[idx_ary]
        return decision_ary
