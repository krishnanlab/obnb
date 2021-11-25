from NLEval.model.BaseModel import BaseModel
import numpy as np

__all__ = ["LP"]


class LP(BaseModel):
    def __init__(self, G):
        super(LP, self).__init__(G)

    def train(self, ID_ary, y):
        pos_idx_ary = self.G.IDmap[ID_ary][y]
        hotvecs_ary = np.zeros(self.G.size)
        hotvecs_ary[pos_idx_ary] = True
        self.coef_ = np.matmul(self.G.mat, hotvecs_ary)

    def decision(self, ID_ary):
        idx_ary = self.G.IDmap[ID_ary]
        decision_ary = self.coef_[idx_ary]
        return decision_ary
