import argparse
import warnings

import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from obnb.dataset import OpenBiomedNetBenchDGL
from obnb.metric import log2_auprc_prior
from obnb.util.logger import log

# Ignore TypedStorage warnings https://github.com/dmlc/dgl/pull/5656
warnings.simplefilter("ignore")


class GCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hid_dim: int, num_layers: int):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(dglnn.GraphConv(in_dim, hid_dim, activation=F.relu))

        for _ in range(num_layers - 2):
            self.layers.append(dglnn.GraphConv(hid_dim, hid_dim, activation=F.relu))

        self.layers.append(dglnn.GraphConv(hid_dim, out_dim))

    def forward(self, g):
        h = g.ndata["feat"]
        for layer in self.layers:
            h = layer(g, h)
        return h


def parse_args():
    parser = argparse.ArgumentParser()
    # Dataset options
    parser.add_argument("--root", type=str, default="../datasets")
    parser.add_argument("--network", type=str, default="BioPlex")
    parser.add_argument("--label", type=str, default="GOBP")
    # Model options
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=128)
    # Training options
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=100)
    return parser.parse_args()


def train_epoch(gnn, g, optimizer):
    gnn.train()
    pred = gnn(g)
    mask = g.ndata["train_mask"][:, 0]
    loss = F.binary_cross_entropy_with_logits(pred[mask], g.ndata["label"][mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test_epoch(gnn, g):
    gnn.eval()
    pred = gnn(g)

    scores = []
    for split in ["train", "val", "test"]:
        mask = g.ndata[f"{split}_mask"][:, 0]
        scores.append(log2_auprc_prior(g.ndata["label"][mask], pred[mask]))

    return scores


def main():
    args = parse_args()

    # Set up data
    dataset = OpenBiomedNetBenchDGL(args.root, network=args.network, label=args.label)
    dglgraph = dataset[0]
    print(f"Graph: {dglgraph}")

    # Set up model and optimizer
    gnn = GCN(
        dglgraph.ndata["feat"].shape[1],
        dglgraph.ndata["label"].shape[1],
        args.hidden_channels,
        args.num_layers,
    )
    print(f"GNN: {gnn}")
    optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr, weight_decay=args.wd)

    # Main training loop
    best_epoch = 0
    best_val_apop = best_test_apop = -np.inf
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(gnn, dglgraph, optimizer)
        train_apop, val_apop, test_apop = test_epoch(gnn, dglgraph)

        if val_apop > best_val_apop:
            best_epoch, best_val_apop, best_test_apop = epoch, val_apop, test_apop

        log(
            Epoch=epoch,
            Loss=loss,
            Train=train_apop,
            Val=val_apop,
            Test=test_apop,
            BestEpoch=best_epoch,
            BestVal=best_val_apop,
            BestTest=best_test_apop,
        )


if __name__ == "__main__":
    main()
