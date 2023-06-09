import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.nn as pygnn

from obnb.dataset import OpenBiomedNetBenchPyG
from obnb.metric import log2_auprc_prior
from obnb.util.logger import log


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


def train_epoch(gnn, data, optimizer):
    gnn.train()
    pred = gnn(data.x, data.edge_index)
    mask = data.train_mask[:, 0]
    loss = F.binary_cross_entropy_with_logits(pred[mask], data.y[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test_epoch(gnn, data):
    gnn.eval()
    pred = gnn(data.x, data.edge_index)

    scores = []
    for split in ["train", "val", "test"]:
        mask = getattr(data, f"{split}_mask")[:, 0]
        scores.append(log2_auprc_prior(data.y[mask], pred[mask]))

    return scores


def main():
    args = parse_args()

    # Set up data
    dataset = OpenBiomedNetBenchPyG(args.root, network=args.network, label=args.label)
    data = dataset[0]

    # Set up model and optimizer
    gnn = pygnn.GCN(
        in_channels=data.x.shape[1],
        out_channels=data.y.shape[1],
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
    )
    optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr, weight_decay=args.wd)

    # Main training loop
    best_epoch = 0
    best_val_apop = best_test_apop = -np.inf
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(gnn, data, optimizer)
        train_apop, val_apop, test_apop = test_epoch(gnn, data)

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
