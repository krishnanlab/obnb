import os.path as osp
import shutil
import subprocess
import tempfile

import numpy as np
from sklearn.linear_model import LogisticRegression
from utils import load_data

from nleval.feature import FeatureVec, MultiFeatureVec
from nleval.label.filters import LabelsetRangeFilterSplit
from nleval.label.split import RatioPartition
from nleval.metric import auroc
from nleval.model_trainer import (
    MultiSupervisedLearningTrainer,
    SupervisedLearningTrainer,
)

raise NotImplementedError("NEED TO FIX")
TEMP_DIR = tempfile.mkdtemp()
NETWORK = "STRING-EXP"
GRAPH_FP = osp.join(osp.pardir, "data", "networks", f"{NETWORK}.edg")
PECANPY_ARGS = [
    "pecanpy",
    "--input",
    GRAPH_FP,
    "--workers",
    "6",
    "--mode",
    "PreComp",
]

# Start embedding processes in the background
print(f"Start generating embeddings and saving to: {TEMP_DIR}")
qs = [0.01, 0.1, 1, 10, 100]
processes = []
for q in qs:
    fp = osp.join(TEMP_DIR, f"{NETWORK}_q={q}.emd")
    args = PECANPY_ARGS + ["--q", str(q), "--output", fp]
    processes.append(subprocess.Popen(args))

# Load dataset
g, lsc = load_data(NETWORK, "KEGGBP", sparse=True)

# 3/2 train/test split using genes with higher PubMed Count for training
splitter = RatioPartition(0.6, 0.2, 0.2, ascending=False)
lsc.iapply(LabelsetRangeFilterSplit(10, splitter))
print(f"Number of labelsets after split filtering: {len(lsc.label_ids)}")

# Select model
mdl = LogisticRegression(penalty="l2", solver="lbfgs", n_jobs=1)
metrics = {"auroc": auroc}

# Wait until all embeddings are generated to proceed
for process in processes:
    process.wait()

fvecs = []
for q in qs:
    fp = osp.join(TEMP_DIR, f"{NETWORK}_q={q}.emd")
    fvecs.append(FeatureVec.from_emd(fp))
    fvecs[-1].align_to_idmap(g.idmap)

mats = [fvec.mat for fvec in fvecs]
mfvec = MultiFeatureVec.from_mats(mats, g.idmap, fset_ids=[f"{q=}" for q in qs])

# Train with hyperparameter selection using validation set
trainer = MultiSupervisedLearningTrainer(metrics, mfvec, log_level="INFO")
scores = []
for label_id in lsc.label_ids:
    y, masks = lsc.split(
        splitter,
        target_ids=g.node_ids,
        labelset_name=label_id,
        consider_negative=True,
    )
    results = trainer.train(mdl, y, masks)
    scores.append(results["test_auroc"])
    print(
        f"Train: {results['train_auroc']:.4f}\t"
        f"Valid: {results['val_auroc']:.4f}\t"
        f"Test: {results['test_auroc']:.4f}\t{label_id}",
    )
print(f"Average test score = {np.mean(scores):.4f}, std = {np.std(scores):.4f}\n")

# No hyperparameter selection
for q, fvec in zip(qs, fvecs):
    trainer = SupervisedLearningTrainer(metrics, fvec)
    scores = []
    for label_id in lsc.label_ids:
        y, masks = lsc.split(
            splitter,
            target_ids=g.node_ids,
            labelset_name=label_id,
            consider_negative=True,
        )
        results = trainer.train(mdl, y, masks)
        scores.append(results["test_auroc"])
    print(
        f"Average test score ({q=}) = {np.mean(scores):.4f}, "
        f"std = {np.std(scores):.4f}",
    )

# Remove temp files after done
print(f"\nFinished testing, removing temporary files in {TEMP_DIR}")
shutil.rmtree(TEMP_DIR)
