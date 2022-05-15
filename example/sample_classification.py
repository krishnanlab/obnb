import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from NLEval import label
from NLEval.graph import MultiFeatureVec

LABEL_FP = "/mnt/research/compbio/krishnanlab/projects/data_onto_mapping/data/tissue/RealTextManualAnnotations.txt"
# EXP_DATA_FP = "/mnt/research/compbio/krishnanlab/data/GEO/2019-07-29_downloaded-files/2019-07-31.npz"
EXP_DATA_FP = (
    "/mnt/home/liurenmi/repo/ContextNet/data/sample/GEO_2019-07-31_filtered.npz"
)

# Load gene expression labels
label_dict = {}  # sample id -> sample label
with open(LABEL_FP) as f:
    for line in f:
        gsm, _, gsm_label = line.strip().split(",")
        label_dict[gsm] = gsm_label
lsc = label.LabelsetCollection.from_dict(label_dict)

# Load gene expression sample data
exp_data = np.load(EXP_DATA_FP)
mfvec = MultiFeatureVec.from_mat(
    exp_data["data"].T,
    ids=exp_data["geneID"].tolist(),
    fset_ids=exp_data["GSM"].tolist(),
)

print(f"Number of labelsets before filtering: {len(lsc.label_ids)}")
lsc.iapply(label.filters.EntityExistenceFilter(mfvec.feature_ids))
lsc.iapply(label.filters.LabelsetRangeFilterSize(min_val=10))
print(f"Number of labelsets after filtering: {len(lsc.label_ids)}")

mdl = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    multi_class="multinomial",
)
y = lsc.get_y(mfvec.feature_ids)
y_multiclass = (y * np.arange(y.shape[1])).sum(axis=1)

print("\nStart trianing")
for train_idx, test_idx in StratifiedKFold(n_splits=2).split(y, y_multiclass):
    X_train, X_test = mfvec.mat.T[train_idx], mfvec.mat.T[test_idx]
    y_train, y_test = y_multiclass[train_idx], y_multiclass[test_idx]
    mdl.fit(X_train, y_train)
    train_score = mdl.score(X_train, y_train)
    test_score = mdl.score(X_test, y_test)
    print(f"{train_score=:.3f}, {test_score=:.3f}")
