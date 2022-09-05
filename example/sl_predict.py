from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auroc
from utils import load_data

from nleval import Dataset
from nleval.label.split import AllHoldout
from nleval.model_trainer import SupervisedLearningTrainer

i = 24  # index of labelset
k = 50  # numbers of top genes to display

# load graph and labelset collection
g, lsc, _ = load_data()

# initialize model
mdl = LogisticRegression(penalty="l2", solver="liblinear")

# diplay choice of labelsets
for j, m in enumerate(lsc.label_ids):
    print(f"Index: {j:>4d}, Labelset size: {len(lsc.get_labelset(m)):>4d}, {m}")
print("")

# get label_id
label_id = lsc.label_ids[i]
print(f"{label_id}\t{lsc.get_info(label_id)}")

# train and get genome wide prediction
y, masks = lsc.split(
    AllHoldout(),
    target_ids=g.node_ids,
    labelset_name=label_id,
    mask_names=("train",),
)
dataset = Dataset(feature=g.to_feature(), y=y, masks=masks)

metrics = {"auroc": auroc}
trainer = SupervisedLearningTrainer(metrics)
trainer.train(mdl, dataset)
score_dict = {i: j for i, j in zip(g.node_ids, mdl.decision_function(g.mat))}

# print top ranked genes and its intersection with known ones
top_list = sorted(score_dict, key=score_dict.get, reverse=True)[:k]
intersection = sorted(set(top_list) & lsc.get_labelset(label_id))

print(f"Top {k} genes: {repr(top_list)}")
print(f"{len(intersection)} known genes in top {k}: {repr(intersection)}")
