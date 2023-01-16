import os.path as osp

import pandas as pd
from tqdm import tqdm
from utils import print_expected

from nleval.data import DisGeNET
from nleval.exception import IDNotExistError
from nleval.graph import OntologyGraph

data_root_dir = "datasets_disgenet"
DisGeNET(data_root_dir)  # download DisGeNET data

data_dir = osp.join(data_root_dir, "DisGeNET", "raw")
do_path = osp.join(data_dir, "mondo.obo")
ga_path = osp.join(data_dir, "all_gene_disease_associations.tsv")

g = OntologyGraph()
umls_to_mondo = g.read_obo(do_path, xref_prefix="UMLS_CUI")
df = pd.read_csv(ga_path, sep="\t")

for geneId, diseaseId in tqdm(df[["geneId", "diseaseId"]].values):
    for mondo in umls_to_mondo[diseaseId]:
        try:
            g._update_node_attr_partial(mondo, geneId)
        except IDNotExistError:
            pass
g._update_node_attr_finalize()

node_attr_sizes = [len(g.get_node_attr(i) or []) for i in g.node_ids]
avg_node_attr_size = sum(node_attr_sizes) / g.size
print(f"Average node attribute size (raw) = {avg_node_attr_size:.2f}")

g.propagate_node_attrs()
node_attr_sizes = [len(g.get_node_attr(i) or []) for i in g.node_ids]
avg_node_attr_size = sum(node_attr_sizes) / g.size
print(f"Average node attribute size (propagated) = {avg_node_attr_size:.2f}")

print_expected(
    "Average node attribute size (raw) = 34.87",
    "Average node attribute size (propagated) = 103.82",
)
