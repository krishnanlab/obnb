import os.path as osp

import pandas as pd
from NLEval.graph import OntologyGraph
from NLEval.util.exceptions import IDNotExistError
from tqdm import tqdm


data_dir = osp.join(osp.pardir, "data")
do_path = osp.join(data_dir, "ontologies", "doid.obo")
ga_path = osp.join(data_dir, "annotations", "all_gene_disease_associations.tsv")

g = OntologyGraph()
umls_to_doid = g.read_obo(do_path, xref_prefix="UMLS_CUI")
df = pd.read_csv(ga_path, sep="\t")

for geneId, diseaseId in tqdm(df[["geneId", "diseaseId"]].values):
    for doid in umls_to_doid[diseaseId]:
        try:
            g._update_node_attr_partial(doid, geneId)
        except IDNotExistError:
            pass
g._update_node_attr_finalize()

node_attr_sizes = [len(g.get_node_attr(i) or []) for i in g.node_ids]
avg_node_attr_size = sum(node_attr_sizes) / g.size
print(f"Average node attribute size (raw) = {avg_node_attr_size:.2f}")

g.complete_node_attrs()
node_attr_sizes = [len(g.get_node_attr(i) or []) for i in g.node_ids]
avg_node_attr_size = sum(node_attr_sizes) / g.size
print(f"Average node attribute size (propagated) = {avg_node_attr_size:.2f}")

print(
    """
Expected outcome
--------------------------------------------------------------------------------
Average node attribute size (raw) = 43.42
Average node attribute size (propagated) = 131.49
--------------------------------------------------------------------------------
""",
)
