import os.path as osp

from NLEval.label import labelset_collection
from NLEval.label import labelset_filter

DATA_DIR = osp.join(osp.pardir, "data")
LABEL_FP = osp.join(DATA_DIR, "labels", "KEGGBP.gmt")

# specify p-val threshold
p_thresh = 0.05

# construct labelset_collection object from KEGGBP.gmt
lsc_orig = labelset_collection.SplitLSC.from_gmt(LABEL_FP)

# apply negative selection filter
lsc = lsc_orig.apply(labelset_filter.NegativeFilterHypergeom(p_thresh))

print(f"p-val threshold = {p_thresh:.2f}")
print(f"Compring the number of negatives before and after filtering")
print(f"{'Term':<62} {'Original':<8} {'Filtered':<8} {'Diff':<8}")
diff_list = []
for ID in lsc.label_ids:
    orig_size = len(lsc_orig.get_negative(ID))
    filtered_size = len(lsc.get_negative(ID))
    diff_list.append(filtered_size - orig_size)
    print(
        f"{ID:<62} {orig_size:>8d} {filtered_size:>8d} {diff_list[-1]:>8}",
    )

print(f"Average diff = {sum(diff_list) / len(diff_list):.2f}")

print(
    """
Expected outcome
--------------------------------------------------------------------------------
Average diff = -543.58
--------------------------------------------------------------------------------
""",
)
