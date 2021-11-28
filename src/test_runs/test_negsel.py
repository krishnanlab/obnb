from sys import path

path.append("../")
from NLEval.label import LabelsetCollection, Filter

# specify p-val threshold
p_thresh = 0.05

# construct LabelsetCollection object from KEGGBP.gmt
lsc_orig = LabelsetCollection.SplitLSC.from_gmt("../../data/labels/KEGGBP.gmt")
# make a copy for comparison, `apply` is inplace
lsc = lsc_orig.copy()
# create filer
f = Filter.NegativeFilterHypergeom(p_thresh)

# apply negative selection filter
lsc.apply(f, inplace=True)

print(f"p-val threshold = {p_thresh:.2f}")
# compare size of negative before and after filter. Size should shrink after
print(f"{'Term':<62} {'Original':<8} {'Filtered':<8}")
for ID in lsc.label_ids:
    print(
        f"{ID:<62} {len(lsc_orig.getNegative(ID)):>8d} "
        f"{len(lsc.getNegative(ID)):>8d}"
    )
