from NLEval.label import labelset_collection
from NLEval.label import labelset_filter

# specify p-val threshold
p_thresh = 0.05

# construct labelset_collection object from KEGGBP.gmt
lsc_orig = labelset_collection.SplitLSC.from_gmt("../../data/labels/KEGGBP.gmt")
# make a copy for comparison, `apply` is inplace
lsc = lsc_orig.copy()
# create filer
f = labelset_filter.NegativeFilterHypergeom(p_thresh)

# apply negative selection filter
lsc.apply(f, inplace=True)

print(f"p-val threshold = {p_thresh:.2f}")
# compare size of negative before and after filter. Size should shrink after
print(f"{'Term':<62} {'Original':<8} {'Filtered':<8}")
for ID in lsc.label_ids:
    print(
        f"{ID:<62} {len(lsc_orig.get_negative(ID)):>8d} "
        f"{len(lsc.get_negative(ID)):>8d}",
    )
