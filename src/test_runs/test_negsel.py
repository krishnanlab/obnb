from sys import path
path.append('../src/')
from NLEval.label import LabelsetCollection, Filter

# specify p-val threshold
p_thresh = 0.05

# construct LabelsetCollection object from KEGGBP.gmt
lsc_orig = LabelsetCollection.SplitLSC.from_gmt('../data/labels/KEGGBP.gmt')
# make a copy for comparison, `apply` is inplace
lsc = lsc_orig.copy()
# create filer
f = Filter.NegativeFilterHypergeom(p_thresh)

# apply negative selection filter
lsc.apply(f)

print("p-val threshold = %.2f" % p_thresh)
# compare size of negative before and after filter. Size should shrink after
print("{:<62} {:<8} {:<8}".format("Term", "Original", "Filtered"))
for ID in lsc.labelIDlst:
    print("{:<62} {:>8d} {:>8d}".format(ID, len(lsc_orig.getNegative(ID)), 
                                        len(lsc.getNegative(ID))))
 