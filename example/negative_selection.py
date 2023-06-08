from obnb.data import GOBP
from obnb.label import filters
from utils import print_expected

# specify p-val threshold
p_thresh = 0.05

# construct labelset_collection object from KEGGBP.gmt
lsc_orig = GOBP(
    "datasets",
    transform=filters.LabelsetRangeFilterSize(min_val=90, max_val=100),
)

# apply negative selection filter
lsc = lsc_orig.apply(filters.NegativeGeneratorHypergeom(p_thresh, log_level="DEBUG"))

print(f"p-val threshold = {p_thresh:.2f}")
print("Compring the number of negatives before and after filtering")
print(f"{'Term':<62} {'Original':<8} {'Filtered':<8} {'Diff':<8}")
diff_list = []
for label_id in lsc.label_ids:
    orig_size = len(lsc_orig.get_negative(label_id))
    filtered_size = len(lsc.get_negative(label_id))
    diff_list.append(filtered_size - orig_size)
    print(
        f"{label_id:<62} {orig_size:>8d} {filtered_size:>8d} {diff_list[-1]:>8}",
    )

print(f"Average diff = {sum(diff_list) / len(diff_list):.2f}")
print_expected("Average diff = -192.03")
