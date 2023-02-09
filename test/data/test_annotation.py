import os.path as osp

import pandas as pd
import pytest

from nleval.data.annotation.disgenet import DisGeNET


@pytest.mark.mediumruns
def test_digenet(tmpdir, subtests):
    datadir = osp.join(tmpdir, "DisGeNET")

    # Normal download and formatting
    with subtests.test("Normal load"):
        data = DisGeNET(tmpdir)
        assert osp.isdir(datadir)
        assert osp.isdir(osp.join(datadir, "processed"))
        assert osp.isdir(osp.join(datadir, "raw"))
        assert osp.isdir(osp.join(datadir, "info"))

        # Check if columns are set to the correct name
        assert data.data.columns.tolist() == ["gene_id", "term_id"]
        # Check if values are prefixed correctly
        assert data.data.iloc[0, 0].startswith("ncbigene:")
        assert data.data.iloc[0, 1].startswith("umls:C")

    # Load full annotation data to be used for checking filtering later
    full_df = pd.read_csv(data.raw_file_path(0), sep="\t")
    full_df["gene_id"] = "ncbigene:" + full_df["geneId"].astype(str).values
    full_df["term_id"] = "umls:" + full_df["diseaseId"].astype(str).values
    index_cols = ["gene_id", "term_id"]
    full_df = full_df.set_index(index_cols)

    with subtests.test("Single filter"):
        data = DisGeNET(tmpdir, dsi_min=0.9, reprocess=True)
        df = data.data.set_index(index_cols)
        aligned_full_df = full_df.align(df, axis=0, join="right")[0]
        assert aligned_full_df["DSI"].min() >= 0.9

        data = DisGeNET(tmpdir, dpi_max=0.1, reprocess=True)
        df = data.data.set_index(index_cols)
        aligned_full_df = full_df.align(df, axis=0, join="right")[0]
        assert aligned_full_df["DPI"].max() <= 0.1

    with subtests.test("Multiple filterx"):
        settings = {"dsi_min": 0.6, "dpi_min": 0.6, "dsi_max": 0.9, "dpi_max": 0.9}
        data = DisGeNET(tmpdir, reprocess=True, **settings)
        df = data.data.set_index(index_cols)
        aligned_full_df = full_df.align(df, axis=0, join="right")[0]
        assert aligned_full_df["DSI"].min() >= 0.6
        assert aligned_full_df["DSI"].max() >= 0.6
        assert aligned_full_df["DPI"].min() <= 0.9
        assert aligned_full_df["DPI"].max() <= 0.9
