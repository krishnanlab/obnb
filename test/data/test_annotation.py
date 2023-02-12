import os.path as osp

import pandas as pd
import pytest

from nleval.data.annotation.disgenet import DisGeNETAnnotation
from nleval.data.annotation.gene_ontology import GeneOntologyAnnotation


@pytest.mark.mediumruns
def test_digenet(tmpdir, subtests):
    datadir = osp.join(tmpdir, "DisGeNETAnnotation")

    # Normal download and formatting
    with subtests.test("Normal load"):
        data = DisGeNETAnnotation(tmpdir)
        assert osp.isdir(datadir)
        assert osp.isdir(osp.join(datadir, "processed"))
        assert osp.isdir(osp.join(datadir, "raw"))
        assert osp.isdir(osp.join(datadir, "info"))

        # Check if columns are set to the correct name
        assert data.data.columns.tolist() == ["gene_id", "term_id"]

    # Load full annotation data to be used for checking filtering later
    full_df = pd.read_csv(data.raw_file_path(0), sep="\t")
    full_df["gene_id"] = full_df["geneId"].astype(str)
    full_df["term_id"] = "UMLS:" + full_df["diseaseId"].astype(str).values
    index_cols = ["gene_id", "term_id"]
    full_df = full_df.set_index(index_cols)

    with subtests.test("Single filter"):
        data = DisGeNETAnnotation(tmpdir, dsi_min=0.9, reprocess=True)
        df = data.data.set_index(index_cols)
        aligned_full_df = full_df.align(df, axis=0, join="right")[0]
        assert aligned_full_df["DSI"].min() >= 0.9

        data = DisGeNETAnnotation(tmpdir, dpi_max=0.1, reprocess=True)
        df = data.data.set_index(index_cols)
        aligned_full_df = full_df.align(df, axis=0, join="right")[0]
        assert aligned_full_df["DPI"].max() <= 0.1

    with subtests.test("Multiple filterx"):
        settings = {"dsi_min": 0.6, "dpi_min": 0.6, "dsi_max": 0.9, "dpi_max": 0.9}
        data = DisGeNETAnnotation(tmpdir, reprocess=True, **settings)
        df = data.data.set_index(index_cols)
        aligned_full_df = full_df.align(df, axis=0, join="right")[0]
        assert aligned_full_df["DSI"].min() >= 0.6
        assert aligned_full_df["DSI"].max() >= 0.6
        assert aligned_full_df["DPI"].min() <= 0.9
        assert aligned_full_df["DPI"].max() <= 0.9


@pytest.mark.mediumruns
def test_gene_ontology(tmpdir, subtests):
    datadir = osp.join(tmpdir, "GeneOntologyAnnotation")

    data = GeneOntologyAnnotation(tmpdir)
    assert osp.isdir(datadir)
    assert osp.isdir(osp.join(datadir, "processed"))
    assert osp.isdir(osp.join(datadir, "raw"))
    assert osp.isdir(osp.join(datadir, "info"))

    # Check if columns are set to the correct name
    assert data.data.columns.tolist() == ["gene_id", "term_id"]
