import os.path as osp

import pytest

from nleval.data.ontology.gene_ontology import GeneOntology
from nleval.data.ontology.mondo import MondoDiseaseOntology


@pytest.mark.mediumruns
def test_gene_ontology(tmpdir):
    datadir = osp.join(tmpdir, "GeneOntology")

    data = GeneOntology(tmpdir)
    assert osp.isdir(datadir)
    assert osp.isdir(osp.join(datadir, "processed"))
    assert osp.isdir(osp.join(datadir, "raw"))
    assert osp.isdir(osp.join(datadir, "info"))

    assert data.data.num_nodes > 0
    assert data.data.num_edges > 0


@pytest.mark.mediumruns
def test_mondo_disease_ontology(tmpdir):
    datadir = osp.join(tmpdir, "MondoDiseaseOntology")

    data = MondoDiseaseOntology(tmpdir)
    assert osp.isdir(datadir)
    assert osp.isdir(osp.join(datadir, "processed"))
    assert osp.isdir(osp.join(datadir, "raw"))
    assert osp.isdir(osp.join(datadir, "info"))

    assert data.data.num_nodes > 0
    assert data.data.num_edges > 0
