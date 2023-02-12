import os.path as osp

import pytest
import yaml

import nleval
from nleval.data.annotated_ontology import (
    GO,
    GOBP,
    GOCC,
    GOMF,
    DISEASES_KnowledgeFiltered,
    DisGeNET,
)
from nleval.label.filters import LabelsetRangeFilterSize
from nleval.util.path import hexdigest


@pytest.mark.mediumruns
def test_disgenet(tmpdir, mocker, subtests):
    datadir = osp.join(tmpdir, "DisGeNET")
    filter_ = LabelsetRangeFilterSize(min_val=100, max_val=200)
    hexhash = hexdigest(yaml.dump(filter_.to_config()))
    config_path = osp.join(datadir, "processed", ".cache", hexhash, "config.yaml")
    spy = mocker.spy(
        nleval.data.annotated_ontology.disgenet.DisGeNET,
        "apply_transform",
    )
    transform_called = 0

    with subtests.test("Normal download"):
        lsc = DisGeNET(tmpdir)
        assert len(lsc.label_ids) > 0
        assert osp.isdir(datadir)
        assert osp.isdir(osp.join(tmpdir, "DisGeNETAnnotation"))
        assert osp.isdir(osp.join(tmpdir, "MondoDiseaseOntology"))
        assert osp.isdir(osp.join(datadir, "processed"))
        assert osp.isdir(osp.join(datadir, "raw"))
        assert osp.isdir(osp.join(datadir, "info"))
        transform_called += 1  # called due to pre-transform
        assert spy.call_count == transform_called

    with subtests.test("Download then transform without saving"):
        DisGeNET(tmpdir, transform=filter_, cache_transform=False)
        transform_called += 1  # called due to transform
        assert spy.call_count == transform_called
        assert not osp.isfile(config_path)  # did not save cache

    with subtests.test("Download then transform and save"):
        DisGeNET(tmpdir, transform=filter_)
        transform_called += 1  # called due to transform
        assert spy.call_count == transform_called
        assert osp.isfile(config_path)

    with subtests.test("Load transformed data from cache"):
        DisGeNET(tmpdir, transform=filter_)
        transform_called += 0  # not called since found in cache
        assert spy.call_count == transform_called

    with subtests.test("Forced retransform due to modified config"):
        with open(config_path, "w") as f:
            f.write("")
        DisGeNET(tmpdir, transform=filter_)
        transform_called += 1  # called due to mismatched config
        assert spy.call_count == transform_called

    # XXX: to be decided whether or not to allow pre-transform archived data
    # with subtests.test("Cannot set pre-transform for archived data"):
    #     with pytest.raises(ValueError):
    #         DisGeNET(tmpdir, pre_transform=filter_, version="nledata-v1.0-test")


@pytest.mark.mediumruns
def test_go(tmpdir, mocker, subtests):
    with subtests.test("GO"):
        lsc = GO(tmpdir)
        assert len(lsc.label_ids) > 0

        assert osp.isdir(osp.join(tmpdir, "GO"))
        assert osp.isdir(osp.join(tmpdir, "GO", "processed"))
        assert osp.isdir(osp.join(tmpdir, "GO", "raw"))
        assert osp.isdir(osp.join(tmpdir, "GO", "info"))

        assert osp.isdir(osp.join(tmpdir, "GeneOntologyAnnotation"))
        assert osp.isdir(osp.join(tmpdir, "GeneOntology"))

    with subtests.test("GO"):
        lsc = GOBP(tmpdir)
        assert len(lsc.label_ids) > 0
        assert osp.isdir(osp.join(tmpdir, "GOBP"))

    with subtests.test("GO"):
        lsc = GOCC(tmpdir)
        assert len(lsc.label_ids) > 0
        assert osp.isdir(osp.join(tmpdir, "GOCC"))

    with subtests.test("GO"):
        lsc = GOMF(tmpdir)
        assert len(lsc.label_ids) > 0
        assert osp.isdir(osp.join(tmpdir, "GOMF"))


@pytest.mark.mediumruns
def test_diseases(tmpdir, mocker, subtests):
    datadir = osp.join(tmpdir, "DISEASES_KnowledgeFiltered")

    spy_ann_download = mocker.spy(
        nleval.data.annotation.diseases.DISEASESAnnotation,
        "download",
    )
    spy_ont_download = mocker.spy(
        nleval.data.ontology.mondo.MondoDiseaseOntology,
        "download",
    )
    spy_annont_download = mocker.spy(
        nleval.data.annotated_ontology.diseases.DISEASES_KnowledgeFiltered,
        "download",
    )

    with subtests.test("First download"):
        lsc = DISEASES_KnowledgeFiltered(tmpdir)
        assert len(lsc.label_ids) > 0

        assert osp.isdir(datadir)
        assert osp.isdir(osp.join(datadir, "processed"))
        assert osp.isdir(osp.join(datadir, "raw"))
        assert osp.isdir(osp.join(datadir, "info"))

        assert osp.isdir(osp.join(tmpdir, "DISEASESAnnotation"))
        assert osp.isdir(osp.join(tmpdir, "MondoDiseaseOntology"))

        # Downloaded annotation and ontology
        assert spy_ann_download.call_count == 1
        assert spy_ont_download.call_count == 1
        # Use the downloaded annotation and ontology
        assert spy_annont_download.call_count == 0

    with subtests.test("Load downloaded"):
        lsc = DISEASES_KnowledgeFiltered(tmpdir)
        assert len(lsc.label_ids) > 0

        # Directly use the previously downloaded data
        assert spy_ann_download.call_count == 1
        assert spy_ont_download.call_count == 1
        # Use the downloaded annotation and ontology
        assert spy_annont_download.call_count == 0

    with subtests.test("Redownload"):
        lsc = DISEASES_KnowledgeFiltered(tmpdir, redownload=True)
        assert len(lsc.label_ids) > 0

        # Forced redownload annotation and ontology
        assert spy_ann_download.call_count == 2
        assert spy_ont_download.call_count == 2
        # Use the downloaded annotation and ontology
        # NOTE: annotated ontology download should never be incremented as
        # the redownload kwarg is never passed to the annotated ontology object
        assert spy_annont_download.call_count == 0
