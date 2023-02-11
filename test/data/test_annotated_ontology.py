import os.path as osp

import pytest
import yaml

import nleval
from nleval.data.annotated_ontology.disgenet import DisGeNET
from nleval.data.annotated_ontology.go import GO, GOBP, GOCC, GOMF
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
