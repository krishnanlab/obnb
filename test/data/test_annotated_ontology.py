import logging
import os.path as osp

import pytest
import yaml

import NLEval
from NLEval.data.annotated_ontology.disgenet import DisGeNet
from NLEval.label.filters import LabelsetRangeFilterSize
from NLEval.util.path import hexdigest


@pytest.mark.mediumruns
def test_disgenet(tmpdir, caplog, mocker, subtests):
    datadir = osp.join(tmpdir, "DisGeNet")
    filter_ = LabelsetRangeFilterSize(min_val=100, max_val=200)
    hexhash = hexdigest(yaml.dump(filter_.to_config()))
    config_path = osp.join(datadir, "processed", hexhash, "config.yaml")
    spy = mocker.spy(NLEval.data.annotated_ontology.disgenet.DisGeNet, "transform")
    transform_called = 0

    with subtests.test("Normal download"):
        lsc = DisGeNet(tmpdir)
        assert osp.isdir(datadir)
        assert osp.isdir(osp.join(datadir, "processed"))
        assert osp.isdir(osp.join(datadir, "raw"))
        assert osp.isdir(osp.join(datadir, "info"))

    with subtests.test("Download then transform"):
        lsc = DisGeNet(tmpdir, transform=filter_)
        transform_called += 1
        assert spy.call_count == transform_called
        assert osp.isfile(config_path)

    with subtests.test("Load transformed data from cache"):
        lsc = DisGeNet(tmpdir, transform=filter_)
        assert spy.call_count == transform_called

    with subtests.test("Forced retransform due to modified config"):
        with open(config_path, "w") as f:
            f.write("")
        lsc = DisGeNet(tmpdir, transform=filter_)
        transform_called += 1
        assert spy.call_count == transform_called
