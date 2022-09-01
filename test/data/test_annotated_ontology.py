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
    config_path = osp.join(datadir, "processed", ".cache", hexhash, "config.yaml")
    spy = mocker.spy(
        NLEval.data.annotated_ontology.disgenet.DisGeNet,
        "apply_transform",
    )
    transform_called = 0

    with subtests.test("Normal download"):
        DisGeNet(tmpdir)
        assert osp.isdir(datadir)
        assert osp.isdir(osp.join(datadir, "processed"))
        assert osp.isdir(osp.join(datadir, "raw"))
        assert osp.isdir(osp.join(datadir, "info"))
        transform_called += 1  # called due to pre-transform
        assert spy.call_count == transform_called

    with subtests.test("Download then transform without saving"):
        DisGeNet(tmpdir, transform=filter_, cache_transform=False)
        transform_called += 1  # called due to transform
        assert spy.call_count == transform_called
        assert not osp.isfile(config_path)  # did not save cache

    with subtests.test("Download then transform and save"):
        DisGeNet(tmpdir, transform=filter_)
        transform_called += 1  # called due to transform
        assert spy.call_count == transform_called
        assert osp.isfile(config_path)

    with subtests.test("Load transformed data from cache"):
        DisGeNet(tmpdir, transform=filter_)
        transform_called += 0  # not called since found in cache
        assert spy.call_count == transform_called

    with subtests.test("Forced retransform due to modified config"):
        with open(config_path, "w") as f:
            f.write("")
        DisGeNet(tmpdir, transform=filter_)
        transform_called += 1  # called due to mismatched config
        assert spy.call_count == transform_called

    with subtests.test("Cannot set pre-transform for archived data"):
        with pytest.raises(ValueError):
            DisGeNet(tmpdir, pre_transform=filter_, version="nledata-v1.0-test")
