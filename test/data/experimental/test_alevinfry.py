import logging

import pytest

from NLEval.data.experimental import AlevinFry


@pytest.mark.mediumruns
def test_alevinfry(tmpdir):
    logging.info(f"{tmpdir=}")

    af = AlevinFry(tmpdir, dataset_id=1)

    for metadata_key in AlevinFry.METADATA_KEYWORDS:
        assert af.metadata[metadata_key] is not None

    # Check feature matrix shape
    assert af.mat.shape == (10620, 36601)

    # Check that we can get item
    af[af.node_ids[:10]]
