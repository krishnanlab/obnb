import logging

import pytest


@pytest.mark.mediumruns
@pytest.mark.xfail
def test_alevinfry(tmpdir):
    from nleval.data.experimental import AlevinFry

    logging.info(f"{tmpdir=}")

    af = AlevinFry(tmpdir, dataset_id=1)

    for metadata_key in AlevinFry.METADATA_KEYWORDS:
        assert af.metadata[metadata_key] is not None

    # Check feature matrix shape
    assert af.mat.shape == (10186, 36601)

    # Check that we can get item
    af[af.ids[:10]]
