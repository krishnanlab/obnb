import pytest

from nleval.config import MAX_DOWNLOAD_RETRIES
from nleval.exception import DataNotFoundError, ExceededMaxNumRetries
from nleval.util.download import download_unzip, get_data_url


def test_download(caplog, tmpdir, requests_mock, subtests):
    url = get_data_url(version="nledata-v1.0-test", name="something")
    print(f"{tmpdir=}\n{url=}")

    with subtests.test(404):
        requests_mock.get(url, status_code=404)
        reason = f"{url} is unavailable, try using a more recent data version"
        with pytest.raises(DataNotFoundError, match=reason):
            download_unzip(url, tmpdir)

    with subtests.test(429):
        requests_mock.get(url, headers={"Retry-after": "0"}, status_code=429)
        reason = f"Max number of retries exceeded {MAX_DOWNLOAD_RETRIES=}"
        with pytest.raises(ExceededMaxNumRetries, match=reason):
            download_unzip(url, tmpdir)
