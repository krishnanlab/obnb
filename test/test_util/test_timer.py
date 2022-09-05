import unittest
from time import sleep

from nleval.util import timer


class TestTimer(unittest.TestCase):
    def test_timeout_success(self):
        with timer.Timeout(2):
            sleep(1)

    def test_timeout_failure(self):
        with timer.Timeout(1):
            self.assertRaises(TimeoutError, sleep, 2)


if __name__ == "__main__":
    unittest.main()
