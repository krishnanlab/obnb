import shutil
import tempfile
import unittest

from NLEval.util import converter


class TestMyGeneInfoConverter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        print(f"Created temporary directory {cls.tmpdir}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)
        print(f"Removed temporary directory {cls.tmpdir}")

    def test_basics(self):
        mgc = converter.MyGeneInfoConverter(
            use_cache=False,
            save_cache=False,
        )
        mgc.convert_map["a"] = "A"
        mgc.convert_map["b"] = "B"

        self.assertEqual(len(mgc), 2)
        self.assertEqual(mgc["a"], "A")
        self.assertEqual(mgc["c"], None)
        self.assertEqual(sorted(mgc), ["a", "b", "c"])

    def test_cache(self):
        mgc = converter.MyGeneInfoConverter(
            root=self.tmpdir,
            log_level="DEBUG",
        )
        mgc.convert_map["a"] = "A"
        mgc.convert_map["b"] = "B"
        mgc._save_cache()

        mgc2 = converter.MyGeneInfoConverter(
            root=self.tmpdir,
            log_level="DEBUG",
        )
        mgc2._load_cache()
        self.assertEqual(mgc.convert_map, mgc2.convert_map)
