"""Utility for exploring NDEx CX data."""
import json

import ndex2
from requests import RequestException

from nleval.typing import Dict, List, Optional, Set


class CXExplorer:
    """Explore NDEx cx stream data."""

    def __init__(self, uuid: Optional[str] = None, path: Optional[str] = None):
        """Initialize CXExplorer.

        Args:
            uuid (str, optional): NDEx network UUID to download. If None, then
                do not download.
            path (str, optional): Path to cx file.

        Notes:
            Either specify the network ndex `uuid` to download the CX stream
            directly, or specify the `path` to the CX file. If neither is
            specified, then initialize an empty `CXExplorer`.

        Raises:
            ValueError: If both `uuid` and `path` are specified.

        """
        if (uuid is not None) and (path is not None):
            raise ValueError("Can not specify uuid and path at the same time.")
        elif uuid is not None:
            self.load_from_uuid(uuid)
        elif path is not None:
            self.load_from_file(path)

    def __getitem__(self, field: str):
        """Get field in the CX stream data."""
        return self._data[self._fields[field]][field]

    @classmethod
    def from_cx_stream(cls, cx_stream):
        """Construct directly from cx_stream."""
        cxe = cls(uuid=None)
        cxe.data = cx_stream
        return cxe

    @property
    def data(self) -> Dict:
        """CX stream data."""
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self._fields = {list(j)[0]: i for i, j in enumerate(self._data)}

    @property
    def fields(self) -> List[str]:
        """All available fields in the CX data."""
        return list(self._fields)

    def show_fields(self, full: bool = False):
        """Print fields with index, size, example, and unique keys."""
        for i, j in enumerate(self.fields):
            field = self[j]

            # Pad left with spaces to align with indentations
            toreplace = ("\n", "\n\t\t")
            example_str = pformat(field[0]).replace(*toreplace)

            print(f"[{i}] {j} (n={len(field):,})")
            print(f"\tExample:\n\t\t{example_str}")

            # Show unique keys and the associated unique value counts
            if full:
                keys = set(itertools.chain.from_iterable(map(list, field)))
                unique_counts = {x: len(self.unique(j, x)) for x in sorted(keys)}
                unique_counts_str = pformat(unique_counts).replace(*toreplace)
                print(f"\tUnique key value counts:\n\t\t{unique_counts_str}")

    def load_from_uuid(self, uuid: str):
        """Load CX stream data using the UUID."""
        client = ndex2.client.Ndex2()
        r = client.get_network_as_cx_stream(uuid)
        if not r.ok:
            raise RequestException(r)
        self.data = json.loads(r.content)

    def load_from_file(self, path: str):
        """Load CX stream from a CX file."""
        with open(path) as f:
            self.data = json.load(f)

    def unique(self, field: str, name: str) -> Set[str]:
        """Return unique elements in a field.

        Example:
            >>> cxe = CXExplorer(uuid)
            >>> cxe.unique("node", "n")  # unique nodes
            >>> cxe.unique("edges", "i")  # unique interactions

        """
        return {i[name] for i in self[field]}
