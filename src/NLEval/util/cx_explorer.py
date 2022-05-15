"""Utility for exploring NDEx CX data."""
import json

import mygene
import ndex2
from requests import RequestException

from ..typing import Dict, List, Optional, Set


class CXExplorer:
    """Explore NDEx cx stream data."""

    def __init__(self, uuid: Optional[str]):
        """Initialize CXExplorer.

        Args:
            uuid (str, optional): NDEx network UUID to download. If None, then
                do not download.

        """
        if uuid:
            self.load_data(uuid)

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

    def load_data(self, uuid: str):
        """Load CX stream data using the UUID."""
        client = ndex2.client.Ndex2()
        r = client.get_network_as_cx_stream(uuid)
        if not r.ok:
            raise RequestException(r)
        self.data = json.loads(r.content)

    def unique(self, field: str, name: str) -> Set[str]:
        """Return unique elements in a field.

        Example:
            >>> cxe = CXExplorer(uuid)
            >>> cxe.unique("node", "n")  # unique nodes
            >>> cxe.unique("edges", "i")  # unique interactions

        """
        return {i[name] for i in self[field]}
