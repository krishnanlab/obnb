"""Utility for exploring NDEx CX data."""
import itertools
import json
from collections import defaultdict
from pprint import pformat
from typing import no_type_check

import ndex2
from requests import RequestException

from obnb.typing import Dict, List, Optional, Set


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

    @no_type_check
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

        Note:
            Only counts `str`, `int`, and `float` types.

        """
        return {
            i[name] for i in self[field] if isinstance(i.get(name), (str, int, float))
        }

    def node_types(self, sep: str = ":", channel: str = "r") -> Set[str]:
        """Return all node types.

        Node types are indicated by the node IDs, where the item before the
        separator (default is `:`) is the node-type, and the one after is the
        actual ID.

        Args:
            sep: Separator for the node ID.
            channel: Which channel in the `nodes` field to use. Typically there
                are three channels: (1) `@id` is the index of the node, (2)
                `n` is the name of the node, and (3) `r` is the alternative
                representation not the node.

        """
        return {i[channel].split(sep)[0] for i in self["nodes"]}

    def edge_types(
        self,
        name_channel: str = "n",
        value_channel: str = "v",
    ) -> Dict[str, int]:
        """Return all edge types with counts.

        Args:
            name_channel: Which channel in the `edgeAttributes` field to use for
                inferring edge type.
            value_channel: Which channel in the `edgeAttributes` field to use
                as edge weights.

        Note:
            Only edges with positive edge weights are counted.

        """
        ets_counts: Dict[str, int] = defaultdict(int)
        for edge_attr in self["edgeAttributes"]:
            name = edge_attr[name_channel]
            value = edge_attr[value_channel]
            try:
                ets_counts[name] += 1 if float(value) > 0 else 0
            except (ValueError, TypeError):
                continue

        # Sort by size
        ets = sorted(ets_counts, key=ets_counts.get, reverse=True)  # type: ignore
        out = {i: ets_counts[i] for i in ets}

        return out
