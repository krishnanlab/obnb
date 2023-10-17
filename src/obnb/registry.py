from obnb.typing import Any
from obnb.util.misc import DotDict

REGISTRIES = DotDict()

NODEFEAT_REGISTRY_KEY = "transform/node_feature"

SPECIAL_KEY_NAME = "_class_"


def register(scope: str, name: str, obj: Any = None):
    if obj is not None:
        if name == SPECIAL_KEY_NAME:
            name = obj.__name__

        if scope not in REGISTRIES:
            REGISTRIES[scope] = DotDict()
        elif name in REGISTRIES[scope]:
            raise KeyError(
                f"{name!r} is already registered for {scope!r}: {REGISTRIES[scope]!r}",
            )

        REGISTRIES[scope][name] = obj

        return

    # Use as decorator
    def bounded_register(obj: Any):
        register(scope, name, obj)
        return obj

    return bounded_register


register_nodefeat = register(scope=NODEFEAT_REGISTRY_KEY, name=SPECIAL_KEY_NAME)
