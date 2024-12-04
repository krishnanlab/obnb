"""Registry module.

Helps interface with user defined modules for OBNB pipelines.

"""
from obnb.alltypes import Any
from obnb.util.misc import DotDict

REGISTRIES = DotDict()

NODEFEAT_REGISTRY_KEY = "transform/node_feature"

SPECIAL_KEY_NAME = "_name_"


def register(scope: str, name: str = SPECIAL_KEY_NAME, obj: Any = None):
    """Register an object.

    Args:
        scope: The scope of this object (i.e., the key for this value in the
            ``REGISTRIES`` dictionary).
        name: Name of the object. By default, this is set to a special key,
            "_name_", which will automatically replaced by the name of the
            object at runtime.
        obj: Object to be registered.

    Note:
        Can be used as decorator or a function. See examples below.

    Examples:
        Use as function:

        >>> register(scope="func", name="custom_func", custom_func)

        Use as decorator:

        >>> @register(scope="func", name="custom_func")
        >>> def custom_func():
        >>>     ...

        If ``name`` is set to special key, e.g., "_name_", then it will be
        automatically converted into the name of the object, which is
        "custom_func" in this case (same as above).

        >>> @register(scope="func", name="_name_")
        >>> def custom_func():
        >>>     ...

    """
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


register_nodefeat = register(scope=NODEFEAT_REGISTRY_KEY)
