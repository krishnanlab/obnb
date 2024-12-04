import warnings
from pprint import pformat

from obnb.alltypes import Any, Dict, Optional, Union
from obnb.registry import REGISTRIES


def resolve_registry(name: str, scope: Optional[str] = None, verbose: bool = True):
    from obnb import logger  # NOTE: logger needs to be initialized first

    for key, registry in REGISTRIES.items():
        if (scope is not None and not key.startswith(key)) or name not in registry:
            continue

        if verbose:
            logger.info(f"Resolved {name} under {key}: {registry[name]}")

        return registry[name]

    raise ValueError(
        f"Failed to find {name} within scope {scope}. "
        f"Full registries: {pformat(REGISTRIES)}",
    )


def resolve_transform(
    # transform: Optional[Union[str, BaseTransform]],
    transform: Optional[Union[str, Any]],
    transform_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
):
    if isinstance(transform, str):
        transform_cls = resolve_registry(transform, scope="transform/")
        transform = transform_cls(**transform_kwargs)

    # FIX: cyclic imports BaseTransform
    # from obnb.transform.base import BaseTransform
    # elif not isinstance(transform, BaseTransform) and transform is not None:
    #     raise TypeError(f"Unknwon transform type {type(transform)}")

    elif transform_kwargs is not None:
        warnings.warn(
            "Specified transform_kwargs, which are only effective when "
            "the transform objective is passed as a string to be resolved. "
            "Please check if transform_kwargs is passed by mistake.",
            RuntimeWarning,
            stacklevel=2,
        )

    return transform
