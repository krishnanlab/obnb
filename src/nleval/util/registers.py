import typing

from nleval.typing import Optional, Type


@typing.no_type_check  # issue with Type
def overload_class(
    BaseClass: Type,
    suffix: str,
    /,
    sep: str = "_",
    docstring: Optional[str] = None,
    **overload_init_kwargs,
) -> Type:
    """Overload a class with specific keyword argument selections.

    This function is analogous to :func:`functools.partial`, but for classes
    instead of functions.

    Args:
        BaseClass: The base class objects to be overloaded.
        suffix: Suffix to append to the base class name as the new overloaded
            class name. For example, given ``SomeClass`` and the suffix ``New``,
            the newly generated class will be named ``SomeClass_New``.
        sep: Separator between the BaseClass object name and the suffix.
        **overload_init_kwargs: Key word arguments to be used for initializing
            the overloaded class.

    """

    class NewClass(BaseClass):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **overload_init_kwargs, **kwargs)

    NewClass.__name__ = sep.join((BaseClass.__name__, suffix))
    default_docstring = (
        f"Overloaded class ``{NewClass.__name__}`` inherited from "
        f"``{BaseClass.__name__}``.\n\nkwargs: {overload_init_kwargs}"
    )
    NewClass.__doc__ = docstring or default_docstring

    return NewClass
