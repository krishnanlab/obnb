from nleval.typing import Type


def overload_class(BaseClass: Type, suffix: str, /, **overload_init_kwargs) -> Type:
    """Overload a class with specific keyword argument selections.

    This function is analogous to :func:`functools.partial`, but for classes
    instead of functions.

    Args:
        BaseClass: The base class objects to be overloaded.
        suffix: Suffix to append to the base class name as the new overloaded
            class name. For example, given ``SomeClass`` and the suffix ``New``,
            the newly generated class will be named ``SomeClass_New``.
        **overload_init_kwargs: Key word arguments to be used for initializing
            the overloaded class.

    """

    class NewClass(BaseClass):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **overload_init_kwargs, **kwargs)

    NewClass.__name__ = "_".join((BaseClass.__name__, suffix))
    NewClass.__doc__ = (
        f"Overloaded class ``{NewClass.__name__}`` inherited from "
        f"``{BaseClass.__name__}`` with kwargs:\n\n{overload_init_kwargs}"
    )

    return NewClass
