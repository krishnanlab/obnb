import pytest

from NLEval.util.checkers import checkConfig

config1 = {"a": 1, "b": "s", "c": None}
config2 = {"a": 1, "b": {"x": 1, "y": "2"}}
config3 = {
    "a": 1,
    "b": {
        "x": 1,
        "y": 2,
        "z": {
            "u": 3,
            "v": 6,
        },
    },
}
config4 = {"a": 3, "b": [1, 2, "3"], "c": {"x": [1, 2, "3"]}}


def test_check_config_valtypes():
    checkConfig("config1", config1, allowed_types=(int, str, type(None)))
    checkConfig("config2", config2, allowed_types=(int, str))

    # int is not float
    with pytest.raises(TypeError):
        checkConfig("config1", config1, allowed_types=(float, str, type(None)))

    # Missing str and None types
    with pytest.raises(TypeError):
        checkConfig("config1", config1, allowed_types=(int,))

    # Missing None type
    with pytest.raises(TypeError):
        checkConfig("config1", config1, allowed_types=(int, str))

    # Incorrect str type in nested dict
    with pytest.raises(TypeError):
        checkConfig("config2", config2, allowed_types=(int,))


def test_check_config_depth():
    checkConfig("config2", config2, allowed_types=(int, str), max_depth=2)
    with pytest.raises(ValueError):
        checkConfig("config2", config2, allowed_types=(int, str), max_depth=1)

    checkConfig("config3", config3, allowed_types=(int,), max_depth=3)
    with pytest.raises(ValueError):
        checkConfig("config3", config3, allowed_types=(int,), max_depth=2)

    checkConfig("config4", config4, allowed_types=(int, str), max_depth=2)
    with pytest.raises(ValueError):
        checkConfig("config4", config4, allowed_types=(int, str), max_depth=1)
