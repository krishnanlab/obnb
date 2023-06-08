"""Configuration of loggers used by obnb."""

LOGGER_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "defaultFormatter": {
            "format": "[%(levelname)s][%(name)s][%(funcName)s] %(message)s",
        },
        "briefFormatter": {
            "format": "%(message)s",
        },
        "preciseFormatter": {
            "format": "[%(levelname)s][%(asctime)s][%(module)s][%(funcName)s] %(message)s",
        },
    },
    "handlers": {
        "defaultConsoleHandler": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "defaultFormatter",
        },
        "briefConsoleHandler": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "briefFormatter",
        },
        "preciseConsoleHandler": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "preciseFormatter",
        },
    },
    "loggers": {
        "obnb": {
            "level": "INFO",
            "handlers": ["defaultConsoleHandler"],
            "propagate": False,
        },
        "obnb_brief": {
            "level": "INFO",
            "handlers": ["briefConsoleHandler"],
            "propagate": False,
        },
        "obnb_precise": {
            "level": "INFO",
            "handlers": ["preciseConsoleHandler"],
            "propagate": False,
        },
    },
}
