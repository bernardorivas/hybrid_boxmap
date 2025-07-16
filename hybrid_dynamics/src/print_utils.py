"""
Utilities for verbosity-controlled printing.
"""

from .config import config


def vprint(message: str, level: str = "normal") -> None:
    """
    Verbosity-controlled print function.

    Args:
        message: Message to print
        level: Print level ('debug', 'normal', 'always')
    """
    if level == "always" or level == "normal" and config.logging.verbose:
        print(message)
    elif level == "debug":
        # Only print debug messages if we're in a very verbose mode
        # For now, we'll skip debug prints
        pass
