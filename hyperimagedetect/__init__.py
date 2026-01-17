"""
HOLO11 Framework - Offline Object Detection Library

See LICENSE file for details.
"""

__version__ = "1.0.0.1"

import importlib
import os
from typing import TYPE_CHECKING

if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"

from hyperimagedetect.utils import ASSETS, SETTINGS
from hyperimagedetect.utils.checks import check_holo as checks


settings = SETTINGS

MODELS = ("HOLO",)

__all__ = (
    "__version__",
    "ASSETS",
    *MODELS,
    "checks",
    "settings",
)

if TYPE_CHECKING:

    from hyperimagedetect.models import HOLO


def __getattr__(name: str):
    """Lazy-import model classes on first access."""
    if name in MODELS:
        return getattr(importlib.import_module("hyperimagedetect.models"), name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """Extend dir() to include lazily available model names for IDE autocompletion."""
    return sorted(set(globals()) | set(MODELS))


if __name__ == "__main__":
    print(__version__)