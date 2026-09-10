"""Native model construction for offline experiments; random weights by default.

Use load_cbramod/load_labram for explicit pretrained checkpoint loading.
"""

from ._vendor.cbramod.model import CBraMod

__all__ = ["CBraMod"]
