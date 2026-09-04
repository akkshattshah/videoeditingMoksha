"""vre — video reference engine.

Decompiles a reference video into a structured Blueprint, then routes it to the
cheapest recompile path that can preserve its structure.
"""

from .schema import Blueprint, Shot, SourceMeta

__version__ = "0.1.0"
__all__ = ["Blueprint", "Shot", "SourceMeta"]
