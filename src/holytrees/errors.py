"""Exception types for :mod:`holytrees`.

A small hierarchy so callers can distinguish "this isn't a run I can read" from
ordinary ``ValueError``/``KeyError`` raised by numpy or h5py.
"""

from __future__ import annotations


class HolyTreesError(Exception):
    """Base class for all :mod:`holytrees` errors."""


class RunFormatError(HolyTreesError):
    """The file/directory is not a readable pipeline run.

    Raised when the path is not an HDF5 file, or when a required key
    (e.g. ``ttree_final`` or ``bg``) is missing from a saved run.
    """


__all__ = ["HolyTreesError", "RunFormatError"]
