"""Low-level decoding of Julia `JLD2 <https://github.com/JuliaIO/JLD2.jl>`_ files.

A ``.jld2`` file is plain HDF5, so :mod:`h5py` can open it directly. The only
work is translating JLD2's encoding of Julia values into native Python/numpy:

- **Object references** -> dereferenced and decoded recursively (with cycle and
  depth guards, since JLD2 can share/intern sub-objects).
- **Compound (struct) datatypes** -> a ``dict`` of decoded fields, with a few
  well-known Julia types recognized by their field signature: ``Box``
  (``intervals``), ``BitVector`` (``chunks``/``len``), ``String``
  (``string``/``ncodeunits``), ``Dict`` (``kvvec``), ``UnitRange``
  (``start``/``stop``), and integer-keyed tuples (``"1"``, ``"2"``, ...).
- **Byte strings** -> ``str``.
- **Numeric arrays** -> numpy arrays with **axes fully reversed**: JLD2 stores
  Julia's column-major arrays such that h5py reports the dimensions in reverse,
  so a Julia ``(Y, X[, Z][, C])`` array arrives as ``(…, X, Y)``. We reverse the
  axes on read so every array is returned in Julia's logical order (the order
  this package uses everywhere: ``(Y, X[, Z][, C])`` spatially).

All decoding copies data out of the file, so the caller can close the HDF5
handle immediately and the returned values are plain, picklable Python objects.

This module is internal (underscore-prefixed); see :func:`holytrees.load`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np

_MAX_DEPTH = 64  # real Julia structs nest shallowly; this only bounds pathological refs


@dataclass
class _Ctx:
    """Traversal context: the open file plus cycle/depth guards."""

    f: h5py.File
    depth: int = 0
    seen: frozenset = field(default_factory=frozenset)


def _is_ref(x: Any) -> bool:
    return isinstance(x, h5py.Reference)


def _addr(f: h5py.File, ref: h5py.Reference) -> int | None:
    """Stable in-file address of a referenced object (for cycle detection)."""
    try:
        return int(h5py.h5o.get_info(f[ref].id).addr)
    except Exception:
        return None


def _decode_string_struct(v: np.void) -> str:
    """Decode a Julia ``String`` stored as a ``{string, offset, ncodeunits}`` compound."""
    raw = v["string"]
    if isinstance(raw, bytes):
        off = int(v["offset"]) if "offset" in v.dtype.names else 0
        n = int(v["ncodeunits"]) if "ncodeunits" in v.dtype.names else len(raw) - off
        return raw[off : off + n].decode("utf-8", errors="replace")
    return _to_python(raw, None)


def _decode_intervals(iv: np.void) -> tuple:
    """Decode a Julia ``Box.intervals`` compound into a tuple of ``(lo, hi)`` pairs.

    Each interval (1-based inclusive) is stored as a ``{left, right}`` sub-struct
    under integer-string field names ``"1"``, ``"2"``, ...; returns them in
    dimension order, e.g. ``((1, 120), (1, 120))``.
    """
    dims = sorted(iv.dtype.names, key=int)
    return tuple((int(iv[d]["left"]), int(iv[d]["right"])) for d in dims)


def _decode_compound(ctx: _Ctx, v: np.void) -> Any:
    """Decode a 0-d structured value (a Julia struct) into a Python object."""
    names = v.dtype.names
    if "string" in names and "ncodeunits" in names:  # Julia String
        return _decode_string_struct(v)
    if "chunks" in names and "len" in names:  # BitVector
        return _decode_bitvector(ctx, v)
    if names == ("intervals",):  # Box
        return _decode_intervals(v["intervals"])
    if {"tiles", "tree", "dims", "valid"} <= set(names):  # TileTree
        # `tree` is the internal BoxTree spatial index (a large, deeply nested
        # structure we never use); skip it entirely rather than decode it.
        return {
            "tiles": _decode_field(ctx, v["tiles"]),
            "valid": _decode_field(ctx, v["valid"]),
            "dims": _decode_field(ctx, v["dims"]),
            "tree": None,
        }
    if names == ("kvvec",):  # Dict -> array of Pair{first, second}
        kv = _decode_field(ctx, v["kvvec"])
        out: dict[Any, Any] = {}
        for entry in kv if isinstance(kv, list) else list(np.atleast_1d(kv)):
            if isinstance(entry, dict):
                k = entry.get("first", entry.get("key"))
                val = entry.get("second", entry.get("value"))
                if isinstance(k, (str, int, float, bytes, tuple)):
                    out[k] = val
        return out
    if names == ("start", "stop"):  # UnitRange
        return (int(v["start"]), int(v["stop"]))
    if all(n.isdigit() for n in names):  # Tuple / Dims
        return tuple(_decode_field(ctx, v[n]) for n in sorted(names, key=int))
    return {name: _decode_field(ctx, v[name]) for name in names}


def _decode_bitvector(ctx: _Ctx, v: np.void) -> np.ndarray:
    """Unpack a Julia ``BitVector`` (``{chunks, len, dims}``) into a bool array."""
    chunks = _decode_field(ctx, v["chunks"])
    n = int(v["len"])
    bits = np.unpackbits(np.asarray(chunks).view(np.uint8), bitorder="little")
    return bits[:n].astype(bool)


def _decode_field(ctx: _Ctx, v: Any) -> Any:
    """Decode a single field value, following references."""
    if _is_ref(v):
        return _follow(ctx, v)
    return _to_python(v, ctx)


def _to_python(v: Any, ctx: _Ctx | None) -> Any:
    """Convert an already-materialized h5py value into a native Python object."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.void):
        if ctx is None or ctx.depth > _MAX_DEPTH:
            return {name: _to_python(v[name], None) for name in (v.dtype.names or ())}
        return _decode_compound(ctx, v)
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            if ctx is None:
                return [_to_python(x, None) for x in v.ravel().tolist()]
            return [_decode_field(ctx, x) for x in v.ravel().tolist()]
        if v.dtype.names is not None:
            # Structured array: keep as-is when every field is a plain numeric
            # scalar (e.g. merge_schedule); otherwise decode element-by-element
            # (e.g. boxes_active, whose field is itself a Box struct).
            simple = all(
                v.dtype[n].names is None and v.dtype[n].kind != "O" for n in v.dtype.names
            )
            if simple or ctx is None:
                return v
            return [_decode_compound(ctx, x) for x in v]
        # Plain numeric array: reverse axes (Julia column-major -> our order).
        return np.ascontiguousarray(v.transpose())
    if isinstance(v, np.generic):
        return v.item()
    return v


def _follow(ctx: _Ctx, ref: Any) -> Any:
    """Dereference (if needed), with cycle + depth guards, and fully decode."""
    if not _is_ref(ref):
        return _to_python(ref, ctx)
    if ctx.depth > _MAX_DEPTH:
        return None
    addr = _addr(ctx.f, ref)
    if addr is not None and addr in ctx.seen:
        return None  # cycle / shared sub-object already being decoded
    child = _Ctx(ctx.f, ctx.depth + 1, ctx.seen | ({addr} if addr is not None else set()))
    obj = ctx.f[ref]
    if isinstance(obj, h5py.Dataset):
        return _to_python(obj[()], child)
    if isinstance(obj, h5py.Group):
        return {k: _follow(child, obj[k].ref) for k in obj.keys()}
    return None


def read_jld2(path: str) -> dict[str, Any]:
    """Read every top-level entry of a ``.jld2`` file into a Python ``dict``.

    Parameters
    ----------
    path : str
        Path to a ``.jld2`` (HDF5) file.

    Returns
    -------
    dict
        Mapping from each top-level key to its decoded value. Numeric arrays are
        returned with Julia's logical axis order ``(Y, X[, Z][, C])``; structs
        become nested ``dict``s; ``Box``/``BitVector``/``String``/``Dict``/
        ``UnitRange`` are recognized and decoded to convenient forms. The JLD2
        ``_types`` group is skipped, and the HDF5 file is closed before returning.

    Raises
    ------
    OSError
        If the file is not a valid HDF5 file.
    """
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        ctx = _Ctx(f)
        for key in f.keys():
            if key == "_types":
                continue
            obj = f[key]
            if isinstance(obj, h5py.Datatype):
                continue
            out[key] = _follow(ctx, obj.ref)
    return out


__all__ = ["read_jld2"]
