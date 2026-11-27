"""
hdf5_reader.py — Custom low-level HDF5/NetCDF4 reader.

h5py 3.16.0 (HDF5 2.0.0) has a regression where the fast_reader fails on
gzip-compressed datasets created by older HDF5 builds.  This module reads
each chunk manually: seek → read raw bytes → zlib decompress → unshuffle →
numpy array, bypassing the broken fast_reader entirely.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Dict

import h5py
import numpy as np


def _read_dataset_all_chunks(h5file: h5py.File, name: str) -> np.ndarray:
    """
    Read a fully gzip+shuffle-compressed dataset by decomposing it chunk by chunk.
    Returns a numpy array in the dataset's native shape and dtype.
    Falls back to normal h5py read if the dataset is not compressed.
    """
    ds = h5file[name]
    shape = ds.shape
    dtype = ds.dtype

    # --- check whether it needs manual read ---
    plist = ds.id.get_create_plist()
    nfilters = plist.get_nfilters()
    if nfilters == 0:
        # no compression — normal read works fine
        return ds[()]

    has_deflate = False
    has_shuffle = False
    elem_size = dtype.itemsize

    for i in range(nfilters):
        ftype, _, _, _ = plist.get_filter(i)
        if ftype == h5py.h5z.FILTER_DEFLATE:
            has_deflate = True
        if ftype == h5py.h5z.FILTER_SHUFFLE:
            has_shuffle = True

    if not has_deflate:
        # only shuffle, try normal path
        return ds[()]

    # --- allocate output ---
    out = np.empty(shape, dtype=dtype)
    nc = ds.id.get_num_chunks()

    nc_path = str(h5file.filename)

    with open(nc_path, "rb") as raw_fp:
        for ci in range(nc):
            info = ds.id.get_chunk_info(ci)
            coff = info.chunk_offset      # tuple, e.g. (0, 0, 0, 0)
            byte_offset = info.byte_offset
            byte_size   = info.size

            # --- read raw compressed bytes ---
            raw_fp.seek(byte_offset)
            compressed = raw_fp.read(byte_size)

            # --- zlib decompress ---
            decompressed = zlib.decompress(compressed)
            raw_bytes = np.frombuffer(decompressed, dtype=np.uint8)

            # --- determine valid slice shape (may be smaller at boundary) ---
            valid_shape = tuple(
                min(cs, s - co)
                for cs, s, co in zip(ds.chunks, shape, coff)
            )
            full_shape = tuple(ds.chunks)
            n_full = int(np.prod(full_shape))
            expected_full = n_full * elem_size
            assert len(raw_bytes) == expected_full, (
                f"chunk {ci}: expected {expected_full} bytes, got {len(raw_bytes)}"
            )

            # --- unshuffle (HDF5 shuffle: byte[b] of all elements stored together) ---
            if has_shuffle:
                unshuffled = np.empty(n_full * elem_size, dtype=np.uint8)
                for b in range(elem_size):
                    unshuffled[b::elem_size] = raw_bytes[b * n_full: (b + 1) * n_full]
            else:
                unshuffled = raw_bytes

            # --- interpret bytes as target dtype and crop to valid boundary ---
            chunk_arr = np.frombuffer(unshuffled.tobytes(), dtype=dtype).reshape(full_shape)
            valid_slices = tuple(slice(0, vs) for vs in valid_shape)
            chunk_arr = chunk_arr[valid_slices]

            # --- place into output ---
            out_slices = tuple(
                slice(co, co + vs) for co, vs in zip(coff, valid_shape)
            )
            out[out_slices] = chunk_arr

    return out


def load_nc_arrays(nc_path: Path) -> Dict[str, np.ndarray]:
    """
    Load all relevant arrays from the NetCDF4/HDF5 file.

    Returns dict with keys:
        'data'   : float32 [T, 6, 32, 36]
        'static' : float32 [2, 32, 36]
        'time'   : int64   [T]  (days since reference — raw HDF5 int values)
        'lat'    : float64 [32]
        'lon'    : float64 [36]
        'channel': object  [6]  (string channel names)
    """
    with h5py.File(nc_path, "r") as f:
        arrays: Dict[str, np.ndarray] = {}
        for key in ("data", "static", "time", "lat", "lon", "channel"):
            arrays[key] = _read_dataset_all_chunks(f, key)
    return arrays
