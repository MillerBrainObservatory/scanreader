from __future__ import annotations

import os
import re
import typing
from glob import glob
from os import path, PathLike
from pathlib import Path
from typing import List, AnyStr

import numpy as np
import tifffile

from . import scans
from .exceptions import ScanImageVersionError, PathnameError

def read_scan(
    pathnames: os.PathLike | typing.Iterable[os.PathLike],
    dtype=np.int16,
    join_contiguous=True,
    debug=False,
) -> scans.ScanLBM:
    """
    Reads a ScanImage scan.

    Parameters
    ----------
    pathnames: os.PathLike
        Pathname(s) or pathname pattern(s) to read.
    dtype: data-type, optional
        Data type of the output array.
    join_contiguous: bool, optional
        For multiROI scans (2016b and beyond) it will join contiguous scanfields in the same depth.
        No effect in non-multiROI scans. See help of ScanMultiROI._join_contiguous_fields for details.
    debug : bool, optional
        If True, it will print debug information.

    Returns
    -------
    ScanLBM
        A Scan object (subclass of ScanMultiROI) with metadata and different offset correction methods.
        See Readme for details.

    """
    # Expand wildcards
    filenames = get_files(pathnames)

    if len(filenames) == 0:
        error_msg = "Pathname(s) {} do not match any files in disk.".format(pathnames)
        raise PathnameError(error_msg)

    # Get metadata from first file
    with tifffile.TiffFile(filenames[0]) as tiff_file:
        series = tiff_file.series[0]
        if "SI.VERSION_MAJOR" not in tiff_file.scanimage_metadata["FrameData"]:
            raise ScanImageVersionError("No SI.VERSION_MAJOR found in metadata.")
        scanimage_metadata = tiff_file.scanimage_metadata
        roi_group = scanimage_metadata["RoiGroups"]["imagingRoiGroup"]["rois"]
        pages = tiff_file.pages
        image_info = {
            "roi_info": roi_group,
            "photometric": 'minisblack',
            "image_height": pages[0].shape[0],
            "image_width": pages[0].shape[1],
            "num_pages": len(pages),
            "dims": series.dims,
            "axes": series.axes,
            "dtype": series.dtype,
            "is_multifile": series.is_multifile,
            "nbytes": series.nbytes,
            "shape": series.shape,
            "size": series.size,
            "dim_labels": series.sizes,
            "num_rois": len(roi_group),
            "si": scanimage_metadata["FrameData"],
        }

    return scans.ScanLBM(
        filenames, image_info, join_contiguous=join_contiguous,
    )

def get_files(
    pathnames: os.PathLike | List[os.PathLike | str],
) -> list[PathLike | str]:
    """
    Expands a list of pathname patterns to form a sorted list of absolute filenames.

    Parameters
    ----------
    pathnames: os.PathLike
        Pathname(s) or pathname pattern(s) to read.

    Returns
    -------
    List[PathLike[AnyStr]]
        List of absolute filenames.
    """
    out_files = []
    if isinstance(pathnames, (list, tuple)):
        for fpath in pathnames:
            out_files.append(str(x) for x in Path(fpath).expanduser().glob("*.tif*"))
    elif Path(pathnames).is_dir():
        file_list = [str(x) for x in Path(pathnames).expanduser().glob("*.tif*")]
        return file_list
    elif Path(pathnames).is_file():
        if Path(pathnames).suffix in ['.tif', '.tiff']:
            out_files.append(str(pathnames))
    return sorted(out_files)

def expand_wildcard(
    wildcard: os.PathLike | str | list[os.PathLike | str],
) -> list[PathLike[AnyStr]]:
    """Expands a list of pathname patterns to form a sorted list of absolute filenames."""
    # Check input type
    wildcart = Path(wildcard)
    wcl = [x for x in wildcard.glob(wildcard)]

    # Expand wildcards
    rel_filenames = [glob(wildcard) for wildcard in wcl]
    rel_filenames = [
        item for sublist in rel_filenames for item in sublist
    ]  # flatten list

    abs_filenames = [path.abspath(filename) for filename in rel_filenames]
    sorted_filenames = sorted(abs_filenames, key=path.basename)
    return sorted_filenames


def get_scanimage_version(info):
    """Looks for the ScanImage version in the tiff file headers."""
    pattern = re.compile(r"SI.?\.VERSION_MAJOR = '?(?P<version>[^\s']*)'?")
    match = re.search(pattern, info)
    if match:
        version = match.group("version")
    else:
        raise ScanImageVersionError(
            "Could not find ScanImage version in the tiff header"
        )

    return version


def is_scan_multiROI(info):
    """Looks whether the scan is multiROI in the tiff file headers."""
    match = re.search(r"hRoiManager\.mroiEnable = (?P<is_multiROI>.)", info)
    is_multiROI = (match.group("is_multiROI") == "1") if match else None
    return is_multiROI

def tiffs2zarr(filenames, zarrurl, chunksize, **kwargs):
    """Write images from sequence of TIFF files as zarr."""

    def imread(filename):
        # return first image in TIFF file as numpy array
        with open(filename, 'rb') as fh:
            data = fh.read()
        from imagecodecs.imagecodecs import imagecodecs
        return imagecodecs.tiff_decode(data)

    with tifffile.FileSequence(imread, filenames) as tifs:
        with tifs.aszarr() as store:
            da = dask.array.from_zarr(store)
            chunks = (chunksize,) + da.shape[1:]
            da.rechunk(chunks).to_zarr(zarrurl, **kwargs)
