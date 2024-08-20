from __future__ import annotations

import os
import re
import typing
from os import PathLike
from pathlib import Path

import tifffile

from . import scans

def parse_tifffile_metadata(tiff_file: tifffile.TiffFile):

    series = tiff_file.series[0]
    scanimage_metadata = tiff_file.scanimage_metadata
    roi_group = scanimage_metadata["RoiGroups"]["imagingRoiGroup"]["rois"]
    pages = tiff_file.pages

    return {
        "roi_info": roi_group,
        "photometric": "minisblack",
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
    

def read_scan(
    pathnames: os.PathLike | typing.Iterable[os.PathLike],
    trim_roi_x: list | tuple = (0,0),
    trim_roi_y: list | tuple = (0,0),
    debug=False,
) -> scans.ScanLBM:
    """
    Reads a ScanImage scan.

    Parameters
    ----------
    pathnames: os.PathLike
        Pathname(s) or pathname pattern(s) to read.
    trim_roi_x: tuple, list, optional
        Indexable (trim_roi_x[0], trim_roi_x[1]) item with 2 integers denoting the amount of pixels to trim on the left [0] and right [1] side of **each roi**.
    trim_roi_y: tuple, list, optional
        Indexable (trim_roi_y[0], trim_roi_y[1]) item with 2 integers denoting the amount of pixels to trim on the top [0] and bottom [1] side of **each roi**.
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

    if debug:
        ic.enable()

    if len(filenames) == 0:
        error_msg = "Pathname(s) {} do not match any files in disk.".format(pathnames)
        raise FileNotFoundError(error_msg)

    # Get metadata from first file
    return scans.ScanLBM(
        filenames,
        trim_roi_x=trim_roi_x,
        trim_roi_y = trim_roi_y
    )

def get_files(
    pathnames: os.PathLike | list[os.PathLike],
) -> list[PathLike]:
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
        if Path(pathnames).suffix in [".tif", ".tiff"]:
            out_files.append(str(pathnames))
    return sorted(out_files)


def get_scanimage_version(info):
    """Looks for the ScanImage version in the tiff file headers."""
    pattern = re.compile(r"SI.?\.VERSION_MAJOR = '?(?P<version>[^\s']*)'?")
    match = re.search(pattern, info)
    version = None
    if match:
        version = match.group("version")
    return version


def is_scan_multiROI(info):
    """Looks whether the scan is multiROI in the tiff file headers."""
    match = re.search(r"hRoiManager\.mroiEnable = (?P<is_multiROI>.)", info)
    is_multiROI = (match.group("is_multiROI") == "1") if match else None
    return is_multiROI
