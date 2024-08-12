from __future__ import annotations

import os
import re
import typing
from glob import glob
from os import path, PathLike
from pathlib import Path
from typing import List, AnyStr

import numpy as np
from tifffile import TiffFile

from . import scans
from .exceptions import ScanImageVersionError, PathnameError

_scans = {'5.1': scans.Scan5Point1, '5.2': scans.Scan5Point2, '5.3': scans.Scan5Point3,
          '5.4': scans.Scan5Point4, '5.5': scans.Scan5Point5,
          '5.6': scans.Scan5Point6, '5.7': scans.Scan5Point7,
          '2016b': scans.Scan2016b,
          '2017a': scans.Scan2017a, '2017b': scans.Scan2017b,
          '2018a': scans.Scan2018a, '2018b': scans.Scan2018b,
          '2019a': scans.Scan2019a, '2019b': scans.Scan2019b,
          '2020': scans.Scan2020, '2021': scans.Scan2021}


def read_scan(pathnames: os.PathLike | typing.Iterable[os.PathLike], dtype=np.int16, join_contiguous=True,
              debug=False) -> scans.BaseScan:
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
        error_msg = 'Pathname(s) {} do not match any files in disk.'.format(pathnames)
        raise PathnameError(error_msg)

    # Get metadata from first file
    with TiffFile(filenames[0]) as tiff_file:
        header = tiff_file.pages[0].description
        head2 = tiff_file.pages[0].software
        series = tiff_file.series[0]
        if 'SI.VERSION_MAJOR' not in tiff_file.scanimage_metadata['FrameData']:
            raise ScanImageVersionError('No SI.VERSION_MAJOR found in metadata.')
        scanimage_metadata = tiff_file.scanimage_metadata
        roi_group = scanimage_metadata['RoiGroups']['imagingRoiGroup']['rois']
        pages = tiff_file.pages
        image_info = {
            'roi_info': roi_group,
            'image_height': pages[0].shape[0],
            'image_width': pages[0].shape[1],
            'num_pages': len(pages),
            'dims': series.dims,
            'axes': series.axes,
            'dtype': series.dtype,
            'is_multifile': series.is_multifile,
            'nbytes': series.nbytes,
            'shape': series.shape,
            'size': series.size,
            'dim_labels': series.sizes,
            'num_rois': len(roi_group),
            'pxy': roi_group[0]['scanfields']['pixelResolutionXY'],
            'sxy': roi_group[0]['scanfields']['sizeXY'],
            'objective_resolution': scanimage_metadata['FrameData']['SI.objectiveResolution'],
            'metadata': scanimage_metadata['FrameData']
        }
    return scans.ScanLBM(image_info, join_contiguous=join_contiguous, header=f'{header}\n{head2}')


def get_files(pathnames: os.PathLike | List[os.PathLike | str]) -> List[PathLike[AnyStr]]:
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
    pathnames = Path(pathnames).expanduser()  # expand ~ to /home/user
    if not pathnames.exists():
        raise FileNotFoundError(f'Path {pathnames} does not exist as a file or directory.')
    if pathnames.is_file():
        return [pathnames]
    if pathnames.is_dir():
        pathnames = [fpath for fpath in pathnames.glob("*.tif*")]  # matches .tif and .tiff
    return sorted(pathnames, key=path.basename)

def expand_wildcard(wildcard: os.PathLike | str | list[os.PathLike | str]) -> list[PathLike[AnyStr]]:
    """ Expands a list of pathname patterns to form a sorted list of absolute filenames. """
    # Check input type
    wildcard = Path(wildcard)

    # Expand wildcards
    rel_filenames = [glob(wildcard) for wildcard in wildcard_list]
    rel_filenames = [item for sublist in rel_filenames for item in sublist]  # flatten list

    abs_filenames = [path.abspath(filename) for filename in rel_filenames]
    sorted_filenames = sorted(abs_filenames, key=path.basename)
    return sorted_filenames


def get_scanimage_version(info):
    """ Looks for the ScanImage version in the tiff file headers. """
    pattern = re.compile(r"SI.?\.VERSION_MAJOR = '?(?P<version>[^\s']*)'?")
    match = re.search(pattern, info)
    if match:
        version = match.group('version')
    else:
        raise ScanImageVersionError('Could not find ScanImage version in the tiff header')

    return version


def is_scan_multiROI(info):
    """ Looks whether the scan is multiROI in the tiff file headers. """
    match = re.search(r'hRoiManager\.mroiEnable = (?P<is_multiROI>.)', info)
    is_multiROI = (match.group('is_multiROI') == '1') if match else None
    return is_multiROI
