from __future__ import annotations
import os
import logging
from pathlib import Path
import tifffile
from .multiroi import Field, ROI
from .scans import ScanLBM
from . import scans
from .utils import fix_scan_phase, return_scan_offset

logging.basicConfig()
logger = logging.getLogger(__name__)

LBM_DEBUG_FLAG = os.environ.get('LBM_DEBUG', 1)

if LBM_DEBUG_FLAG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

lbm_home_dir = Path().home() / 'lbm_data'

if not lbm_home_dir.is_dir():
    lbm_home_dir.mkdir()
    logger.info(f'Creating lbm home directory in {lbm_home_dir}')


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


def get_files(
        pathnames: os.PathLike | str | list[os.PathLike | str],
        ext: str = 'tif',
        exclude_pattern: str = '_plane_',
        debug: bool = False,
) -> list[os.PathLike | str] | os.PathLike:
    """
    Expands a list of pathname patterns to form a sorted list of absolute filenames.

    Parameters
    ----------
    pathnames: os.PathLike
        Pathname(s) or pathname pattern(s) to read.
    ext: str
        Extention, string giving the filetype extention.
    exclude_pattern: str | list
        A string or list of strings that match to files marked as excluded from processing.
    debug: bool
        Flag to print found, excluded and all files.

    Returns
    -------
    List[PathLike[AnyStr]]
        List of absolute filenames.
    """
    if '.' in ext or 'tiff' in ext:
        ext = 'tif' #glob tiff and tif
    if isinstance(pathnames, (list, tuple)):
        out_files = []
        excl_files = []
        for fpath in pathnames:
            if exclude_pattern not in str(fpath):
                if Path(fpath).is_file():
                    out_files.extend([fpath])
                elif Path(fpath).is_dir():
                    fnames = [x for x in Path(fpath).expanduser().glob(f"*{ext}*")]
                    out_files.extend(fnames)
            else:
                excl_files.extend(fpath)
        return sorted(out_files)
    if isinstance(pathnames, (os.PathLike, str)):
        pathnames = Path(pathnames).expanduser()
        if pathnames.is_dir():
            files_with_ext = [x for x in pathnames.glob(f"*{ext}*")]
            if debug:
                excluded_files = [x for x in pathnames.glob(f"*{ext}*") if exclude_pattern in str(x)]
                all_files = [x for x in pathnames.glob("*")]
                logger.debug(excluded_files, all_files)
            return sorted(files_with_ext)
        elif pathnames.is_file():
            if exclude_pattern not in str(pathnames):
                return pathnames
            else:
                raise FileNotFoundError(f"No {ext} files found in directory: {pathnames}")
    else:
        raise ValueError(
            f"Input path should be an iterable list/tuple or PathLike object (string, pathlib.Path), not {pathnames}")


def get_single_file(filepath, ext='tif'):
    return [x for x in Path(filepath).glob(f"*{ext}*")][0]


def read_scan(
        pathnames: os.PathLike | str,
        trim_roi_x: list | tuple = (0, 0),
        trim_roi_y: list | tuple = (0, 0),
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

    if isinstance(filenames, (list, tuple)):
        if len(filenames) == 0:
            raise FileNotFoundError(f"Pathname(s) {filenames} do not match any files in disk.")

    # Get metadata from first file
    return scans.ScanLBM(
        filenames,
        trim_roi_x=trim_roi_x,
        trim_roi_y=trim_roi_y
    )


__all__ = [
    "read_scan",
    "get_files",
    "get_single_file",
    "parse_tifffile_metadata",
    "ScanLBM",
    "Field",
    "ROI",
]
