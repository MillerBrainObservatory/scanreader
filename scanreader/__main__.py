"""
__main__.py: scanreader entrypoint.
"""

import typing
from pathlib import Path
import argparse

import dask

import scanreader as sr


def parse_args():
    parser = argparse.ArgumentParser(description="scanreader CLI. Read a LBM ScanImage .tiff file.")
    help_str = (
        "Path to the .tiff file or directory containing the .tiff files. \n"
        "If a directory is given, all .tiff files in the directory will be read. \n"
        "If a file is given, only that file will be read."
    )
    parser.add_argument("path", type=Path, help=help_str)
    parser.add_argument(
        "-t", "--timepoints", type=str, help="Frames to read (i.e. 1:50)", default=":"
    )
    parser.add_argument(
        "-z", "--zslice", type=str, help="Z-Planes to read (i.e. 1:50)", default=":"
    )
    parser.add_argument(
        "-x", "--xslice", type=str, help="X-pixels to read (i.e. 1:50)", default=":"
    )
    parser.add_argument(
        "-y", "--yslice", type=str, help="Y-pixels to read (i.e. 1:50)", default=":"
    )
    parser.add_argument(
        "-d", "--debug", type=str,
        help="Flag (True/False) to enable debug logs to the terminal (to file coming soon).",
        default="False"
    )
    _args = parser.parse_args()
    if _args.path is None:
        path = Path().home() / "caiman_data"
        if not path.is_dir():
            path.mkdir()
        else:
            tiff_files = [str(x) for x in path.glob("*.tif*")]
            print(f"Files found in {path}: \n{tiff_files}")
    for arg_slice in ["timepoints", "zslice", "xslice", "yslice"]:
        setattr(_args, arg_slice, process_slice_str(getattr(_args, arg_slice)))
    return _args


def process_slice_str(slice_str):
    if not isinstance(slice_str, str):
        raise ValueError(f"Expected a string argument, received: {slice_str}")
    else:
        parts = slice_str.split(":")
    return slice(*[int(p) if p else None for p in parts])


def process_slice_objects(slice_str):
    return tuple(map(process_slice_str, slice_str.split(",")))

# entrypoint to napari
def imread(path, slice_objects: typing.Iterable) -> dask.core.Any:
    __scan = sr.read_scan(path, debug=True)
    return __scan[slice_objects]

def quickplot(array):
    import matplotlib.pyplot as plt
    plt.imshow(array, cmap='gray')
    plt.show()

if __name__ == "__main__":

    args = parse_args()
    _scan = sr.read_scan(args.path,)
    _scan.trim_x = (8,8)
    quickplot(_scan[2,0,:,:])
    x=2