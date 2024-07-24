"""
__main__.py: scanreader entrypoint.
"""
import typing
from pathlib import Path

import dask
import scanreader as sr
import argparse


def main(path) -> sr.ScanLBM:
    datapath = Path(path)
    files = [str(x) for x in datapath.glob("*.tif*")]
    return sr.read_scan(files, join_contiguous=True)


def imread(path, slice_objects: typing.Iterable) -> dask.core.Any:
    _scan = main(path)
    return _scan[slice_objects]


def parse_args():
    parser = argparse.ArgumentParser(description="Read a LBM ScanImage .tiff file.")
    help_str = (
        "Path to the .tiff file or directory containing the .tiff files. \n"
        "If a directory is given, all .tiff files in the directory will be read. \n"
        "If a file is given, only that file will be read."
    )
    parser.add_argument("path", type=Path, help=help_str)
    parser.add_argument(
        "slice_objects",
        type=str,
        help="Slicing objects in the format start:stop:step for each dimension, separated by commas."
    )
    return parser.parse_args()


def open_gui(_scan):
    import napari
    _viewer = napari.Viewer()
    _viewer.add_image(_scan, name="data", colormap='gray')
    napari.run()


if __name__ == "__main__":
    args = parse_args()
    slices = tuple(slice(*map(lambda x: int(x) if x else None, s.split(':'))) for s in args.slice_objects.split(','))
    scan = imread(args.path, slices)
    open_gui(scan)
