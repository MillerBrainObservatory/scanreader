"""
__main__.py: scanreader entrypoint.
"""

import os
import click
import logging

import scanreader as sr

logging.basicConfig()
logger = logging.getLogger(__name__)

LBM_DEBUG_FLAG = os.environ.get('LBM_DEBUG', 1)

if LBM_DEBUG_FLAG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


@click.command()
@click.argument("path", type=click.Path(exists=False, file_okay=True, dir_okay=True))
@click.option(
    "-f", "--frames", type=str, default=":",
    help="Frames to read. Use slice notation like NumPy arrays (e.g., 1:50, 10:100:2)."
)
@click.option(
    "-z", "--zplanes", type=str, default=":",
    help="Z-Planes to read. Use slice notation like NumPy arrays (e.g., 1:50, 5:15:2)."
)
@click.option(
    "-x", "--xslice", type=str, default=":",
    help="X-pixels to read. Use slice notation like NumPy arrays (e.g., 100:500, 0:200:5)."
)
@click.option(
    "-y", "--yslice", type=str, default=":",
    help="Y-pixels to read. Use slice notation like NumPy arrays (e.g., 100:500, 50:250:10)."
)
@click.option(
    "-tx", "--trim_x", type=tuple, default=(0, 0),
    help="Number of x-pixels to trim from each ROI. Tuple or list (Python syntax, e.g., (4,4)). Left edge, right edge"
)
@click.option(
    "-ty", "--trim_y", type=tuple, default=(0, 0),
    help="Number of y-pixels to trim from each ROI. Tuple or list (Python syntax, e.g., (4,4)). Top edge, bottom edge"
)
@click.option(
    "-d", "--debug", type=click.BOOL, default=False,
    help="Enable debug logs to the terminal."
)
def main(datapath, frames, zplanes, xslice, yslice, trim_x, trim_y, debug):
    if not datapath:
        datapath = sr.lbm_home_dir

    files = sr.get_files(datapath, ext='.tif')
    if len(files) < 1:
        raise ValueError(
            f"Input path given is a non-tiff file: {datapath}.\n"
            f"scanreader is currently limited to scanimage .tiff files."
        )

    frames = process_slice_str(frames)
    zplanes = process_slice_str(zplanes)
    xslice = process_slice_str(xslice)
    yslice = process_slice_str(yslice)

    scan = sr.ScanLBM(
        files,
        trim_roi_x=trim_x,
        trim_roi_y=trim_y,
        debug=debug,
        save_path=datapath / 'zarr',
    )
    return scan


def process_slice_str(slice_str):
    if not isinstance(slice_str, str):
        raise ValueError(f"Expected a string argument, received: {slice_str}")
    if slice_str.isdigit():
        return int(slice_str)
    else:
        parts = slice_str.split(":")
    return slice(*[int(p) if p else None for p in parts])


def process_slice_objects(slice_str):
    return tuple(map(process_slice_str, slice_str.split(",")))


if __name__ == "__main__":
    from pathlib import Path
    scan = sr.read_scan("~/caiman_data/mk717/")
    scan.trim_x = (8,8)
    scan.trim_y = (17,0)
    arr = scan[2, 0, :, :]
    path = Path().home() / 'caiman_data' / 'high_res'
    # scan.save_as_zarr(path)
