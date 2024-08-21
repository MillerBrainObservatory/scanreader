"""
__main__.py: scanreader entrypoint.
"""

import typing
from pathlib import Path
import click
from icecream import ic

import scanreader as sr


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
def main(path, frames, zplanes, xslice, yslice, trim_x, trim_y, debug):
    if debug:
        ic.enable()

    files = sr.get_files(path, ext='.tif')
    if len(files) < 1:
        raise ValueError(
            f"Input path given is a non-tiff file: {path}.\n"
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
        debug=debug
    )
    arr = scan[2, 0, :, :]
    quickplot(arr)
    return scan

    # return scan[frames, zplanes, yslice, xslice]


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
    scan = sr.read_scan("~/caiman_data/high_res")
    scan.trim_x = (8,8)
    scan.trim_y = (17,0)
    # _scan = main()
    arr = scan[2, 0, :, :]

    # viewer = napari.Viewer()
    # viewer.add_image(temp, multiscale=False, colormap='gray')
    # napari.run()
    # _scan.save_as_tiff(args.path)
    x = 2
