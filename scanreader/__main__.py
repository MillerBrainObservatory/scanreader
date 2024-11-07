"""
__main__.py: scanreader entrypoint.
"""

import os
import argparse
import logging
import scanreader as sr
from scanreader.scans import get_metadata
from scanreader import get_files, get_single_file

logging.basicConfig()
logger = logging.getLogger(__name__)

LBM_DEBUG_FLAG = os.environ.get('LBM_DEBUG', 1)

if LBM_DEBUG_FLAG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Scanreader CLI for processing ScanImage tiff files.")
    parser.add_argument("path", type=str, nargs="?", default=None,
                        help="Path to the file or directory to process.")
    parser.add_argument("-f", "--frames", type=str, default=":",
                        help="Frames to read. Use slice notation like NumPy arrays (e.g., 1:50, 10:100:2).")
    parser.add_argument("-m", "--metadata", action="store_true",
                        help="Print a dictionary of metadata.")
    parser.add_argument("-z", "--zplanes", type=str, default=":",
                        help="Z-Planes to read. Use slice notation like NumPy arrays (e.g., 1:50, 5:15:2).")
    parser.add_argument("-x", "--xslice", type=str, default=":",
                        help="X-pixels to read. Use slice notation like NumPy arrays (e.g., 100:500, 0:200:5).")
    parser.add_argument("-y", "--yslice", type=str, default=":",
                        help="Y-pixels to read. Use slice notation like NumPy arrays (e.g., 100:500, 50:250:10).")
    parser.add_argument("-tx", "--trim_x", type=int, nargs=2, default=(0, 0),
                        help="Number of x-pixels to trim from each ROI. Tuple or list (e.g., 4 4 for left and right edges).")
    parser.add_argument("-ty", "--trim_y", type=int, nargs=2, default=(0, 0),
                        help="Number of y-pixels to trim from each ROI. Tuple or list (e.g., 4 4 for top and bottom edges).")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug logs to the terminal.")
    
    args = parser.parse_args()

    if not args.path:
        args.path = sr.lbm_home_dir

    files = sr.get_files(args.path, ext='.tif')
    if len(files) < 1:
        raise ValueError(
            f"Input path given is a non-tiff file: {args.path}.\n"
            f"scanreader is currently limited to scanimage .tiff files."
        )
    else:
        print(f'Found files in {args.path}:\n{files}')

    if args.metadata:
        metadata = get_metadata(files[0])
        return metadata

    frames = process_slice_str(args.frames)
    zplanes = process_slice_str(args.zplanes)
    xslice = process_slice_str(args.xslice)
    yslice = process_slice_str(args.yslice)

    scan = sr.ScanLBM(
        files,
        trim_roi_x=args.trim_x,
        trim_roi_y=args.trim_y,
        debug=args.debug,
        save_path=os.path.join(args.path, 'zarr'),
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
    main()
