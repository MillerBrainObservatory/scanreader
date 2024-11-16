"""
__main__.py: scanreader entrypoint.
"""
import time
import argparse
import logging
import warnings
from functools import partial
from pathlib import Path
import scanreader as sr
from scanreader.scans import get_metadata
from scanreader.utils import listify_index

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# suppress warnings
warnings.filterwarnings("ignore")

print = partial(print, flush=True)


def print_params(params, indent=5):
    for k, v in params.items():
        # if value is a dictionary, recursively call the function
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            print_params(v, indent + 4)
        else:
            print(" " * indent + f"{k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Scanreader CLI for processing ScanImage tiff files.")
    parser.add_argument(
        "path",
        type=str,
        default=None,
        help="Path to the file or directory to process."
    )
    parser.add_argument(
        "--frames",
        type=str,
        default=":",  # all frames
        help="Frames to read. Use slice notation like NumPy arrays (e.g., 1:50 gives frames 1 to 50, 10:100:2 gives " \
             "frames 10, 20, 30...)."
    )
    parser.add_argument(
        "--zplanes",
        type=str,
        default=":",  # all planes
        help="Z-Planes to read. Use slice notation like NumPy arrays (e.g., 1:50, 5:15:2)."
    )
    parser.add_argument(
        "--trim_x",
        type=int,
        nargs=2,
        default=(0, 0),
        help="Number of x-pixels to trim from each ROI. Tuple or list (e.g., 4 4 for left and right "
             "edges).")
    parser.add_argument(
        "--trim_y", type=int, nargs=2, default=(0, 0),
        help="Number of y-pixels to trim from each ROI. Tuple or list (e.g., 4 4 for top and bottom edges).")
    # Boolean Flags
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Print a dictionary of scanimage metadata for files at the given path.")
    parser.add_argument(
        "--roi",
        action='store_true',
        help="Save each ROI in its own folder, organized like 'zarr/roi_1/plane_1/, without this "
             "arguemnet it would save like 'zarr/plane_1/roi_1'."
    )

    parser.add_argument("--save", type=str, nargs='?', help="Path to save data to. If not provided, metadata will be printed.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing files if saving data..")
    parser.add_argument("--tiff", action='store_false', help="Flag to save as .tiff. Default is True")
    parser.add_argument("--zarr", action='store_true', help="Flag to save as .zarr. Default is False")
    parser.add_argument("--assemble", action='store_true', help="Flag to assemble the each ROI into a single image.")
    parser.add_argument("--debug", action='store_true', help="Print debug information during processing.")

    # Commands
    args = parser.parse_args()

    process_start = time.time()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    metadata = None
    if not args.path:
        args.path = sr.lbm_home_dir
        logger.info(f"No path provided, using default path: {args.path}")
    files = sr.get_files(args.path, ext='.tif')

    if files is None:
        raise FileNotFoundError(f"No .tif files found in directory: {args.path}")

    if len(files) < 1:
        raise ValueError(
            f"Input path given is a non-tiff file: {args.path}.\n"
            f"scanreader is currently limited to scanimage .tiff files."
        )

    logger.debug(f"Files found: {files}")
    if args.save:
        savepath = Path(args.save).expanduser()
        print(f'Saving z-planes to {savepath}.')

        scan = sr.ScanLBM(
            files,
            trim_roi_x=args.trim_x,
            trim_roi_y=args.trim_y,
        )

        frames = listify_index(process_slice_str(args.frames), scan.num_frames)
        zplanes = listify_index(process_slice_str(args.zplanes), scan.num_planes)

        if args.zarr:
            ext = '.zarr'
        elif args.tiff:
            ext = '.tiff'
        else:
            raise NotImplementedError("Only .zarr and .tif are supported file formats.")
        scan.save_as(
            savepath,
            frames=frames,
            planes=zplanes,
            by_roi=args.roi,
            overwrite=args.overwrite,
            ext=ext,
            assemble=args.assemble
        )
        process_stop = time.time()
        logger.info(f"Processing completed in {process_stop - process_start:.2f} seconds.")
        return scan
    else:
        logger.info(f"Gathering metadata for {files}")
        if metadata is None:
            metadata = get_metadata(files[0])
        print_params({k: v for k, v in metadata.items() if k not in ['si', 'roi_info']})


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
