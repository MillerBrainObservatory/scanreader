"""
__main__.py: scanreader entrypoint.
"""
import argparse
import time
import logging
import warnings
from functools import partial
from pathlib import Path
import scanreader as sr
from scanreader.scans import get_metadata
from scanreader.utils import listify_index

# set logging to critical only
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)

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

    parser.add_argument("path",
                        type=str,
                        default=None,
                        help="Path to the file or directory to process.")
    parser.add_argument("--frames",
                        type=str,
                        default=":",  # all frames
                        help="Frames to read. Use slice notation like NumPy arrays ("
                             "e.g., 1:50 gives frames 1 to 50, 10:100:2 gives frames 10, 20, 30...)."
                        )
    parser.add_argument("--zplanes",
                        type=str,
                        default=":",  # all planes
                        help="Z-Planes to read. Use slice notation like NumPy arrays (e.g., 1:50, 5:15:2).")
    parser.add_argument("--trim_x",
                        type=int,
                        nargs=2,
                        default=(0, 0),
                        help="Number of x-pixels to trim from each ROI. Tuple or list (e.g., 4 4 for left and right "
                             "edges).")
    parser.add_argument("--trim_y", type=int, nargs=2, default=(0, 0),
                        help="Number of y-pixels to trim from each ROI. Tuple or list (e.g., 4 4 for top and bottom "
                             "edges).")
    # Boolean Flags
    parser.add_argument("--metadata", action="store_true",
                        help="Print a dictionary of scanimage metadata for files at the given path.")
    parser.add_argument("--roi",
                        action='store_true',
                        help="Save each ROI in its own folder, organized like 'zarr/roi_1/plane_1/, without this "
                             "arguemnet it would save like 'zarr/plane_1/roi_1'."
                        )

    parser.add_argument("--save", type=str, nargs='?', help="Path to save data to. If not provided, metadata will be "
                                                            "printed.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing files if saving data..")
    parser.add_argument("--tiff", action='store_false', help="Flag to save as .tiff. Default is True")
    parser.add_argument("--zarr", action='store_true', help="Flag to save as .zarr. Default is False")
    parser.add_argument("--assemble", action='store_true', help="Flag to assemble the each ROI into a single image.")

    # Commands
    args = parser.parse_args()
    if not args.path:
        args.path = sr.lbm_home_dir

    files = sr.get_files(args.path, ext='.tiff')
    if len(files) < 1:
        raise ValueError(
            f"Input path given is a non-tiff file: {args.path}.\n"
            f"scanreader is currently limited to scanimage .tiff files."
        )
    else:
        print(f'Found files in {args.path}:\n{files}')

    if args.metadata:
        t_metadata = time.time()
        metadata = get_metadata(files[0])
        t_metadata_end = time.time() - t_metadata
        print(f"Metadata read in {t_metadata_end:.2f} seconds.")
        print(f"Metadata for {files[0]}:")
        # filter out the verbose scanimage frame/roi metadata
        print_params({k: v for k, v in metadata.items() if k not in ['si', 'roi_info']})
    if args.save:
        savepath = Path(args.save).expanduser()
        print(f'Saving z-planes to {savepath}.')

        t_scan_init = time.time()
        scan = sr.ScanLBM(
            files,
            trim_roi_x=args.trim_x,
            trim_roi_y=args.trim_y,
        )
        t_scan_init_end = time.time() - t_scan_init
        print(f"Scan initialized in {t_scan_init_end:.2f} seconds.")

        frames = listify_index(process_slice_str(args.frames), scan.num_frames)
        zplanes = listify_index(process_slice_str(args.zplanes), scan.num_planes)

        if args.zarr:
            ext = '.zarr'
        elif args.tiff:
            ext = '.tiff'
        else:
            raise NotImplementedError("Only .zarr and .tif are supported file formats.")

        t_save = time.time()
        scan.save_as(
            savepath,
            frames=frames,
            planes=zplanes,
            by_roi=args.roi,
            overwrite=args.overwrite,
            ext=ext,
            assemble=args.assemble
        )
        t_save_end = time.time() - t_save
        print(f"Data saved in {t_save_end:.2f} seconds.")
        return scan
    else:
        print(args.path)


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
