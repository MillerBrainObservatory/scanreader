"""
__main__.py: scanreader entrypoint.
"""

import os
import argparse
import logging
from pathlib import Path
import scanreader as sr
from scanreader.scans import get_metadata
from scanreader import get_files, get_single_file
from tqdm import tqdm
import tifffile

logging.basicConfig()
logger = logging.getLogger(__name__)

LBM_DEBUG_FLAG = os.environ.get('LBM_DEBUG', 1)

if LBM_DEBUG_FLAG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


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
    parser.add_argument("path", type=str, nargs="?", default=None,
                        help="Path to the file or directory to process.")
    parser.add_argument("-t", "--frames", type=str, default=":",
                        help="Frames to read. Use slice notation like NumPy arrays (e.g., 1:50, 10:100:2).")
    parser.add_argument("-z", "--zplanes", type=str, default=":",
                        help="Z-Planes to read. Use slice notation like NumPy arrays (e.g., 1:50, 5:15:2).")
    parser.add_argument("-x", "--trim_x", type=int, nargs=2, default=(0, 0),
                        help="Number of x-pixels to trim from each ROI. Tuple or list (e.g., 4 4 for left and right edges).")
    parser.add_argument("-y", "--trim_y", type=int, nargs=2, default=(0, 0),
                        help="Number of y-pixels to trim from each ROI. Tuple or list (e.g., 4 4 for top and bottom edges).")
    # Boolean Flags
    parser.add_argument("-m", "--metadata", action="store_true",
                        help="Print a dictionary of metadata.")
    parser.add_argument( "--volume",
                         action='store_true',
                         help="Save the data as a 3D volumetric recording"
                         )
    parser.add_argument( "--roi",
                         action='store_true',
                         help="Save each ROI in its own folder"
                         )
    # Commands
    parser.add_argument("--extract", type=str, help="Extract data to designated filetype")

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
        # filter out the verbose scanimage frame/roi metadata
        print_params({k: v for k, v in metadata.items() if k not in ['si', 'roi_info']})

    if args.extract:
        savepath = Path(args.extract).expanduser()
        print(f'Saving z-planes to {savepath}.')

        frames = process_slice_str(args.frames)
        zplanes = process_slice_str(args.zplanes)

        scan = sr.ScanLBM(
            files,
            trim_roi_x=args.trim_x,
            trim_roi_y=args.trim_y,
        )
        # --volume
        # --roi
        if args.volume:
            data = scan[:]
            from tqdm import tqdm

            # check if roi-based processing is enabled
        if args.roi:
            print('Separating z-planes by ROI.')
            # loop over planes with a progress bar
            for plane in tqdm(range(scan.num_planes), desc="Planes", leave=True):
                # loop over ROIs with a progress bar
                for roi in tqdm(scan.yslices, desc=f"ROIs for plane {plane + 1}", leave=False):
                    data = scan[:, plane, roi, :]
                    name = savepath / f'assembled_plane_{plane + 1}_roi_{roi}.tif'
                    tifffile.imwrite(name, data, bigtiff=True)
        else:
            # loop over planes with a progress bar
            for plane in tqdm(range(scan.num_planes), desc="Planes"):
                data = scan[:, plane, :, :]
                name = savepath / f'assembled_plane_{plane + 1}.tif'
                tifffile.imwrite(name, data, bigtiff=True)
        return scan
    else:
        return metadata



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
