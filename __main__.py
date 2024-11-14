"""
__main__.py: scanreader entrypoint.
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import scanreader as sr


def quick_save_image(arr, save_path):
    fig, ax = plt.subplots()
    ax.imshow(arr)
    plt.savefig(save_path)


def main():
    # set up directories
    datapath = Path().home() / "caiman_data" / 'raw'
    # read in the data
    files = [
        str(x) for x in datapath.glob("*full.tif*")
    ]  # grab the first file in the directory
    # return sr.read_scan(files, join_contiguous=True, lbm=True, x_cut=(0, 0), y_cut=(0, 0))
    return sr.read_scan(files)


def handle_args():
    parser = argparse.ArgumentParser(description="Extract raw ScanImage Tiff Files")
    parser.add_argument("--input", default=Path().home(),
                        help="Path/Directory containing raw ScanImage .tiff files.")
    parser.add_argument("--input", action="append", help="File(s) to work on, provide multiple times for more files")
    return parser.parse_args()


if __name__ == "__main__":
    scan = main()
