"""
__main__.py: scanreader entrypoint.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import scanreader as sr


def quick_save_image(arr, save_path):
    fig, ax = plt.subplots()
    ax.imshow(arr)
    plt.savefig(save_path)


def main():
    # set up directories
    datapath = Path().home() / "data"
    rawpath = datapath / "raw"

    # read in the data
    files = [
        str(x) for x in rawpath.glob("*.tif*")
    ]  # grab the first file in the directory
    # return sr.read_scan(files, join_contiguous=True, lbm=True, x_cut=(0, 0), y_cut=(0, 0))
    return sr.read_scan(files, join_contiguous=True)


if __name__ == "__main__":
    scan = main()
