"""
__main__.py: scanreader entrypoint.
"""

from pathlib import Path
import zarr
import napari
from matplotlib import pyplot as plt

import scanreader as sr


def main(*args, **kwargs) -> sr.ScanLBM:

    # set the data path
    if args:
        datapath = Path(args[0])
    else:
        datapath = kwargs.get("datapath", Path().home() / "caiman_data")

    # read in the data
    files = [str(x) for x in datapath.glob("*.tif*")]
    return sr.read_scan(files, join_contiguous=True)


if __name__ == "__main__":
    scan = main()
    data = scan[:, :, :, 0, :].squeeze()
    viewer = napari.Viewer()
    viewer.add_image(data, name="data", colormap='gray')
    napari.run()

x = 2
