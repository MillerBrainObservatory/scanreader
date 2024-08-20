---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Demo: scanreader

Examples of scanreader usage from a [jupyter notebook](https://jupyter.org/) and from the commmand line.

## Libraries:

- [tifffile](https://github.com/cgohlke/tifffile/blob/master/tifffile/tifffile.py) to read ScanImage BigTiff files.
- [zarr](https://zarr.readthedocs.io/en/stable/) and [dask](https://www.dask.org/) for lazy-loading operations

## Imports

```{code-cell} ipython3
from pathlib import Path
import matplotlib.pyplot as plt
from scanreader import read_scan

%load_ext autoreload
%autoreload 2
```

## Data path setup

Put all of the `.tiff` output files in a single directory. There should be **no other `.tiff` files other than those belonging to this session.

```{code-cell} ipython3
datapath = Path().home() / 'caiman_data' / 'high_res'
if datapath.is_dir():
    print([x.expanduser() for x in datapath.glob("*.tif*")])
else:
    print(f"No tiff files found in {datapath}")
```

## scanreader

Initialize a [scanreader](https://millerbrainobservatory.github.io/LBM-CaImAn-Python/scanreader.html) class object.

- The object returned from `read_scan` can be visualized just like a [dask array](https://examples.dask.org/array.html#Create-Random-array)

The resulting class holds **metadata and slice locations** that re-tile the strip when indexed.

```{code-cell} ipython3
data = read_scan(datapath)
data.shape
```

```{code-cell} ipython3
scan = data[:200,0,:,:]
scan.shape
```
