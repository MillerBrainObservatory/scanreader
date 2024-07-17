# scanreader Documentation

Python based tiff reader for ScanImage recordings.
Supports scans starting at ScanImage 2016 through the current version (2022).

```{image} https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg
:alt: Doi badge
:target: https://doi.org/10.1038/s41592-021-01239-8
```

```{image} https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg
:alt: Publication Link
:target: https://doi.org/10.1038/s41592-021-01239-8
```

```{image} https://img.shields.io/badge/Repository-black?style=flat-square&logo=github&logoColor=white&link=https%3A%2F%2Fmillerbrainobservatory.github.io%2FLBM-CaImAn-MATLAB%2F
:alt: Repo Link
:target: https://github.com/MillerBrainObservatory/LBM-CaImAn-MATLAB
```

We treat a scan as a collection of recording fields: rectangular planes at a given x, y, z position in the scan recorded in a number of channels during a preset amount of time.
All fields have the same number of channels and number of frames.


## Installation

To install the latest stable version:

```{code-block} bash
pip3 install git+https://github.com/MillerBrainObservatory/scanreader.git
```

This is best done inside a conda environment.

## Usage

You can get relevant metadata without actually reading any large data into memory:

```{code-block} python
import scanreader
scan = scanreader.read_scan('/data/my_scan_*.tif')  # non-mROI
scan = scanreader.read_scan('/data/my_scan_*.tif', dtype=np.float32, join_contiguous=True)
```

Your scan object now contains several useful attributes:

```python
print(scan.version)
print(scan.num_frames)
print(scan.num_channels)
print(scan.num_fields)
```

You can iterate over each ROI/Field of the scan and process them independently:

```python
for field in scan:
    # process field (4-d array: [y_center_coordinate, x_center_coordinate, channels, frames])
    del field  # free memory before next iteration
```

The resulting scan is a **5-d array** [fields, y, x, z-plane, frames]

```python
x = scan[:] # everything 5D
y = scan[:2, :, :, 0, -1000:]  # 5-d array: last 1000 frames of first 2 fields on the first channel
z = scan[1]  # 4-d array: the second field (over all channels and time)
```

You can extract the index of the ROI slices that are saved:

```python
output_xslices = scan.fields[0].output_xslices
```

And use them to trim your image:

```python
# Trim 1 pixel on the left and right edge of each ROI
new_slice = [slice(s.start + 1, s.stop - 1) for s in scan.fields[0].output_xslices]
trim_x = [i for s in new_slice for i in range(s.start, s.stop)]
```

`trim_x` now contains a new slice object you can use to trim your image:

```python

import matplotlib.pyplot as plt

y = scan[:, :, :, 0, 2:15].squeeze()  # untrimmed
y2 = scan[:, :, trim_x, 0, 2:15].squeeze()  # trimmed
plt.imshow(y)  # show untrimmed data
plt.figure()
plt.imshow(y2)  # show trimmed data
plt.show()

```

::::{admonition} A note on *stack acquisition*
:class: dropdown

Each tiff page holds a single depth/channel/frame combination.

For **slow** stacks, channels change first, timeframes change second and slices/depths change last.

    For two channels, three slices, two frames.
        Page:       0   1   2   3   4   5   6   7   8   9   10  11
        Channel:    0   1   0   1   0   1   0   1   0   1   0   1
        Frame:      0   0   1   1   2   2   0   0   1   1   2   2
        Slice:      0   0   0   0   0   0   1   1   1   1   1   1

For scans, channels change first, slices/depths change second and timeframes
change last.

    For two channels, three slices, two frames.
        Page:       0   1   2   3   4   5   6   7   8   9   10  11
        Channel:    0   1   0   1   0   1   0   1   0   1   0   1
        Slice:      0   0   1   1   2   2   0   0   1   1   2   2
        Frame:      0   0   0   0   0   0   1   1   1   1   1   1

::::

Scan objects (returned by `read_scan()`) are iterable and indexable (as shown).
Indexes can be integers, slice objects (:) or lists/tuples/arrays of integers.
It should act like a numpy 5-d array---no boolean indexing, though.

```{toctree}
:includehidden:
:maxdepth: 2
:caption: Contents

apidocs/index.rst
```

## Developer Note

As of this version, `scanreader` relies on [`tifffile`](https://pypi.org/project/tifffile/) to read the underlying tiff files. Reading a scan happens in three stages:

1. `scan = scanreader.read_scan(filename)` will create a list of `tifffile.TiffFile`s, one per each tiff file in the scan. This entails opening a file handle and reading the tags of the first page of each; tags for the rest of pages are ignored (they have the same info).
2. `scan.num_frames`, `scan.shape` or another operation that requires the number of frames in the scan---which includes the first stage of any data loading operation---will need the number of pages in each tiff file. `tifffile` was designed for files with pages of varying shapes so it iterates over each page looking for its offset (number of bytes from the start of the file until the very first byte of the page), which it saves to use for reading. After this operation, it knows the number of pages per file.
3. Once the file has been opened and the offset to each page has been calculated we can load the actual data. We load each page sequentially and take care of reformatting them to match the desired output.

This reader and documentation are based off of  is based on a previous [version](https://github.com/atlab/scanreader) developed by [atlab](https://github.com/atlab/).

Some of the older scans have been removed for general cleanliness. These can be reimplemented by cherry-picking the commit. See documentation on `git reflog` to find the commits you want and `git cherry-pick` to apply changes that were introduced by those commits.
