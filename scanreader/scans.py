from __future__ import annotations
import itertools
import functools
import logging
import os
import re
from pathlib import Path
import time
import numpy as np
import tifffile
from tifffile.tifffile import matlabstr2py
import zarr

import scanreader
from .utils import listify_index, check_index_type, fill_key, fix_scan_phase, return_scan_offset
from .multiroi import ROI

import dask.array as da

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ARRAY_METADATA = ["dtype", "shape", "nbytes", "size"]

CHUNKS = {0: 'auto', 1: -1, 2: -1}

# https://brainglobe.info/documentation/brainglobe-atlasapi/adding-a-new-atlas.html
BRAINGLOBE_STRUCTURE_TEMPLATE = {
    "acronym": "VIS",  # shortened name of the region
    "id": 3,  # region id
    "name": "visual cortex",  # full region name
    "structure_id_path": [1, 2, 3],  # path to the structure in the structures hierarchy, up to current id
    "rgb_triplet": [255, 255, 255],
    # default color for visualizing the region, feel free to leave white or randomize it
}


def make_json_serializable(obj):
    """Convert metadata to JSON serializable format."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def get_metadata(file: os.PathLike):
    if not file:
        return None

    tiff_file = tifffile.TiffFile(file)
    meta = tiff_file.scanimage_metadata
    str_f = tiff_file.filename

    if 'plane_' in str_f and meta is None:
        raise ValueError(f"No metadata found in {str_f}. This appears to be a processed file. "
                         f"The called function operates only on raw scanimage tiff files.")

    si = meta.get('FrameData', {})
    if not si:
        print(f"No FrameData found in {file}.")
        return None

    series = tiff_file.series[0]
    pages = tiff_file.pages

    # Extract ROI and imaging metadata
    roi_group = meta["RoiGroups"]["imagingRoiGroup"]["rois"]

    num_rois = len(roi_group)
    num_planes = len(si["SI.hChannels.channelSave"])
    scanfields = roi_group[0]["scanfields"]  # assuming single ROI scanfield configuration

    # ROI metadata
    center_xy = scanfields["centerXY"]
    size_xy = scanfields["sizeXY"]
    num_pixel_xy = scanfields["pixelResolutionXY"]

    # TIFF header-derived metadata
    sample_format = pages[0].dtype.name
    objective_resolution = si["SI.objectiveResolution"]
    frame_rate = si["SI.hRoiManager.scanFrameRate"]

    # Field-of-view calculations
    fov_x = round(objective_resolution * size_xy[0])  # adjusted for number of ROIs
    fov_y = round(objective_resolution * size_xy[1])
    fov_xy = (fov_x, fov_y)

    # Pixel resolution calculation
    pixel_resolution = (fov_x / num_pixel_xy[0], fov_y / num_pixel_xy[1])

    # Assembling metadata
    return {
        "image_height": pages[0].shape[0],
        "image_width": pages[0].shape[1],
        "num_pages": len(pages),
        "dims": series.dims,
        "ndim": series.ndim,
        "dtype": 'uint16',
        "is_multifile": series.is_multifile,
        "nbytes": series.nbytes,
        "size": series.size,
        "dim_labels": series.sizes,
        "shape": series.shape,
        "num_planes": num_planes,
        "num_rois": num_rois,
        "num_frames": len(pages) / num_planes,
        "frame_rate": frame_rate,
        "fov": fov_xy,  # in microns
        "pixel_resolution": np.round(pixel_resolution, 2),
        "roi_width_px": num_pixel_xy[0],
        "roi_height_px": num_pixel_xy[1],
        "sample_format": sample_format,
        "num_lines_between_scanfields": round(si["SI.hScan2D.flytoTimePerScanfield"] / si["SI.hRoiManager.linePeriod"]),
        "center_xy": center_xy,
        "line_period": si["SI.hRoiManager.linePeriod"],
        "size_xy": size_xy,
        "objective_resolution": objective_resolution,
        "si": si,
        "roi_info": roi_group
    }


class ScanLBM:
    def __init__(self, files: list[os.PathLike], **kwargs):
        logger.info(f"Initializing scan with files: {[str(x) for x in files]}")
        self.files = files
        if not self.files:
            logger.info("No files given to reader. Returning.")
            return
        self._frame_slice = slice(None)
        self._channel_slice = slice(None)
        self._trim_x = kwargs.get("trim_roi_x", (0, 0))
        self._trim_y = kwargs.get("trim_roi_y", (0, 0))
        self.metadata = kwargs.get("metadata", None)
        tiffs = Path(files[0]).parent.glob("*.tif*")
        self.tiff_files = [tifffile.TiffFile(fname) for fname in tiffs]
        self.header = '{}\n{}'.format(self.tiff_files[0].pages[0].description,
                                      self.tiff_files[0].pages[0].software)  # set header (ScanImage metadata)
        self._dtype = np.uint16

        if not self.metadata:
            self.metadata = get_metadata(files[0])

        self.roi_metadata = self.metadata.pop("roi_info")
        self.si_metadata = self.metadata.pop("si")
        self.arr_metadata = {k: v for k, v in self.metadata.items() if k in ARRAY_METADATA}

        self.raw_shape = self.metadata["shape"]
        self._dims = self.metadata["dims"]
        self._ndim = self.metadata['ndim']
        self._shape = self.metadata["shape"]

        # Create ROIs -----
        self.rois = self._create_rois()
        self.fields = self._create_fields()
        self._join_contiguous_fields()

        # Adjust height/width for trimmings
        self._width = self.fields[0].width - sum(self.trim_x)
        self._height = self.fields[0].height - sum(self.trim_y)

        # Track where to slice the vertically stacked tiff
        self._xslices = self.fields[0].xslices
        self._yslices = self.fields[0].yslices

        # Track where these will be stored
        self._xslices_out = self.fields[0].output_xslices
        self._yslices_out = self.fields[0].output_yslices

        self._data = da.empty((self.num_frames, self.num_channels, self.height, self.width), chunks=CHUNKS)
        self.metadata['fps'] = self.fps
        self._fix_scan_offset = kwargs.get('fix_scan_offset', False)

    def save_as(
            self,
            savedir: os.PathLike,
            planes=None,
            frames=None,
            metadata=None,
            overwrite=True,
            by_roi=False,
            ext='.tiff',
            assemble=False
    ):
        savedir = Path(savedir)
        if planes is None:
            planes = list(range(self.num_planes))
        elif not isinstance(planes, (list, tuple)):
            planes = [planes]
        if frames is None:
            frames = list(range(self.num_frames))
        if metadata:
            self.metadata.update(metadata)

        self.metadata = make_json_serializable(self.metadata)
        if not savedir.exists():
            logger.debug(f"Creating directory: {savedir}")
            savedir.mkdir(parents=True)
        if assemble:
            self._save_assembled(savedir, planes, frames, overwrite, ext=ext, by_roi=by_roi)
        else:
            self._save_data(savedir, planes, frames, overwrite, ext, by_roi)

    def _save_assembled(self, path, planes, frames, overwrite, ext, by_roi=False):
        if by_roi:
            raise NotImplementedError("--assemble and --roi both given as arguments, Cannot assemble ROI's and save "
                                      "by ROI.")

        logger.info(f"Saving assembled data to {path}")

        x_in, y_in = slice(None), slice(None)

        y_list = listify_index(y_in, self.height)
        x_list = listify_index(x_in, self.width)

        if [] in [*y_list, *x_list, planes, frames]:
            return np.empty(0)

        save_start = time.time()
        for p in planes:
            p_start = time.time()

            # cast to TCYX
            item = da.empty(
                [
                    len(frames),
                    len(y_list),
                    len(x_list),
                ],
                dtype=self.dtype,
                chunks=({0: 'auto', 1: -1, 2: -1})  # chunk along time axis
            )
            logger.info(f"Processing plane {p + 1} of {len(planes)}")

            # Initialize the starting index for the next iteration, only relevant for scan phase
            current_x_start = 0

            # Over each subfield in the field (only one for non-contiguous fields)
            slices = zip(self.yslices, self.xslices, self.yslices_out, self.xslices_out)
            for idx, (yslice, xslice, output_yslice, output_xslice) in enumerate(slices):
                field_start = time.time()
                logger.info(f"Processing subfield {idx + 1} of {len(self.yslices)}")

                # Read the required pages (and slice out the subfield)
                pages = self._read_pages([0], [p], frames, yslice, xslice)
                pages_read = time.time() - field_start
                logger.debug(f"--- Pages read in {pages_read:.2f} seconds.")

                x_range = range(output_xslice.start, output_xslice.stop)  # adjust for offset
                y_range = range(output_yslice.start, output_yslice.stop)

                ys = [[y - output_yslice.start] for y in y_list if y in y_range]
                xs = [x - output_xslice.start for x in x_list if x in x_range]

                x_width = output_xslice.stop - output_xslice.start
                item_start = time.time()
                item[:, output_yslice, current_x_start:current_x_start + x_width] = np.squeeze(pages[:, :, ys, xs])
                item_appended = time.time() - item_start
                logger.debug(f"--- ROI appended in {item_appended:.2f} seconds.")

                current_x_start += x_width
            field_end = time.time() - p_start
            logger.debug(f"--- Field processed in {field_end:.2f} seconds.")
            self._write_tiff(path, f'assembled_plane_{p + 1}', item, metadata=self.metadata, overwrite=overwrite)
            p_end = time.time() - p_start
            logger.debug(f"--- Plane {p + 1} processed in {p_end:.2f} seconds.")
        logger.debug(f"--- Assembled data saved in {time.time() - save_start:.2f} seconds.")

    def _save_data(self, path, planes, frames, overwrite, ext, by_roi):
        p = None

        path.mkdir(parents=True, exist_ok=True)
        print(f'Planes: {planes}')

        file_writer = self._get_file_writer(ext, overwrite)
        roi_slices = list(zip(self.yslices, self.xslices, self.rois))

        if by_roi:
            # When saving by ROI
            outer_iter = enumerate(roi_slices)
            outer_label = 'ROI'
            inner_label = 'Plane'
        else:
            # When saving by Plane
            outer_iter = enumerate(planes)
            outer_label = 'Plane'
            inner_label = 'ROI'

        for outer_idx, outer_val in outer_iter:
            print(f'-- Saving {outer_label} {outer_idx + 1} --')

            if by_roi:
                # Outer loop over ROIs
                slce_y, slce_x, roi = outer_val
                subdir = path / f'roi_{outer_idx + 1}'
                inner_iter = planes  # Inner loop over planes
            else:
                # Outer loop over planes
                p = outer_val
                subdir = path / f'plane_{p + 1}'
                inner_iter = roi_slices  # Inner loop over ROIs

            subdir.mkdir(parents=True, exist_ok=True)

            for inner_idx, inner_val in enumerate(inner_iter):
                if by_roi:
                    # Inner loop over planes
                    p = inner_val
                    name = f'plane_{p + 1}'
                else:
                    # Inner loop over ROIs
                    slce_y, slce_x, roi = inner_val
                    name = f'roi_{inner_idx + 1}'

                print(f'-- Reading pages: {outer_label} {outer_idx + 1}, {inner_label} {inner_idx + 1} --')
                t_start = time.time()
                pages = self._read_pages([0], [p], frames, slce_y, slce_x)
                t_end = time.time() - t_start
                logger.info(f"TiffFile pages read in {t_end:.2f} seconds.")
                file_writer(subdir, name, pages, roi.roi_info if roi else None)

    def _get_file_writer(self, ext, overwrite):
        if ext in ['.tif', '.tiff']:
            return functools.partial(self._write_tiff, overwrite=overwrite)
        elif ext == '.zarr':
            return functools.partial(self._write_zarr, overwrite=overwrite)
        else:
            raise ValueError(f'Unsupported file extension: {ext}')

    def _write_tiff(self, path, name, data, metadata=None, overwrite=True):
        filename = Path(path / f'{name}.tiff')
        if filename.exists() and not overwrite:
            logger.warning(
                f'File already exists: {filename}. To overwrite, set overwrite=True (--overwrite in command line)')
            return
        logger.info(f"Writing {filename}")
        t_write = time.time()
        tifffile.imwrite(filename, data, bigtiff=True, metadata=metadata, photometric='minisblack', )
        t_write_end = time.time() - t_write
        logger.info(f"Data written in {t_write_end:.2f} seconds.")

    def _write_zarr(self, path, name, data, metadata=None, overwrite=True):
        store = zarr.DirectoryStore(path)
        root = zarr.group(store, overwrite=overwrite)
        ds = root.create_dataset(name=name, data=data, overwrite=True)
        if metadata:
            ds.attrs['metadata'] = metadata

    def __repr__(self):
        return self.data.__repr__()

    @property
    def fix_scan_offset(self):
        return self._fix_scan_offset

    @fix_scan_offset.setter
    def fix_scan_offset(self, value: bool):
        assert isinstance(value, bool)
        self._fix_scan_offset = value

    @property
    def data(self):
        return self._data

    def __str__(self):
        return f"Tiled shape: {self.shape} with axes {self.dims} and dtype {self.dtype}."

    def __getitem__(self, key):
        full_key = fill_key(key, num_dimensions=4)  # key represents the scanfield index
        for i, index in enumerate(full_key):
            check_index_type(i, index)

        self.frame_slice = full_key[0]
        self.channel_slice = full_key[1]
        x_in, y_in = slice(None), slice(None)
        image_slice_x = full_key[2]
        image_slice_y = full_key[3]

        frame_list = listify_index(self.frame_slice, self.num_frames)
        channel_list = listify_index(self.channel_slice, self.num_channels)
        y_list = listify_index(y_in, self.height)
        x_list = listify_index(x_in, self.width)

        if [] in [*y_list, *x_list, channel_list, frame_list]:
            return np.empty(0)

        # cast to TCYX
        item = da.empty(
            [
                len(frame_list),
                len(channel_list),
                len(y_list),
                len(x_list),
            ],
            dtype=self.dtype,
            chunks=({0: 'auto', 1: 'auto', 2: -1, 3: -1})
        )

        # Initialize the starting index for the next iteration, only relevant for scan phase
        current_x_start = 0

        # Over each subfield in the field (only one for non-contiguous fields)
        slices = zip(self.yslices, self.xslices, self.yslices_out, self.xslices_out)
        for idx, (yslice, xslice, output_yslice, output_xslice) in enumerate(slices):
            # Read the required pages (and slice out the subfield)
            pages = self._read_pages([0], channel_list, frame_list, yslice, xslice)

            x_range = range(output_xslice.start, output_xslice.stop)  # adjust for offset
            y_range = range(output_yslice.start, output_yslice.stop)

            ys = [[y - output_yslice.start] for y in y_list if y in y_range]
            xs = [x - output_xslice.start for x in x_list if x in x_range]

            # Assign to the output item
            # Instead of using `output_xslice` directly, use `current_x_start` and calculate the width
            x_width = output_xslice.stop - output_xslice.start
            item[:, :, output_yslice, current_x_start:current_x_start + x_width] = pages[:, :, ys, xs]

            # Update `current_x_start` for the next iteration
            current_x_start += x_width
        return item[..., image_slice_y, image_slice_x]

    @property
    def height(self):
        """Height of the final tiled image."""
        return self._height - sum(self.trim_y)

    @property
    def width(self):
        """Width of the final tiled image."""
        return self._width - (sum(self.trim_x) * 4)

    @property
    def ndim(self):
        """Shape of the final tiled image."""
        return self._data.ndim

    @property
    def dtype(self):
        """Datatype of the final tiled image."""
        return self._dtype

    @property
    def shape(self):
        """Shape of the final tiled image."""
        return self._data.shape

    @property
    def trim_x(self):
        """
        Number of px to trim on the (left_edge, right_edge)
        """
        return self._trim_x

    @trim_x.setter
    def trim_x(self, values):
        """
        Number of px to trim on the (left_edge, right_edge)
        """
        assert (len(values) == 2)
        self._trim_x = values

    @property
    def trim_y(self):
        return self._trim_y

    @trim_y.setter
    def trim_y(self, values):
        assert (len(values) == 2)
        self._trim_y = values

    @property
    def xslices_out(self):
        new_slice = [slice(v.start + self.trim_x[0], v.stop - self.trim_x[1]) for v in self._xslices_out]

        adjusted_slices = []
        previous_stop = 0
        for s in new_slice:
            length = s.stop - s.start
            adjusted_slices.append(slice(previous_stop, previous_stop + length, None))
            previous_stop += length

        return adjusted_slices

    @property
    def yslices_out(self):
        new_slice = [slice(v.start + self.trim_y[0], v.stop - self.trim_y[1]) for v in self._yslices_out]
        return [slice(0, v.stop - v.start) for v in new_slice]

    @property
    def yslices(self):
        return [slice(v.start + self.trim_y[0], v.stop - self.trim_y[1]) for v in self._yslices]

    @property
    def xslices(self):
        return [slice(v.start + self.trim_x[0], v.stop - self.trim_x[1]) for v in self._xslices]

    @property
    def frame_slice(self):
        return self._frame_slice

    @frame_slice.setter
    def frame_slice(self, value):
        self._frame_slice = value

    @property
    def channel_slice(self):
        return self._channel_slice

    @channel_slice.setter
    def channel_slice(self, value):
        self._channel_slice = value

    def _read_pages(
            self,
            slice_list,
            channel_list,
            frame_list,
            yslice=slice(None),
            xslice=slice(None),
    ):
        """
        Reads the tiff pages with the content of each slice, channel, frame
        combination and slices them in the y_center_coordinate, x_center_coordinate dimension.

        Each tiff page holds a single depth/channel/frame combination.

        For slow stacks, channels change first, timeframes change second and slices/depths change last.
        Example:
            For two channels, three slices, two frames.
                Page:       0   1   2   3   4   5   6   7   8   9   10  11
                Channel:    0   1   0   1   0   1   0   1   0   1   0   1
                Frame:      0   0   1   1   2   2   0   0   1   1   2   2
                Slice:      0   0   0   0   0   0   1   1   1   1   1   1

        For fast-stack aquisition scans, channels change first, slices/depths change second and timeframes
        change last.
        Example:
            For two channels, three slices, two frames.
                Page:       0   1   2   3   4   5   6   7   8   9   10  11
                Channel:    0   1   0   1   0   1   0   1   0   1   0   1
                Slice:      0   0   1   1   2   2   0   0   1   1   2   2
                Frame:      0   0   0   0   0   0   1   1   1   1   1   1


        Parameters
        ----------
        slice_list: List of integers. Slices to read.
        channel_list: List of integers. Channels to read.
        frame_list: List of integers. Frames to read
        yslice: Slice object. How to slice the pages in the y_center_coordinate axis.
        xslice: Slice object. How to slice the pages in the x_center_coordinate axis.

        Returns
        -------
        pages: np.ndarray
        A 5-D array (num_slices, output_height, output_width, num_channels, num_frames).

        Required pages reshaped to have slice, channel and frame as different
        dimensions. Channel, slice and frame order received in the input lists are
        respected; for instance, if slice_list = [1, 0, 2, 0], then the first
        dimension will have four slices: [1, 0, 2, 0].

        Notes
        -----

        We use slices in y_center_coordinate, x_center_coordinate for memory efficiency, If lists were passed another copy
        of the pages will be needed coming up to 3x the amount of data we actually
        want to read (the output array, the read pages and the list-sliced pages).
        Slices limit this to 2x (output array and read pages which are sliced in place).

        """
        # Compute pages to load from tiff files
        if self.is_slow_stack:
            frame_step = self.num_channels
            slice_step = self.num_channels * self.num_frames
        else:
            slice_step = self.num_channels
            frame_step = self.num_channels * 1
        pages_to_read = []
        for frame in frame_list:
            for slice_ in slice_list:
                for channel in channel_list:
                    new_page = frame * frame_step + slice_ * slice_step + channel
                    pages_to_read.append(new_page)

        # Compute output dimensions
        out_height = len(listify_index(yslice, self._page_height))
        out_width = len(listify_index(xslice, self._page_width))

        # Read pages
        pages = np.empty([len(pages_to_read), out_height, out_width], dtype=self.dtype)
        start_page = 0
        for tiff_file in self.tiff_files:

            # Get indices in this tiff file and in output array
            final_page_in_file = start_page + len(tiff_file.pages)
            is_page_in_file = lambda page: page in range(start_page, final_page_in_file)
            pages_in_file = filter(is_page_in_file, pages_to_read)
            file_indices = [page - start_page for page in pages_in_file]
            global_indices = [is_page_in_file(page) for page in pages_to_read]

            # Read from this tiff file
            if len(file_indices) > 0:
                # this line looks a bit ugly but is memory efficient. Do not separate
                pages[global_indices] = tiff_file.asarray(key=file_indices)[
                    ..., yslice, xslice
                ]
            start_page += len(tiff_file.pages)

        new_shape = [len(frame_list), len(channel_list), out_height, out_width]
        return pages.reshape(new_shape)

    @property
    def _num_fly_back_lines(self):
        """Lines/mirror cycles scanned from the start of one field to the start of the next."""
        return int(
            self.si_metadata["SI.hScan2D.flytoTimePerScanfield"]
            / float(self.si_metadata["SI.hRoiManager.linePeriod"])
        )

    @property
    def _num_lines_between_fields(self):
        """Lines/mirror cycles scanned from the start of one field to the start of the
        next."""
        return int(self._page_height + self._num_fly_back_lines)

    def _create_rois(self):
        """Create scan rois from the configuration file. """
        roi_infos = self.roi_metadata
        roi_infos = roi_infos if isinstance(roi_infos, list) else [roi_infos]
        roi_infos = list(filter(lambda r: isinstance(r['zs'], (int, float, list)),
                                roi_infos))  # discard empty/malformed ROIs

        rois = [ROI(roi_info) for roi_info in roi_infos]
        return rois

    def _join_contiguous_fields(self):
        """In each scanning depth, join fields that are contiguous.

        Fields are considered contiguous if they appear next to each other and have the
        same size in their touching axis. Process is iterative: it tries to join each
        field with the remaining ones (checked in order); at the first union it will break
        and restart the process at the first field. When two fields are joined, it deletes
        the one appearing last and modifies info such as field height, field width and
        slices in the one appearing first.

        Any rectangular area in the scan formed by the union of two or more fields which
        have been joined will be treated as a single field after this operation.
        """
        two_fields_were_joined = True
        while two_fields_were_joined:  # repeat until no fields were joined
            two_fields_were_joined = False

            for field1, field2 in itertools.combinations(self.fields, 2):

                if field1.is_contiguous_to(field2):
                    # Change info in field 1 to reflect the union
                    field1.join_with(field2)

                    # Delete field 2 in self.fields
                    self.fields.remove(field2)

                    # Restart join contiguous search (at while)
                    two_fields_were_joined = True
                    break

    def _create_fields(self):
        """Go over each slice depth and each roi generating the scanned fields."""
        fields = []
        previous_lines = 0
        next_line_in_page = 0  # each slice is one tiff page
        for roi_id, roi in enumerate(self.rois):
            new_field = roi.get_field_at(0)

            if new_field is not None:
                if next_line_in_page + new_field.height > self._page_height:
                    raise RuntimeError(
                        f"Overestimated number of fly to lines ({self._num_fly_to_lines})"
                    )

                # Set xslice and yslice (from where in the page to cut it)
                new_field.yslices = [
                    slice(next_line_in_page, next_line_in_page + new_field.height)
                ]
                new_field.xslices = [slice(0, new_field.width)]

                # Set output xslice and yslice (where to paste it in output)
                new_field.output_yslices = [slice(0, new_field.height)]
                new_field.output_xslices = [slice(0, new_field.width)]

                # Set slice and roi id
                new_field.roi_ids = [roi_id]

                # Compute next starting y_center_coordinate
                next_line_in_page += new_field.height + self._num_fly_to_lines

                # Add field to fields
                fields.append(new_field)

        # Accumulate overall number of scanned lines
        previous_lines += self._num_lines_between_fields

        return fields

    @property
    def _num_pages(self):
        num_pages = sum([len(tiff_file.pages) for tiff_file in self.tiff_files])
        return num_pages

    @property
    def _page_height(self):
        """Width of the raw .tiff in the fast-galvo scan direction (y)."""
        return self.metadata["image_height"]

    @property
    def _page_width(self):
        """Width of the raw .tiff in the slow-galvo scan direction (x)."""
        return self.metadata["image_width"]

    @property
    def num_frames(self):
        """ Each tiff page is an image at a given channel, scanning depth combination."""
        if self.is_slow_stack:
            num_frames = min(self.num_requested_frames / self._num_averaged_frames,
                             self._num_pages / self.num_channels)  # finished in the first slice
        else:
            num_frames = self._num_pages / (self.num_channels * self.num_scanning_depths)
        num_frames = int(num_frames)  # discard last frame if incomplete
        return num_frames

    @property
    def num_requested_frames(self):
        if self.is_slow_stack:
            match = re.search(r'hStackManager\.framesPerSlice = (?P<num_frames>.*)',
                              self.header)
        else:
            match = re.search(r'hFastZ\.numVolumes = (?P<num_frames>.*)', self.header)
        num_requested_frames = int(1e9 if match.group('num_frames') == 'Inf' else
                                   float(match.group('num_frames'))) if match else None
        return num_requested_frames

    @property
    def num_scanning_depths(self):
        if self.is_slow_stack:
            """ Number of scanning depths actually recorded in this stack."""
            num_scanning_depths = self._num_pages / (self.num_channels * self.num_frames)
            num_scanning_depths = int(num_scanning_depths)  # discard last slice if incomplete
        else:
            num_scanning_depths = len(self.requested_scanning_depths)
        return num_scanning_depths

    @property
    def _num_averaged_frames(self):
        """ Number of requested frames are averaged to form one saved frame. """
        match = re.search(r'hScan2D\.logAverageFactor = (?P<num_avg_frames>.*)', self.header)
        num_averaged_frames = int(float(match.group('num_avg_frames'))) if match else None
        return num_averaged_frames

    @property
    def requested_scanning_depths(self):
        match = re.search(r'hStackManager\.zs = (?P<zs>.*)', self.header)
        if match:
            zs = matlabstr2py(match.group('zs'))
            scanning_depths = zs if isinstance(zs, list) else [zs]
        else:
            scanning_depths = None
        return scanning_depths

    @property
    def num_channels(self):
        """Number of channels (planes) in this session."""
        return self.raw_shape[1]

    @property
    def num_planes(self):
        """Number of planes (channels) in this session. In multi-ROI sessions, plane is an alias for channel."""
        return self.raw_shape[1]

    @property
    def objective_resolution(self):
        return self.si_metadata["SI.objectiveResolution"]

    @property
    def _num_fly_to_lines(self):
        return int(
            self.si_metadata["SI.hScan2D.flytoTimePerScanfield"]
            / float(self.si_metadata["SI.hRoiManager.linePeriod"])
        )

    @property
    def is_slow_stack(self):
        """ True if fastZ is disabled. All frames for one slice are recorded first before
        moving to the next slice."""
        match = re.search(r'hFastZ\.enable = (?P<is_slow>.*)', self.header)
        is_slow_stack = (match.group('is_slow') in ['false', '0']) if match else None
        return is_slow_stack

    @property
    def multi_roi(self):
        """If ScanImage 2016 or newer. This should be True"""
        return self.si_metadata["SI.hRoiManager.mroiEnable"]

    @property
    def fps(self):
        """
        Frame rate of each planar timeseries.
        """
        # This check is due to us not knowing which metadata value to trust for the scan rate.
        if (
                not self.si_metadata["SI.hRoiManager.scanFrameRate"]
                    == self.si_metadata["SI.hRoiManager.scanVolumeRate"]
        ):
            raise ValueError(
                "ScanImage metadata used for frame rate is inconsistent. Double check values for SI.hRoiManager.scanFrameRate and SI.hRoiManager.scanVolumeRate"
            )
        return self.si_metadata["SI.hRoiManager.scanFrameRate"]

    @property
    def bidirectional(self):
        """If ScanImage 2016 or newer. This should be True"""
        # This check is due to us not knowing which metadata value to trust for the scan rate.
        return self.si_metadata["SI.hScan2D.bidirectional"]

    @property
    def uniform_sampling(self):
        """If ScanImage 2016 or newer. This should be True"""
        # This check is due to us not knowing which metadata value to trust for the scan rate.
        return self.si_metadata["SI.hScan2D.uniformSampling"]

    @dtype.setter
    def dtype(self, value):
        self._dtype = value
