import itertools
import logging
import json
import os
from pathlib import Path
import time

import numpy as np
import tifffile
import zarr

from .utils import listify_index, check_index_type, fill_key, fix_scan_phase, return_scan_offset
from .multiroi import ROI

import dask.array as da

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ARRAY_METADATA = ["dtype", "shape", "nbytes", "size"]

IJ_METADATA = ["axes", "photometric", "dtype", "nbytes"]

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


class ScanLBM:

    # WIP FO 08/20/24
    @classmethod
    def from_metadata(cls, metadata):
        # Parse the reconstruction metadata
        reconstruction_metadata = json.loads(metadata['reconstruction_metadata'])

        # Instantiate the class using the stored metadata
        instance = cls(
            files=reconstruction_metadata['files'],
            fix_scan_phase=reconstruction_metadata['fix_scan_phase'],
            trim_roi_x=reconstruction_metadata['trim_x'],
            trim_roi_y=reconstruction_metadata['trim_y']
        )

        # Restore other attributes
        instance._channel_slice = reconstruction_metadata['channel_slice']
        instance._frame_slice = reconstruction_metadata['frame_slice']
        instance.metadata = reconstruction_metadata['metadata']
        instance.axes = reconstruction_metadata['axes']
        instance.dims = reconstruction_metadata['dims']
        instance.roi_metadata = reconstruction_metadata['roi_metadata']
        instance.si_metadata = reconstruction_metadata['si_metadata']
        instance.ij_metadata = reconstruction_metadata['ij_metadata']
        instance.arr_metadata = reconstruction_metadata['arr_metadata']

        return instance

    def __init__(self, files: list[os.PathLike], **kwargs):
        logger.debug(f"Initializing scan with files: {[str(x) for x in files]}")
        self.files = files
        if not self.files:
            logger.info("No files given to reader. Returning.")
            return
        self.name = ''
        self._frame_slice = slice(None)
        self._channel_slice = slice(None)
        self._fix_scan_offset = kwargs.get("fix_scan_offset", False)
        self._trim_x = kwargs.get("trim_roi_x", (0, 0))
        self._trim_y = kwargs.get('trim_roi_y', (0, 0))
        self._path = kwargs.get('save_path', Path(files[0]).parent)

        self.tiff_files = [tifffile.TiffFile(fname) for fname in files]

        # with tifffile.TiffFile(self.tiff_files[0]) as tiff_file:
        tiff_file = self.tiff_files[0]
        series = tiff_file.series[0]
        scanimage_metadata = tiff_file.scanimage_metadata
        roi_group = scanimage_metadata["RoiGroups"]["imagingRoiGroup"]["rois"]
        pages = tiff_file.pages
        metadata = {
            "roi_info": roi_group,
            "image_height": pages[0].shape[0],
            "image_width": pages[0].shape[1],
            "num_pages": len(pages),
            "dims": series.dims,
            "ndim": series.ndim,
            "axes": series.axes,
            "dtype": 'uint16',
            "is_multifile": series.is_multifile,
            "nbytes": series.nbytes,
            "shape": series.shape,
            "size": series.size,
            "dim_labels": series.sizes,
            "num_rois": len(roi_group),
            "si": scanimage_metadata["FrameData"],
        }

        # Set Metadata -----
        self.metadata = metadata

        self.roi_metadata = metadata.pop("roi_info")
        self.si_metadata = metadata.pop("si")
        self.ij_metadata = {k: v for k, v in metadata.items() if k in IJ_METADATA}
        self.arr_metadata = {k: v for k, v in metadata.items() if k in ARRAY_METADATA}

        self.raw_shape = self.metadata["shape"]
        self._axes = self.metadata["axes"]
        self._dims = self.metadata["dims"]
        self._ndim = metadata['ndim']
        self._shape = self.metadata["shape"]

        # Create ROIs -----
        self.rois = self._create_rois()
        self.fields = self._create_fields()
        self._join_contiguous_fields()

        if len(self.fields) > 1:
            raise NotImplementedError("Too many fields for an LBM recording.")

        # Initialize dynamic variables -----

        # Adjust height/width for trimmings
        self._width = self.fields[0].width - sum(self.trim_x)
        self._height = self.fields[0].height - sum(self.trim_y)

        # Track where to slice the vertically stacked tiff
        self._xslices = self.fields[0].xslices
        self._yslices = self.fields[0].yslices

        # Track where these will be stored
        self._xslices_out = self.fields[0].output_xslices
        self._yslices_out = self.fields[0].output_yslices

        self._data = da.empty((self.num_frames, self.height, self.width), chunks=CHUNKS)
        self.metadata['fps'] = self.fps

    def save_as_tiff(self, savedir: os.PathLike, metadata=None, prepend_str='extracted'):
        savedir = Path(savedir)
        if not metadata:
            metadata = {}

        # Generate the reconstruction metadata
        reconstruction_metadata = self._generate_reconstruction_metadata()

        # Combine existing metadata with reconstruction metadata
        combined_metadata = {**metadata, 'reconstruction_metadata': reconstruction_metadata}
        if isinstance(self.channel_slice, slice):
            channels = list(range(self.num_channels))[self.channel_slice]
        elif isinstance(self.channel_slice, int):
            channels = [self.channel_slice]
        else:
            raise ValueError(
                f"ScanLBM.channel_size should be an integer or slice object, not {type(self.channel_slice)}.")
        for idx, num in enumerate(channels):
            filename = savedir / f'{prepend_str}_plane_{num}.tif'
            data = self[:, channels, :, :]
            tifffile.imwrite(filename, data, bigtiff=True, metadata=combined_metadata)

    def save_as_zarr(self, savedir: os.PathLike, planes=None, metadata=None, prepend_str='extracted'):
        savedir = Path(savedir)
        if not metadata:
            metadata = self.arr_metadata
        if planes is None:
            planes = list(range(0, self.num_planes))
        if not isinstance(planes, (list, tuple)):
            planes = [planes]

        for idx in planes:
            filename = savedir / f'{prepend_str}_plane_{idx}.zarr'
            da.to_zarr(self[:, idx, :, :], filename)

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
        return f"Tiled shape: {self.shape}"

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
        logger.debug(f'y_list: {y_list}')
        x_list = listify_index(x_in, self.width)
        logger.debug(f'x_list: {x_list}')

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

        start = time.time()

        # Initialize the starting index for the next iteration, only relevant for scan phase
        current_x_start = 0

        # Over each subfield in the field (only one for non-contiguous fields)
        slices = zip(self.yslices, self.xslices, self.yslices_out, self.xslices_out)
        for idx, (yslice, xslice, output_yslice, output_xslice) in enumerate(slices):
            # Read the required pages (and slice out the subfield)
            pages = self._read_pages([0], channel_list, frame_list, yslice, xslice)
            x_range = range(output_xslice.start, output_xslice.stop)  # adjust for offset
            if self.fix_scan_offset:
                phase = return_scan_offset(pages, nvals=4)
                logger.info(f'roi {idx} with optimal phase shift of {phase} px')
                if phase != 0:
                    pages = fix_scan_phase(pages, phase)
                    x_range = range(output_xslice.start, output_xslice.stop - abs(phase))  # adjust for offset
            else:
                phase = 0

            y_range = range(output_yslice.start, output_yslice.stop)
            ys = [[y - output_yslice.start] for y in y_list if y in y_range]
            xs = [x - output_xslice.start for x in x_list if x in x_range]

            # Assign to the output item
            # Instead of using `output_xslice` directly, use `current_x_start` and calculate the width
            x_width = output_xslice.stop - output_xslice.start - abs(phase)
            item[:, :, output_yslice, current_x_start:current_x_start + x_width] = pages[:, :, ys, xs]

            # Update `current_x_start` for the next iteration
            current_x_start += x_width

        print(f'Roi data loaded in {time.time() - start} seconds')

        return item[..., image_slice_y, image_slice_x]

    def lazy_read(self, slices: tuple | list):

        for i, index in enumerate(slices):
            check_index_type(i, index)

        self.frame_slice = slices[0]
        self.channel_slice = slices[1]
        x_in = slices[2]
        y_in = slices[3]

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

        start = time.time()
        # Over each subfield in field (only one for non-contiguous fields)
        slices = zip(self.yslices, self.xslices, self.yslices_out, self.xslices_out)
        for yslice, xslice, output_yslice, output_xslice in slices:
            # Read the required pages (and slice out the subfield)
            pages = self._read_pages([0], channel_list, frame_list, yslice, xslice)
            y_range = range(output_yslice.start, output_yslice.stop)
            x_range = range(output_xslice.start, output_xslice.stop)
            ys = [[y - output_yslice.start] for y in y_list if y in y_range]
            xs = [x - output_xslice.start for x in x_list if x in x_range]
            item[:, :, output_yslice, output_xslice] = pages[:, :, ys, xs]
        logger.info(f'Time to read data: {np.round(time.time() - start, 1)} seconds')
        return item

    @property
    def path(self):
        return self._path

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
        return self._data.dtype

    def init_shape(self):
        frame_list = listify_index(self.frame_slice, self.num_frames)
        channel_list = listify_index(self.channel_slice, self.num_channels)
        return len(frame_list), len(channel_list), self.height, self.width

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
    def yslices_pretrim(self):
        return self._yslices

    @property
    def xslices_pretrim(self):
        return self._xslices

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
        """Create scan rois from the configuration file."""
        roi_infos = self.roi_metadata
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
        """Number of tiff directories in the raw .tiff file. For LBM scans, will be num_planes * num_frames"""
        return self.metadata["num_pages"]

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
        """Number of timepoints in each 2D planar timeseries."""
        return self.raw_shape[0]

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
        """
        Fast stack or slow stack. Fast stacks collect all frames for one slice before moving on.
        """
        return self.si_metadata["SI.hFastZ.enable"]

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

    def _generate_reconstruction_metadata(self):
        # Convert the slices to a serializable format
        channel_slice_repr = (self.channel_slice.start, self.channel_slice.stop, self.channel_slice.step) if isinstance(
            self.channel_slice, slice) else self.channel_slice
        frame_slice_repr = (self.frame_slice.start, self.frame_slice.stop, self.frame_slice.step) if isinstance(
            self.frame_slice, slice) else self.frame_slice

        # Build the reconstruction metadata
        reconstruction_metadata = {
            'files': [f.filename for f in self.tiff_files],
            'trim_x': self._trim_x,
            'trim_y': self._trim_y,
            'height': self._height,
            'width': self._width,
            'channel_slice': channel_slice_repr,
            'frame_slice': frame_slice_repr,
            'roi_metadata': self.roi_metadata,
            'si_metadata': self.si_metadata,
        }

        # Convert the dictionary to a JSON string for storage in TIFF metadata
        return reconstruction_metadata
