import itertools
import json
import os
from pathlib import Path

from scipy.signal import correlate
import numpy as np
import tifffile

from .utils import listify_index, check_index_type, fill_key
from .multiroi import ROI
try:
    import dask.array as da
    has_dask = True
except:
    has_dask=False

ARRAY_METADATA = ["dtype", "shape", "nbytes", "size"]

IJ_METADATA = ["axes", "photometric", "dtype", "nbytes"]

def apply_slice_to_dask(array, channel_list, frame_list, yslice, xslice):
    return array[channel_list, frame_list, yslice, xslice]

def return_scan_offset(image_in, num_values: int):
    """
    Compute the scan offset correction between interleaved lines or columns in an image.

    This function calculates the scan offset correction by analyzing the cross-correlation
    between interleaved lines or columns of the input image. The cross-correlation peak
    determines the amount of offset between the lines or columns, which is then used to
    correct for any misalignment in the imaging process.

    Parameters:
    -----------
    image_in : ndarray
        2D [Y, X] input image.
    num_values : int
        The number of shifts to apply when comparing shift neighbors. Lower values will increase performance.

    Returns:
    --------
    offset
        The number of pixels to shift every other row (slow-galvo direction) to optimize the correlation between neighboring rows.

    Examples:
    ---------
    >>> img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    >>> return_scan_offset(img, 1)

    Notes:
    ------
    This function assumes that the input image contains interleaved lines or columns that
    need to be analyzed for misalignment. The cross-correlation method is sensitive to
    the similarity in pattern between the interleaved lines or columns. Hence, a strong
    and clear peak in the cross-correlation result indicates a good alignment, and the
    corresponding lag value indicates the amount of misalignment.
    """

    if len(image_in.shape) != 2:
        raise ValueError(f"Input image should be 2 dimensions, not {image_in.shape}")
    if image_in.shape[0] < image_in.shape[1]:
        raise Warning("Image is longer than it is wide. Ensure this image is in the shape [Y, X]")

    num_values = 8

    Iv1 = image_in[::2, :]
    Iv2 = image_in[1::2, :]

    min_len = min(Iv1.shape[0], Iv2.shape[0])
    Iv1 = Iv1[:min_len, :]
    Iv2 = Iv2[:min_len, :]

    buffers = np.zeros((Iv1.shape[0], num_values))

    Iv1 = np.hstack((buffers, Iv1, buffers))
    Iv2 = np.hstack((buffers, Iv2, buffers))

    Iv1 = Iv1.T.ravel(order="F")
    Iv2 = Iv2.T.ravel(order="F")

    # Zero-center and clip negative values to zero
    Iv1 = Iv1 - np.mean(Iv1)
    Iv1[Iv1 < 0] = 0

    Iv2 = Iv2 - np.mean(Iv2)
    Iv2[Iv2 < 0] = 0

    Iv1 = Iv1[:, np.newaxis]
    Iv2 = Iv2[:, np.newaxis]

    r_full = correlate(Iv1[:, 0], Iv2[:, 0], mode="full", method="auto")
    unbiased_scale = len(Iv1) - np.abs(np.arange(-len(Iv1) + 1, len(Iv1)))
    r = r_full / unbiased_scale

    mid_point = len(r) // 2
    lower_bound = mid_point - num_values
    upper_bound = mid_point + num_values + 1
    r = r[lower_bound:upper_bound]
    lags = np.arange(-num_values, num_values + 1)

    correction_index = np.argmax(r)
    return lags[correction_index]

def fix_scan_phase(data_in, offset, dim):
    """
    Corrects the scan phase of the data based on a given offset along a specified dimension.

    Parameters:
    -----------
    dataIn : ndarray
        The input data of shape (sy, sx, sc, sz).
    offset : int
        The amount of offset to correct for.
    dim : int
        Dimension along which to apply the offset.
        1 for vertical (along height/sy), 2 for horizontal (along width/sx).

    Returns:
    --------
    ndarray
        The data with corrected scan phase, of shape (sy, sx, sc, sz).
    """
    sy, sx, sc, sz = data_in.shape
    data_out = None
    if dim == 1:
        if offset > 0:
            data_out = np.zeros((sy, sx + offset, sc, sz))
            data_out[0::2, :sx, :, :] = data_in[0::2, :, :, :]
            data_out[1::2, offset : offset + sx, :, :] = data_in[1::2, :, :, :]
        elif offset < 0:
            offset = abs(offset)
            data_out = np.zeros((sy, sx + offset, sc, sz))  # This initialization is key
            data_out[0::2, offset : offset + sx, :, :] = data_in[0::2, :, :, :]
            data_out[1::2, :sx, :, :] = data_in[1::2, :, :, :]
        else:
            half_offset = int(offset / 2)
            data_out = np.zeros((sy, sx + 2 * half_offset, sc, sz))
            data_out[:, half_offset : half_offset + sx, :, :] = data_in

    elif dim == 2:
        data_out = np.zeros(sy, sx, sc, sz)
        if offset > 0:
            data_out[:, 0::2, :, :] = data_in[:, 0::2, :, :]
            data_out[offset : (offset + sy), 1::2, :, :] = data_in[:, 1::2, :, :]
        elif offset < 0:
            offset = abs(offset)
            data_out[offset : (offset + sy), 0::2, :, :] = data_in[:, 0::2, :, :]
            data_out[:, 1::2, :, :] = data_in[:, 1::2, :, :]
        else:
            data_out[int(offset / 2) : sy + int(offset / 2), :, :, :] = data_in
    return data_out

def clear_zeros(_scan, rmz_threshold=1e-5):
    non_zero_rows = ~np.all(np.abs(_scan) < rmz_threshold, axis=(0, 2))
    non_zero_cols = ~np.all(np.abs(_scan) < rmz_threshold, axis=(0, 1))
    cleaned = _scan[:, non_zero_rows, :]
    return cleaned[:, :, non_zero_cols]

class ScanLBM:

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
        instance.dim_labels = reconstruction_metadata['dim_labels']
        instance.roi_metadata = reconstruction_metadata['roi_metadata']
        instance.si_metadata = reconstruction_metadata['si_metadata']
        instance.ij_metadata = reconstruction_metadata['ij_metadata']
        instance.arr_metadata = reconstruction_metadata['arr_metadata']

        return instance
    def __init__(self, files: list[os.PathLike], fix_scan_phase: bool=True, **kwargs):
        self._frame_slice = None
        self._channel_slice = None
        self._meta = None
        self._trim_x = kwargs.get("trim_roi_x", (0,0))
        self._trim_y = kwargs.get('trim_roi_y', (0,0))

        with tifffile.TiffFile(files[0]) as tiff_file:
            series = tiff_file.series[0]
            scanimage_metadata = tiff_file.scanimage_metadata
            roi_group = scanimage_metadata["RoiGroups"]["imagingRoiGroup"]["rois"]
            pages = tiff_file.pages
            metadata = {
                "roi_info": roi_group,
                "photometric": "minisblack",
                "image_height": pages[0].shape[0],
                "image_width": pages[0].shape[1],
                "num_pages": len(pages),
                "dims": series.dims,
                "axes": series.axes,
                "dtype": series.dtype,
                "is_multifile": series.is_multifile,
                "nbytes": series.nbytes,
                "shape": series.shape,
                "size": series.size,
                "dim_labels": series.sizes,
                "num_rois": len(roi_group),
                "si": scanimage_metadata["FrameData"],
            }
        self.tiff_files = [tifffile.TiffFile(fname) for fname in files]

        self.metadata = metadata
        self.fix_scan_phase = fix_scan_phase

        self.shape = None
        self.dtype = None
        self.axes = None
        self.dims = None
        self.dim_labels = None

        self.roi_metadata = metadata.pop("roi_info")
        self.si_metadata = metadata.pop("si")
        self.ij_metadata = {k: v for k, v in metadata.items() if k in IJ_METADATA}
        self.arr_metadata = {k: v for k, v in metadata.items() if k in ARRAY_METADATA}

        self.axes = self.metadata["axes"]
        self.shape = self.metadata["shape"]
        self.raw_shape = self.metadata["shape"]
        self.dims = self.metadata["dims"]
        self.dtype = self.metadata["dtype"]
        self.dim_labels = self.metadata["dim_labels"]

        self.rois = self._create_rois()
        self.fields = self._create_fields()
        self._join_contiguous_fields()

        if len(self.fields) > 1:
            raise NotImplementedError("Too many fields for an LBM recording.")

        # Track the total height_width of the *tiled* image
        self._width = self.fields[0].width - sum(self.trim_x)
        self._height = self.fields[0].height - sum(self.trim_x)

        # Track where to slice the vertically stacked tiff
        self._xslices = self.fields[0].xslices
        self._yslices = self.fields[0].yslices

        # Track where these will be stored
        self._xslices_out = self.fields[0].output_xslices
        self._yslices_out = self.fields[0].output_yslices

        self.frame_slice = slice(None)
        self.channel_slice = slice(None)

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
            raise ValueError(f"ScanLBM.channel_size should be an integer or slice object, not {type(self.channel_slice)}.")
        for idx, num in enumerate(channels):
            filename = savedir / f'{prepend_str}_plane_{num}.tif'
            data = self[:,channels,:,:]
            tifffile.imwrite(filename,data,bigtiff=True,metadata=combined_metadata)

    def __repr__(self):
        return f"Tiled: {(self.num_frames, self.num_channels, self._height, self._width)} [T,C,Y,X]"

    def __getitem__(self, key):
        ic()
        full_key = fill_key(key, num_dimensions=4)  # key represents the scanfield index
        for i, index in enumerate(full_key):
            check_index_type(i, index)

        self.frame_slice = full_key[0]
        self.channel_slice = full_key[1]
        x_in = full_key[2]
        y_in = full_key[3]

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

        # Over each subfield in field (only one for non-contiguous fields)
        slices = zip(self.yslices, self.xslices, self.yslices_out, self.xslices_out)
        for yslice, xslice, output_yslice, output_xslice in slices:
            # Read the required pages (and slice out the subfield)
            pages = self._read_pages([0], channel_list, frame_list, yslice, xslice)

            y_range = range(output_yslice.start, output_yslice.stop)
            x_range = range(output_xslice.start, output_xslice.stop)
            ys = [[y - output_yslice.start] for y in y_list if y in y_range]
            xs = [x - output_xslice.start for x in x_list if x in x_range]

            item[ :, :, output_yslice, output_xslice] = pages[ :, :, ys, xs]
        return item.squeeze()

    @property
    def height(self):
        """Height of the final tiled image."""
        return self._height - (sum(self.trim_y)*4)

    @property
    def width(self):
        """Width of the final tiled image."""
        return self._width - (sum(self.trim_x)*4)

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
        assert(len(values) == 2)
        self._trim_x = values

    @property
    def trim_y(self):
        return self._trim_y

    @property
    def xslices_out(self):
        new_slice = [slice(v.start+self.trim_x[0], v.stop-self.trim_x[1]) for v in self._xslices_out]

        adjusted_slices = []
        previous_stop = 0
        for s in new_slice:
            length = s.stop - s.start
            adjusted_slices.append(slice(previous_stop, previous_stop + length, None))
            previous_stop += length

        return adjusted_slices

    @property
    def yslices_out(self):
        return [slice(v.start+self.trim_y[0], v.stop+self.trim_y[1]) for v in self._yslices_out]

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
        if self.is_slow_stack:
            num_lines_between_fields = (
                self._page_height + self._num_fly_back_lines
            ) * (self.num_frames * self._num_averaged_frames)
        else:
            num_lines_between_fields = self._page_height + self._num_fly_back_lines
        return int(num_lines_between_fields)

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
        return self.dim_labels.get("time", None)

    @property
    def num_channels(self):
        """Number of channels (planes) in this session."""
        return self.dim_labels.get("channel", None)

    @property
    def num_planes(self):
        """Number of planes (channels) in this session. In multi-ROI sessions, plane is an alias for channel."""
        return self.dim_labels.get("channel", None)

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
        return self.metadata["si"]["SI.hScan2D.uniformSampling"]

    def _generate_reconstruction_metadata(self):
        # Convert the slices to a serializable format
        channel_slice_repr = (self.channel_slice.start, self.channel_slice.stop, self.channel_slice.step) if isinstance(self.channel_slice, slice) else self.channel_slice
        frame_slice_repr = (self.frame_slice.start, self.frame_slice.stop, self.frame_slice.step) if isinstance(self.frame_slice, slice) else self.frame_slice

        # Build the reconstruction metadata
        reconstruction_metadata = {
            'files': [f.filename for f in self.tiff_files],
            'trim_x': self._trim_x,
            'trim_y': self._trim_y,
            'height': self._height,
            'width': self._width,
            'channel_slice': channel_slice_repr,
            'frame_slice': frame_slice_repr,
            'axes': self.axes,
            'dims': self.dims,
            'dim_labels': self.dim_labels,
            'roi_metadata': self.roi_metadata,
            'si_metadata': self.si_metadata,
        }

        # Convert the dictionary to a JSON string for storage in TIFF metadata
        return reconstruction_metadata


# lazy_imread = delayed(tifffile.imread)
# lazy_arrays = [lazy_imread(self.zarr_store)]
# dask_arrays = [da.from_delayed(delayed_reader, shape=self.shape, dtype=self.dtype) for delayed_reader in lazy_arrays]
# arr = dask_arrays[0]
# ysl = self.fields[0].yslices
# xsl = self.fields[0].xslices
# slices = []
# for y,x in zip(ysl,xsl):
#     slices.append(arr[frame_list,channel_list,y,x])
# return da.block(slices).rechunk()
# @property
# def trimx(self):
#     return self._trimx
#
# @trimx.setter
# def trimx(self, value):
#     self._trimx = value
#
# @property
# def trimy(self):
#     return self._trimy
#
# @trimy.setter
# def trimy(self, value):
#     self._trimy = value
#     def save(self, pages, yx, xs, output_ys, output_xs, prev_start):
#         ic()
#         if self.fix_scan_phase:
#             phase = return_scan_offset(
#                 pages[..., ys, xs].transpose((1, 2, 0)).squeeze(), 5
#             )
#             new_page = fix_scan_phase(
#                 pages[..., ys, xs].transpose((2, 3, 1, 0)), phase, 1
#             )
#             pages = new_page[
#                     17:, abs(phase) + 2: len(output_xs) - (abs(phase) + 2), ...
#                     ].transpose((3, 2, 0, 1))
#             nx = pages.shape[-1]
#
#             # Calculate the new slice index for the current strip
#             new_slice = slice(prev_start, prev_start + nx)
#             new_list = listify_index(new_slice, prev_start + nx)
