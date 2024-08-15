"""
scans.py

1. If it (method or property) is shared among all subclasses it should be here, either implemented or as an abstract
method. Even if private (e.g. _page_height) 2. If one subclass needs to overwrite it, then erase it here and
implement them in the subclasses (this applies for now that I only have two subclasses). If code needs to be shared
add it as a private method here. 3. If it is not in every subclass, it should not be here.

"""
import itertools
import re
import typing
from ctypes import ARRAY
from typing import overload, Any

import dask.array as da
import numpy as np
import zarr
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit
from tifffile import TiffFile, tifffile
from tifffile.tifffile import matlabstr2py

import scanreader.core
from .utils import listify_index, check_index_is_in_bounds, check_index_type, fill_key, compute
from .multiroi import ROI
from .exceptions import FieldDimensionMismatch

ARRAY_METADATA = ['dtype', 'shape', 'nbytes', 'size']

IJ_METADATA = ['axes', 'photometric', 'dtype', 'nbytes']

def apply_slice_to_dask(array, channel_list, frame_list, yslice, xslice):
    return array[channel_list, frame_list, yslice, xslice]

class BaseScan:
    """
    Properties and methods shared among all scan versions.

    Scan objects are a collection of recording fields: rectangular planes at a given x_center_coordinate, y_center_coordinate,
    z position in the scan recorded in a number of channels during a preset amount of
    time. All fields have the same number of channels and number of frames.
    Scan objects are:
        indexable: scan[field, y_center_coordinate, x_center_coordinate, channel, frame] works as long as the fields' spatial
            dimensions (y_center_coordinate, x_center_coordinate) match.
        iterable: 'for field in scan:' iterates over all fields (4-d array) in the scan.

    Examples:
        scan.version                ScanImage version of the scan.
        scan[:, :, :3, :, :1000]    5-d numpy array with the first 1000 frames of the
            first 3 fields (if x_center_coordinate, y_center_coordinate dimensions match).
        for field in scan:          generates 4-d numpy arrays ([y_center_coordinate, x_center_coordinate, channels, frames]).

    Note:
        We use the word 'frames' as in video frames, i.e., number of timesteps the scan
        was recorded; ScanImage uses frames to refer to slices/scanning depths in the
        scan.
    """
    """
    Interface rules:
        If it (method or property) is shared among all subclasses it should be here,
            either implemented or as an abstract method. Even if private (e.g. _page_height)
        If one subclass needs to overwrite it, then erase it here and implement them in
            the subclasses (this applies for now that I only have two subclasses). If code
            needs to be shared add it as a private method here.
        If it is not in every subclass, it should not be here.
    """

    def __init__(self):
        self.filenames = None
        self.dtype = None
        self.header = ''

    @property
    def tiff_files(self):
        if self._tiff_files is None:
            self._tiff_files = [TiffFile(filename) for filename in self.filenames]
        return self._tiff_files

    @tiff_files.deleter
    def tiff_files(self):
        if self._tiff_files is not None:
            for tiff_file in self._tiff_files:
                tiff_file.close()
            self._tiff_files = None

    @property
    def version(self):
        match = re.search(r"SI.?\.VERSION_MAJOR = '?(?P<version>[^\s']*)'?", self.header)
        version = match.group('version') if match else None
        return version

    #https://github.com/SFB1089/scanreader/blob/7625ad397728aff57c32bd4ab38b35383440c52d/scanreader/scans.py#L536
    @property  #added property to get the power percent
    def power_percent(self):
        match = re.search(r'hBeams\.powers = (?P<power>.*)', self.header)
        power = float(match.group('power')) if match else None
        return power

    @property
    def is_slow_stack(self):
        """ True if fastZ is disabled. All frames for one slice are recorded first before
        moving to the next slice."""
        match = re.search(r'hFastZ\.enable = (?P<is_slow>.*)', self.header)
        is_slow_stack = (match.group('is_slow') in ['false', '0']) if match else None
        return is_slow_stack

    @property
    def is_multiROI(self):
        """Only True if mroiEnable exists (2016b and up) and is set to True."""
        match = re.search(r'hRoiManager\.mroiEnable = (?P<is_multiROI>.)', self.header)
        is_multiROI = (match.group('is_multiROI') == '1') if match else False
        return is_multiROI

    @property
    def num_channels(self):
        match = re.search(r'hChannels\.channelSave = (?P<channels>.*)', self.header)
        if match:
            channels = matlabstr2py(match.group('channels'))
            num_channels = len(channels) if isinstance(channels, list) else 1
        else:
            num_channels = None
        return num_channels

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
    def num_scanning_depths(self):
        if self.is_slow_stack:
            """ Number of scanning depths actually recorded in this stack."""
            num_scanning_depths = self._num_pages / (self.num_channels * self.num_frames)
            num_scanning_depths = int(num_scanning_depths)  # discard last slice if incomplete
        else:
            num_scanning_depths = len(self.requested_scanning_depths)
        return num_scanning_depths

    @property
    def scanning_depths(self):
        return self.requested_scanning_depths[:self.num_scanning_depths]

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
    def is_bidirectional(self):
        match = re.search(r'hScan2D\.bidirectional = (?P<is_bidirectional>.*)', self.header)
        is_bidirectional = (match.group('is_bidirectional') == 'true') if match else False
        return is_bidirectional

    @property
    def scanner_frequency(self):
        match = re.search(r'hScan2D\.scannerFrequency = (?P<scanner_freq>.*)', self.header)
        scanner_frequency = float(match.group('scanner_freq')) if match else None
        return scanner_frequency

    @property
    def seconds_per_line(self):
        if np.isnan(self.scanner_frequency):
            match = re.search(r'hRoiManager\.linePeriod = (?P<secs_per_line>.*)', self.header)
            seconds_per_line = float(match.group('secs_per_line')) if match else None
        else:
            scanner_period = 1 / self.scanner_frequency  # secs for mirror to return to initial position
            seconds_per_line = scanner_period / 2 if self.is_bidirectional else scanner_period
        return seconds_per_line

    @property
    def _num_pages(self):
        num_pages = sum([len(tiff_file.pages) for tiff_file in self.tiff_files])
        return num_pages

    @property
    def _page_height(self):
        return self.tiff_files[0].pages[0].imagelength

    @property
    def _page_width(self):
        return self.tiff_files[0].pages[0].imagewidth

    @property
    def _num_averaged_frames(self):
        """ Number of requested frames are averaged to form one saved frame. """
        match = re.search(r'hScan2D\.logAverageFactor = (?P<num_avg_frames>.*)', self.header)
        num_averaged_frames = int(float(match.group('num_avg_frames'))) if match else None
        return num_averaged_frames

    @property
    def num_fields(self):
        raise NotImplementedError('Subclasses of BaseScan must implement this property')

    @property
    def field_depths(self):
        raise NotImplementedError('Subclasses of BaseScan must implement this property')

    # Properties from here on are not strictly necessary
    @property
    def fps(self):
        match = re.search(r'hRoiManager\.scanVolumeRate = (?P<fps>.*)', self.header)
        fps = float(match.group('fps')) if match else None
        return fps

    @property
    def spatial_fill_fraction(self):
        match = re.search(r'hScan2D\.fillFractionSpatial = (?P<spatial_ff>.*)', self.header)
        spatial_fill_fraction = float(match.group('spatial_ff')) if match else None
        return spatial_fill_fraction

    @property
    def temporal_fill_fraction(self):
        match = re.search(r'hScan2D\.fillFractionTemporal = (?P<temporal_ff>.*)', self.header)
        temporal_fill_fraction = float(match.group('temporal_ff')) if match else None
        return temporal_fill_fraction

    @property
    def scanner_type(self):
        match = re.search(r"hScan2D\.scannerType = '(?P<scanner_type>.*)'", self.header)
        scanner_type = match.group('scanner_type') if match else None
        return scanner_type

    @property
    def motor_position_at_zero(self):
        """ Motor position (x_center_coordinate, y_center_coordinate and z in microns) corresponding to the scan's (0, 0, 0)
        point. For non-multiroi scans, (x_center_coordinate=0, y_center_coordinate=0) marks the center of the FOV."""
        match = re.search(r'hMotors\.motorPosition = (?P<motor_position>.*)', self.header)
        motor_position = matlabstr2py(match.group('motor_position'))[:3] if match else None
        return motor_position

    @property
    def initial_secondary_z(self):
        """ Initial position in z (microns) of the secondary motor (if any)."""
        match = re.search(r'hMotors\.motorPosition = (?P<motor_position>.*)', self.header)
        if match:
            motor_position = matlabstr2py(match.group('motor_position'))
            secondary_z = motor_position[3] if len(motor_position) > 3 else None
        else:
            secondary_z = None
        return secondary_z

    @property
    def _initial_frame_number(self):
        match = re.search(r'\sframeNumbers = (?P<frame_number>.*)', self.header)
        initial_frame_number = int(match.group('frame_number')) if match else None
        return initial_frame_number


    @property
    def _num_fly_back_lines(self):
        """ Lines/mirror cycles that it takes to move from one depth to the next."""
        match = re.search(r'hScan2D\.flybackTimePerFrame = (?P<fly_back_seconds>.*)',
                          self.header)
        if match:
            fly_back_seconds = float(match.group('fly_back_seconds'))
            num_fly_back_lines = self._seconds_to_lines(fly_back_seconds)
        else:
            num_fly_back_lines = None
        return num_fly_back_lines

    @property
    def _num_lines_between_fields(self):
        """ Lines/mirror cycles scanned from the start of one field to the start of the
        next. """
        if self.is_slow_stack:
            num_lines_between_fields = ((self._page_height + self._num_fly_back_lines) *
                                        (self.num_frames * self._num_averaged_frames))
        else:
            num_lines_between_fields = self._page_height + self._num_fly_back_lines
        return int(num_lines_between_fields)

    @property
    def is_slow_stack_with_fastZ(self):
        raise NotImplementedError('Subclasses of BaseScan must implement this property')

    @property
    def field_offsets(self):
        raise NotImplementedError('Subclasses of BaseScan must implement this property')

    def read_data(self, filenames, dtype):
        """ Set self.header, self.filenames and self.dtype. Data is read lazily when needed.

        Args:
            filenames: List of strings. Tiff filenames.
            dtype: Data type of the output array.
        """
        self.filenames = filenames  # set filenames
        self.dtype = dtype  # set dtype of read data
        self.header = '{}\n{}'.format(self.tiff_files[0].pages[0].description,
                                      self.tiff_files[0].pages[0].software)  # set header (ScanImage metadata)

    def __array__(self):
        return self[:]

    def __str__(self):
        msg = '{}\n{}\n{}'.format(type(self), '*' * 80, self.header, '*' * 80)
        return msg

    def __len__(self):
        return 0 if self.num_fields is None else self.num_fields

    def __getitem__(self, key):
        """ Index scans by field, y_center_coordinate, x_center_coordinate, channels, frames. Supports integer, slice and
        array/tuple/list of integers as indices."""
        raise NotImplementedError('Subclasses of BaseScan must implement this method')

    def __iter__(self):
        class ScanIterator:
            """ Iterator for Scan objects."""

            def __init__(self, scan):
                self.scan = scan
                self.next_field = 0

            def __next__(self):
                if self.next_field < self.scan.num_fields:
                    field = self.scan[self.next_field]
                    self.next_field += 1
                else:
                    raise StopIteration
                return field

        return ScanIterator(self)

    def _read_pages(self, slice_list, channel_list, frame_list, yslice=slice(None),
                    xslice=slice(None)):
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
            frame_step = self.num_channels * self.num_scanning_depths
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

            # Read from this tiff file (if needed)
            if len(file_indices) > 0:
                # this line looks a bit ugly but is memory efficient. Do not separate
                pages[global_indices] = tiff_file.asarray(key=file_indices)[..., yslice, xslice]
            start_page += len(tiff_file.pages)

        # Reshape the pages into (slices, y_center_coordinate, x_center_coordinate, channels, frames)
        new_shape = [len(frame_list), len(slice_list), len(channel_list), out_height, out_width]
        pages = pages.reshape(new_shape).transpose([1, 3, 4, 2, 0])

        return pages

    def _seconds_to_lines(self, seconds):
        """ Compute how many lines would be scanned in the given amount of seconds."""
        num_lines = int(np.ceil(seconds / self.seconds_per_line))
        if self.is_bidirectional:
            # scanning starts at one end of the image so num_lines needs to be even
            num_lines += (num_lines % 2)

        return num_lines

    def _compute_offsets(self, field_height, start_line):
        """
        Computes the time offsets at which a given field was recorded.

        Computes the time delay at which each pixel was recorded using the start of the
        scan as zero. It first creates an image with the number of lines scanned until
        that point and then uses self.seconds_per_line to  transform it into seconds.

        :param int field_height: Height of the field.
        :param int start_line: Line at which this field starts.

        :returns: A field_height x_center_coordinate page_width mask with time offsets in seconds.
        """
        # Compute offsets within a line (negligible if seconds_per_line is small)
        max_angle = (np.pi / 2) * self.temporal_fill_fraction
        line_angles = np.linspace(-max_angle, max_angle, self._page_width + 2)[1:-1]
        line_offsets = (np.sin(line_angles) + 1) / 2

        # Compute offsets for entire field
        field_offsets = np.expand_dims(np.arange(0, field_height), -1) + line_offsets
        if self.is_bidirectional:  # odd lines scanned from left to right
            field_offsets[1::2] = field_offsets[1::2] - line_offsets + (1 - line_offsets)

        # Transform offsets from line counts to seconds
        field_offsets = (field_offsets + start_line) * self.seconds_per_line

        return field_offsets

class ScanLegacy(BaseScan):
    """ Scan versions 4 and below. Not implemented."""

    def __init__(self):
        raise NotImplementedError('Legacy scans not supported')

class NewerScan:
    """ Shared features among all newer scans. """

    @property
    def is_slow_stack_with_fastZ(self):
        match = re.search(r'hStackManager\.slowStackWithFastZ = (?P<slow_with_fastZ>.*)',
                          self.header)
        slow_with_fastZ = (match.group('slow_with_fastZ') in ['true', '1']) if match else None
        return slow_with_fastZ

class ScanMultiROI(NewerScan, BaseScan):
    """An extension of ScanImage v5 that manages multiROI data (output from mesoscope).

     Attributes:
         join_contiguous: A bool. Whether contiguous fields are joined into one.
         rois: List of ROI objects (defined in multiroi.py)
         fields: List of Field objects (defined in multiroi.py)
     """

    def __init__(self, join_contiguous):
        super().__init__()
        self.join_contiguous = join_contiguous
        self.rois = None
        self.fields = None

    @property
    def num_fields(self):
        return len(self.fields)

    @property
    def num_rois(self):
        return len(self.rois)

    @property
    def field_heights(self):
        return [field.height for field in self.fields]

    @property
    def field_widths(self):
        return [field.width for field in self.fields]

    @property
    def field_depths(self):
        return [field.depth for field in self.fields]

    @property
    def field_slices(self):
        return [field.slice_id for field in self.fields]

    @property
    def field_rois(self):
        return [field.roi_ids for field in self.fields]

    @property
    def field_masks(self):
        return [field.roi_mask for field in self.fields]

    @property
    def field_offsets(self):
        return [field.offset_mask for field in self.fields]

    @property
    def field_heights_in_microns(self):
        field_heights_in_degrees = [field.height_in_degrees for field in self.fields]
        return [round(self._degrees_to_microns(deg)) for deg in field_heights_in_degrees]

    @property
    def field_widths_in_microns(self):
        field_widths_in_degrees = [field.width_in_degrees for field in self.fields]
        return [round(self._degrees_to_microns(deg)) for deg in field_widths_in_degrees]

    @property
    def _num_fly_to_lines(self):
        """ Number of lines recorded in the tiff page while flying to a different field,
        i.e., distance between fields in the tiff page."""
        match = re.search(r'hScan2D\.flytoTimePerScanfield = (?P<fly_to_seconds>.*)',
                          self.header)
        if match:
            fly_to_seconds = float(match.group('fly_to_seconds'))
            num_fly_to_lines = self._seconds_to_lines(fly_to_seconds)
        else:
            num_fly_to_lines = None
        return num_fly_to_lines

    def _degrees_to_microns(self, degrees):
        """ Convert scan angle degrees to microns using the objective resolution."""
        match = re.search(r'objectiveResolution = (?P<deg2um_factor>.*)', self.header)
        microns = (degrees * float(match.group('deg2um_factor'))) if match else None
        return microns

    def _microns_to_decrees(self, microns):
        """ Convert microns to scan angle degrees using the objective resolution."""
        match = re.search(r'objectiveResolution = (?P<deg2um_factor>.*)', self.header)
        degrees = (microns / float(match.group('deg2um_factor'))) if match else None
        return degrees

    def read_data(self, filenames, dtype):
        """ Set the header, create rois and fields (joining them if necessary)."""
        super().read_data(filenames, dtype)
        self.rois = self._create_rois()
        self.fields = self._create_fields()
        if self.join_contiguous:
            self._join_contiguous_fields()

    def _create_fields(self):
        """ Go over each slice depth and each roi generating the scanned fields. """
        fields = []
        previous_lines = 0
        for slice_id, scanning_depth in enumerate(self.scanning_depths):
            next_line_in_page = 0  # each slice is one tiff page
            for roi_id, roi in enumerate(self.rois):
                new_field = roi.get_field_at(scanning_depth)

                if new_field is not None:
                    if next_line_in_page + new_field.height > self._page_height:
                        error_msg = ('Overestimated number of fly to lines ({}) at '
                                     'scanning depth {}'.format(self._num_fly_to_lines,
                                                                scanning_depth))
                        raise RuntimeError(error_msg)

                    # Set xslice and yslice (from where in the page to cut it)
                    new_field.yslices = [slice(next_line_in_page, next_line_in_page
                                               + new_field.height)]
                    new_field.xslices = [slice(0, new_field.width)]

                    # Set output xslice and yslice (where to paste it in output)
                    new_field.output_yslices = [slice(0, new_field.height)]
                    new_field.output_xslices = [slice(0, new_field.width)]

                    # Set slice and roi id
                    new_field.slice_id = slice_id
                    new_field.roi_ids = [roi_id]

                    # Set timing offsets
                    offsets = self._compute_offsets(new_field.height, previous_lines +
                                                    next_line_in_page)
                    new_field.offsets = [offsets]

                    # Compute next starting y_center_coordinate
                    next_line_in_page += new_field.height + self._num_fly_to_lines

                    # Add field to fields
                    fields.append(new_field)

            # Accumulate overall number of scanned lines
            previous_lines += self._num_lines_between_fields

        return fields

    def _join_contiguous_fields(self):
        """ In each scanning depth, join fields that are contiguous.

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

    def __getitem__(self, key):
        # Fill key to size 5 (raises IndexError if more than 5)
        full_key = fill_key(key, num_dimensions=5)  # key represents the scanfield index

        # Check index types are valid
        for i, index in enumerate(full_key):
            check_index_type(i, index)

        # Check each dimension is in bounds
        check_index_is_in_bounds(0, full_key[0], self.num_fields)
        for field_id in listify_index(full_key[0], self.num_fields):
            check_index_is_in_bounds(1, full_key[1], self.field_heights[field_id])
            check_index_is_in_bounds(2, full_key[2], self.field_widths[field_id])
        check_index_is_in_bounds(3, full_key[3], self.num_channels)
        check_index_is_in_bounds(4, full_key[4], self.num_frames)

        # Get fields, channels and frames as lists
        field_list = listify_index(full_key[0], self.num_fields)
        y_lists = [listify_index(full_key[1], self.field_heights[field_id]) for
                   field_id in field_list]
        x_lists = [listify_index(full_key[2], self.field_widths[field_id]) for
                   field_id in field_list]
        channel_list = listify_index(full_key[3], self.num_channels)
        frame_list = listify_index(full_key[4], self.num_frames)

        # Edge case when slice index gives 0 elements or index is empty list, e.g., scan[10:0], scan[[]]
        if [] in [field_list, *y_lists, *x_lists, channel_list, frame_list]:
            return np.empty(0)

        # Check output heights and widths match for all fields
        if not all(len(y_list) == len(y_lists[0]) for y_list in y_lists):
            raise FieldDimensionMismatch('Image heights for all fields do not match')
        if not all(len(x_list) == len(x_lists[0]) for x_list in x_lists):
            raise FieldDimensionMismatch('Image widths for all fields do not match')

        # Over each field, read required pages and slice
        item = np.empty([len(field_list), len(y_lists[0]), len(x_lists[0]),
                         len(channel_list), len(frame_list)], dtype=self.dtype)
        for i, (field_id, y_list, x_list) in enumerate(zip(field_list, y_lists, x_lists)):
            field = self.fields[field_id]

            # Over each subfield in field (only one for non-contiguous fields)
            slices = zip(field.yslices, field.xslices, field.output_yslices, field.output_xslices)
            for yslice, xslice, output_yslice, output_xslice in slices:
                # Read the required pages (and slice out the subfield)
                pages = self._read_pages([field.slice_id], channel_list, frame_list,
                                         yslice, xslice)

                # Get x_center_coordinate, y_center_coordinate indices that need to be accessed in this subfield
                y_range = range(output_yslice.start, output_yslice.stop)
                x_range = range(output_xslice.start, output_xslice.stop)
                ys = [[y - output_yslice.start] for y in y_list if y in y_range]
                xs = [x - output_xslice.start for x in x_list if x in x_range]
                output_ys = [[index] for index, y in enumerate(y_list) if y in y_range]
                output_xs = [index for index, x in enumerate(x_list) if x in x_range]
                # ys as nested lists are needed for numpy to slice them correctly

                # Index pages in y_center_coordinate, x_center_coordinate
                item[i, output_ys, output_xs] = pages[0, ys, xs]

        # If original index was an integer, delete that axis (as in numpy indexing)
        squeeze_dims = [i for i, index in enumerate(full_key) if np.issubdtype(type(index),
                                                                               np.signedinteger)]
        item = np.squeeze(item, axis=tuple(squeeze_dims))
        return item

class ScanLBM(ScanMultiROI, BaseScan):
    def __init__(self, paths, metadata, join_contiguous=True, **kwargs):
        self._trimx = None
        self._trimy = None
        self._slicey = None
        self._slicex = None
        self._frame_slice = None
        self._channel_slice = None
        self._meta = None

        # store original dims
        self.shape = None
        self.dtype = None
        self.axes = None
        self.dims = None
        self.dim_labels = None

        # new dims as properties
        self.shape = None
        self.dtype = None
        self.axes = None
        self.dims = None
        self.dim_labels = None

        self._tiff_files = [TiffFile(filename) for filename in paths]
        self.header = kwargs.get('header', None)
        super().__init__(join_contiguous)

        self.roi_metadata = metadata.pop('roi_info')
        self.si_metadata = metadata.pop("si")
        self.ij_metadata = {k: v for k,v in metadata.items() if k in IJ_METADATA}
        self.arr_metadata = {k: v for k,v in metadata.items() if k in ARRAY_METADATA}

        self.metadata = metadata

        self.axes = self.metadata['axes']
        self.shape = self.metadata['shape']
        self.raw_shape = self.metadata['shape']
        self.dims = self.metadata['dims']
        self.dtype = self.metadata['dtype']
        self.dim_labels = self.metadata['dim_labels']

        self.rois = self._create_rois()
        self.fields = self._create_fields()

        if self.join_contiguous:
            self._join_contiguous_fields()

        if len(self.fields) > 1:
            raise NotImplementedError

        ## Field Slices
        self._xslices_out = self.fields[0].output_xslices
        self._yslices_out = self.fields[0].output_yslices
        self._xslices = self.fields[0].xslices
        self._yslices = self.fields[0].yslices

    def __repr__(self):
        return f'{self.get_new_dims()}'


    @property
    def num_scanning_depths(self):
        return 1

    @property
    def _num_pages(self):
        return self.metadata['num_pages']

    @property
    def _page_height(self):
        return self.metadata['image_height']

    @property
    def _page_width(self):
        return self.metadata['image_width']

    @property
    def num_frames(self):
        return self.dim_labels.get('time', None)

    @property
    def num_channels(self):
        return self.dim_labels.get('channel', None)

    @property
    def num_planes(self):
        return self.dim_labels.get('channel', None)

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
        """If ScanImage 2016 or newer. This should be True"""
        # This check is due to us not knowing which metadata value to trust for the scan rate.
        if not self.si_metadata["SI.hRoiManager.scanFrameRate"] == self.si_metadata["SI.hRoiManager.scanVolumeRate"]:
            raise ValueError("ScanImage metadata used for frame rate is inconsistent. Double check values for SI.hRoiManager.scanFrameRate and SI.hRoiManager.scanVolumeRate")
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

    @property
    def _num_fly_back_lines(self):
        """ Lines/mirror cycles scanned from the start of one field to the start of the next. """
        return int(self.si_metadata["SI.hScan2D.flytoTimePerScanfield"] / float(self.si_metadata["SI.hRoiManager.linePeriod"]))

    @property
    def _num_lines_between_fields(self):
        """ Lines/mirror cycles scanned from the start of one field to the start of the
        next. """
        if self.is_slow_stack:
            num_lines_between_fields = ((self._page_height + self._num_fly_back_lines) *
                                        (self.num_frames * self._num_averaged_frames))
        else:
            num_lines_between_fields = self._page_height + self._num_fly_back_lines
        return int(num_lines_between_fields)

    def _degrees_to_microns(self, degrees):
        return int(degrees * self.objective_resolution)

    def _microns_to_degrees(self, microns):
        return int(microns / self.objective_resolution)

    @property
    def xslices_out(self):
        return self._xslices_out

    @xslices_out.setter
    def xslices_out(self, value):
        self._xslices_out = value

    @property
    def yslices_out(self):
        return self._yslices_out

    @yslices_out.setter
    def yslices_out(self, value):
        self._yslices_out = value

    @property
    def yslices(self):
        return self._yslices

    @yslices.setter
    def yslices(self, value):
        self._yslices = value

    @property
    def xslices(self):
        return self._xslices

    @xslices.setter
    def xslices(self, value):
        self._xslices = value

    @property
    def trimx(self):
        return self._trimx

    @trimx.setter
    def trimx(self, value):
        self._trimx = value

    @property
    def trimy(self):
        return self._trimy

    @trimy.setter
    def trimy(self, value):
        self._trimy = value

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

    def get_new_dims(self):
        field = self.fields[0]

        # Get fields, channels and frames as lists
        frame_list = listify_index(self.frame_slice, self.num_frames)
        channel_list = listify_index(self.channel_slice, self.num_channels)

        y_lists = listify_index(self.trimy, field.height)
        x_lists = listify_index(self.trimx, field.width)

        new_dims = [len(frame_list), len(channel_list), len(y_lists), len(x_lists)]
        return new_dims

    def __getitem__(self, key):

        field = self.fields[0]
        full_key = fill_key(key, num_dimensions=4)  # key represents the scanfield index
        for i, index in enumerate(full_key):
            check_index_type(i, index)

        self.frame_slice = full_key[0]
        self.channel_slice = full_key[1]
        self.xslice = full_key[-2]
        self.yslice = full_key[-1]

        frame_list = listify_index(self.frame_slice, self.num_frames)
        channel_list = listify_index(self.channel_slice, self.num_channels)
        y_lists = listify_index(self.yslice, field.height)
        x_lists = listify_index(self.xslice, field.width)

        if [] in [*y_lists, *x_lists, channel_list, frame_list]:
            return np.empty(0)

        # Over each field, read required pages and slice
        # item = np.empty([len(y_lists), len(x_lists),
        #                  len(channel_list), len(frame_list)], dtype=self.dtype)

        # cast to TCYX
        item = np.empty([
            len(frame_list),
            len(channel_list),
            len(y_lists),
            len(x_lists),
        ], dtype=self.dtype)

        # Over each subfield in field (only one for non-contiguous fields)
        slices = zip(self.yslices, self.xslices, field.output_yslices, field.output_xslices)
        for yslice, xslice, output_yslice, output_xslice in slices:
            # Read the required pages (and slice out the subfield)
            pages = self._read_pages([0], channel_list, frame_list,
                                     yslice, xslice)

            y_range = range(output_yslice.start, output_yslice.stop)
            x_range = range(output_xslice.start, output_xslice.stop)
            ys = [[y - output_yslice.start] for y in y_lists if y in y_range]
            xs = [x - output_xslice.start for x in x_lists if x in x_range]
            output_ys = [[index] for index, y in enumerate(y_lists) if y in y_range]
            output_xs = [index for index, x in enumerate(x_lists) if x in x_range]

            item[..., output_ys, output_xs] = pages[..., ys, xs]

        item = np.squeeze(item)
        return item

    def _read_pages(self, slice_list, channel_list, frame_list, yslice=slice(None), xslice=slice(None)):
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
            frame_step = self.num_channels * self.num_scanning_depths
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

            # Read from this tiff file (if needed)
            if len(file_indices) > 0:
                # this line looks a bit ugly but is memory efficient. Do not separate
                pages[global_indices] = tiff_file.asarray(key=file_indices)[..., yslice, xslice]
            start_page += len(tiff_file.pages)

        new_shape = [len(frame_list), len(channel_list), out_height, out_width]
        return pages.reshape(new_shape)

    def _create_fields(self):
        """ Go over each slice depth and each roi generating the scanned fields. """
        fields = []
        previous_lines = 0
        next_line_in_page = 0  # each slice is one tiff page
        for roi_id, roi in enumerate(self.rois):
            new_field = roi.get_field_at(0)

            if new_field is not None:
                if next_line_in_page + new_field.height > self._page_height:
                    raise RuntimeError(f'Overestimated number of fly to lines ({self._num_fly_to_lines})')

                # Set xslice and yslice (from where in the page to cut it)
                new_field.yslices = [slice(next_line_in_page, next_line_in_page + new_field.height)]
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

    def _create_rois(self):
        """Create scan rois from the configuration file. """
        roi_infos = self.roi_metadata
        rois = [ROI(roi_info) for roi_info in roi_infos]
        return rois

    def _join_contiguous_fields(self):
        """ In each scanning depth, join fields that are contiguous.

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

    def _degrees_to_microns(self, degrees):
        """ Convert scan angle degrees to microns using the objective resolution."""
        return degrees * self.objective_resolution

    def _microns_to_decrees(self, microns):
        """ Convert microns to scan angle degrees using the objective resolution."""
        return microns / self.objective_resolution

    def _init_zarr(self, channel_list=slice(None), frame_list=slice(None), yslice=slice(None), xslice=slice(None), **kwargs):
        return 2
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

