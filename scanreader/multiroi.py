"""multiroi.py: Holds ROI, Scanfield, and Field classes"""
import numpy as np


class ROI:
    """
    Holds ROI (Region of Interest) information and computes a two-dimensional scanfield at a specified depth.

    ScanImage defines an ROI as the interpolation between a set of scanfields. For more details, see their documentation.

    Parameters
    ----------
    roi_info : dict
        A dictionary containing the definition of the ROI extracted from the TIFF header.

    Attributes
    ----------
    roi_info : dict
        Stores the original ROI information from the TIFF file.
    _scanfields : list of Scanfield
        Cached list of scanfields that form this ROI.
    """

    def __init__(self, roi_info):
        """
        Initialize the ROI with information from the TIFF header.

        Parameters
        ----------
        roi_info : dict
            The ROI definition as extracted from the TIFF header.

        """
        self.roi_info = roi_info
        self._scanfields = None

    @property
    def scanfields(self):
        if self._scanfields is None:
            self._scanfields = self._create_scanfields()
        return self._scanfields

    @property
    def is_discrete_plane_mode_on(self):
        return bool(self.roi_info["discretePlaneMode"])

    def _create_scanfields(self):
        """
        Create all the scanfields that form this ROI. Each roi can have multiple scanfields, each with its own depth.

        .. note::
            This method is called only once, when the scanfields are first accessed.
            This namespace only has access to frame-specific metadata, so it cannot convert degrees to microns
            as the objective resolution is attached to the .tiff header:

            `header['SI.objectiveResolution']`
        """
        # Get scanfield configuration info
        scanfield_infos = self.roi_info["scanfields"]
        if not isinstance(scanfield_infos, list):
            scanfield_infos = [scanfield_infos]  # make list if single scanfield

        # Get scanfield depths
        # scanfield_depths = self.roi_info["zs"]
        scanfield_depths = [0]
        if not isinstance(scanfield_depths, list):
            scanfield_depths = [scanfield_depths]

        scanfields = []
        for scanfield_info, scanfield_depth in zip(scanfield_infos, scanfield_depths):
            # if scanfield_info['enable']: # this is always 1 even if ROI is disabled

            # tuple with the number of pixels in (x_center_coordinate, y_center_coordinate)
            width, height = scanfield_info["pixelResolutionXY"]
            # tuple with the center of the ROI in deg(x_center_coordinate, y_center_coordinate)
            xcenter, ycenter = scanfield_info["centerXY"]
            # tuple with the size of the ROI in deg(x_center_coordinate, y_center_coordinate)
            size_in_x, size_in_y = scanfield_info["sizeXY"]

            # Create scanfield
            new_scanfield = Scanfield(
                height=height,
                width=width,
                depth=scanfield_depth,
                y_center_coordinate=ycenter,
                x_center_coordinate=xcenter,
                height_in_degrees=size_in_y,
                width_in_degrees=size_in_x,
            )
            scanfields.append(new_scanfield)

        # Sort them by depth (to ease interpolation)
        scanfields = sorted(scanfields, key=lambda scanfield: scanfield.depth)

        return scanfields

    def get_field_at(self, scanning_depth):
        """
        Interpolates between the ROI scanfields to generate the 2-d field at the
        desired depth.

        Args:
            scanning_depth: An integer. Depth at which we want to obtain the field.

        Returns:
            field. A Field object, the desired field at scanning_depth

        Warning:
            Does not work for rotated ROIs.
            If there were more than one scanfield at the same depth, it will only consider
                the one defined last.
        """
        field = None

        if self.is_discrete_plane_mode_on:  # only check at each scanfield depth
            for scanfield in self.scanfields:
                if scanning_depth == scanfield.depth:
                    field = scanfield.as_field()
        else:
            if len(self.scanfields) == 1:  # single scanfield extending from -inf to inf
                field = self.scanfields[0].as_field()
                field.depth = scanning_depth

            else:  # interpolate between scanfields
                scanfield_depths = [sf.depth for sf in self.scanfields]
                valid_range = range(min(scanfield_depths), max(scanfield_depths) + 1)
                if scanning_depth in valid_range:
                    field = Field()

                    scanfield_heights = [sf.height for sf in self.scanfields]
                    field.height = np.interp(
                        scanning_depth, scanfield_depths, scanfield_heights
                    )
                    field.height = (
                        int(round(field.height / 2)) * 2
                    )  # round to the closest even

                    scanfield_widths = [sf.width for sf in self.scanfields]

                    # The x_center_coordinate - coordinate sequence is expected to be increasing, but this is not explicitly enforced.
                    # if the sequence `xp` is non-increasing, interpolation results are meaningless.
                    # assert np.all(np.diff(xp) > 0)  # check that the depths are in increasing order

                    field.width = np.interp(
                        scanning_depth, scanfield_depths, scanfield_widths
                    )
                    field.width = (
                        int(round(field.width / 2)) * 2
                    )  # round to the closest even

                    field.depth = scanning_depth

                    scanfield_ys = [sf.y_center_coordinate for sf in self.scanfields]
                    field.y_center_coordinate = np.interp(
                        scanning_depth, scanfield_depths, scanfield_ys
                    )

                    scanfield_xs = [sf.x_center_coordinate for sf in self.scanfields]
                    field.x_center_coordinate = np.interp(
                        scanning_depth, scanfield_depths, scanfield_xs
                    )

                    scanfield_heights = [sf.height_in_degrees for sf in self.scanfields]
                    field.height_in_degrees = np.interp(
                        scanning_depth, scanfield_depths, scanfield_heights
                    )

                    scanfield_widths = [sf.width_in_degrees for sf in self.scanfields]
                    field.width_in_degrees = np.interp(
                        scanning_depth, scanfield_depths, scanfield_widths
                    )

        return field


class Scanfield:
    """
    Container for information about a scanfield, which defines a part of an ROI.

    Attributes
    ----------
    height : int
        Height of the field in pixels.
    width : int
        Width of the field in pixels.
    depth : float
        Depth at which this field was recorded, in microns relative to the absolute Z-coordinate.
    y_center_coordinate : float
        Y-coordinate of the center of the field in scan angle degrees.
    x_center_coordinate : float
        X-coordinate of the center of the field in scan angle degrees.
    height_in_degrees : float
        Height of the field in degrees of the scan angle.
    width_in_degrees : float
        Width of the field in degrees of the scan angle.

    """

    def __init__(
        self,
        height=None,
        width=None,
        depth=None,
        y_center_coordinate=None,
        x_center_coordinate=None,
        height_in_degrees=None,
        width_in_degrees=None,
    ):
        """
        Initialize a scanfield with dimensions and position.

        Parameters
        ----------
        height : int, optional
            Height of the field in pixels.
        width : int, optional
            Width of the field in pixels.
        depth : float, optional
            Depth of the field in microns.
        y_center_coordinate : float, optional
            Y-coordinate of the center in scan angle degrees.
        x_center_coordinate : float, optional
            X-coordinate of the center in scan angle degrees.
        height_in_degrees : float, optional
            Height of the field in scan angle degrees.
        width_in_degrees : float, optional
            Width of the field in scan angle degrees.
        """
        self.height = height
        self.width = width
        self.depth = depth
        self.y_center_coordinate = y_center_coordinate
        self.x_center_coordinate = x_center_coordinate
        self.height_in_degrees = height_in_degrees
        self.width_in_degrees = width_in_degrees

    def as_field(self):
        """
        Convert this Scanfield into a Field object.

        Returns
        -------
        Field
            A new Field object with attributes copied from this Scanfield.
        """
        return Field(
            height=self.height,
            width=self.width,
            depth=self.depth,
            y_center_coordinate=self.y_center_coordinate,
            x_center_coordinate=self.x_center_coordinate,
            height_in_degrees=self.height_in_degrees,
            width_in_degrees=self.width_in_degrees,
        )


class Field(Scanfield):
    """
    Represents a two-dimensional scanning plane, an extension of a scanfield with additional functionalities.

    Inherits all attributes from Scanfield and adds slicing information for integration into larger datasets.

    Attributes
    ----------
    yslices : list of slice
        Slices defining how to cut the page in the Y-axis to retrieve this field.
    xslices : list of slice
        Slices defining how to cut the page in the X-axis to retrieve this field.
    output_yslices : list of slice
        Slices defining where to paste this field in the output.
    output_xslices : list of slice
        Slices defining where to paste this field in the output.
    slice_id : int
        Index of the slice in the scan to which this field belongs.
    roi_ids : list of int
        List of ROI indices to which each subfield belongs.
    offsets : list of float
        Time offsets per pixel in seconds for each subfield.

    Example
    -------
    Assuming a setup where each page represents a different depth and each ROI is a different region,
    you can extract and manipulate specific fields as follows::

        field = roi.get_field_at(depth)
        output_field[field.output_yslices, field.output_xslices] = page[field.yslices, field.xslices]

    Notes
    -----
    When a field is formed by joining two or more subfields (via join_contiguous), the slice lists hold
    multiple slices representing where each subfield will be taken from the page and inserted into the (joint) output
    field. - For non-contiguous fields, each slice list has a single slice. - The attributes `height`, `width`,
    `x_center_coordinate`, `y_center_coordinate`, `height_in_degrees`, and `width_in_degrees` are adjusted
    accordingly when fields are joined.
    """

    def __init__(
        self,
        height=None,
        width=None,
        depth=None,
        y_center_coordinate=None,
        x_center_coordinate=None,
        height_in_degrees=None,
        width_in_degrees=None,
        yslices=None,
        xslices=None,
        output_yslices=None,
        output_xslices=None,
        slice_id=None,
        roi_ids=None,
        offsets=(),
    ):
        """
        Initialize a field with dimensions and position.

        Parameters
        ----------
        height : int, optional
            Height of the field in pixels.
        width : int, optional
            Width of the field in pixels.
        depth : float, optional
            Depth of the field in microns.
        y_center_coordinate : float, optional
            Y-coordinate of the center in scan angle degrees.
        x_center_coordinate : float, optional
            X-coordinate of the center in scan angle degrees.
        height_in_degrees : float, optional
            Height of the field in scan angle degrees.
        width_in_degrees : float, optional
            Width of the field in scan angle degrees.
        yslices : list of slice, optional
            Slices defining how to cut the page in the Y-axis to retrieve this field.
        xslices : list of slice, optional
            Slices defining how to cut the page in the X-axis to retrieve this field.
        output_yslices : list of slice, optional
            Slices defining where to paste this field in the output.
        output_xslices : list of slice, optional
            Slices defining where to paste this field in the output.
        slice_id : int, optional
            Index of the slice in the scan to which this field belongs.
        roi_ids : list of int, optional
            List of ROI indices to which each subfield belongs.
        offsets : list of float, optional
            Time offsets per pixel in seconds for each subfield.

        """
        super().__init__(
            height,
            width,
            depth,
            y_center_coordinate,
            x_center_coordinate,
            height_in_degrees,
            width_in_degrees,
        )
        self.yslices = yslices
        self.xslices = xslices
        self.output_yslices = output_yslices
        self.output_xslices = output_xslices
        self.slice_id = slice_id
        self.roi_ids = roi_ids
        self.offsets = offsets

    @property
    def has_contiguous_subfields(self):
        """Whether field is formed by many contiguous subfields."""
        return len(self.xslices) > 1

    @property
    def roi_mask(self):
        """Mask of the size of the field. Each pixel shows the ROI from where it comes."""
        mask = np.full([self.height, self.width], -1, dtype=np.int8)
        for roi_id, output_yslice, output_xslice in zip(
            self.roi_ids, self.output_yslices, self.output_xslices
        ):
            mask[output_yslice, output_xslice] = roi_id
        return mask

    @property
    def offset_mask(self):
        """Mask of the size of the field. Each pixel shows its time offset in seconds."""
        mask = np.full([self.height, self.width], -1, dtype=np.float32)
        for offsets, output_yslice, output_xslice in zip(
            self.offsets, self.output_yslices, self.output_xslices
        ):
            mask[output_yslice, output_xslice] = offsets
        return mask

    def _type_of_contiguity(self, field2):
        """Compute how field 2 is contiguous to this one.

        Args:
            field2: A second field object.

        Returns:
            An integer {NONCONTIGUOUS = 0, ABOVE = 1, BELOW = 2, LEFT = 3, RIGHT = 4}.
               Whether field 2 is above, below, to the left or to the right of this field.
        """
        position = Position.NONCONTIGUOUS
        if np.isclose(self.width_in_degrees, field2.width_in_degrees):
            expected_distance = (
                self.height_in_degrees / 2 + field2.height_in_degrees / 2
            )
            if np.isclose(
                self.y_center_coordinate, field2.y_center_coordinate + expected_distance
            ):
                position = Position.ABOVE
            if np.isclose(
                field2.y_center_coordinate, self.y_center_coordinate + expected_distance
            ):
                position = Position.BELOW
        if np.isclose(self.height_in_degrees, field2.height_in_degrees):
            expected_distance = self.width_in_degrees / 2 + field2.width_in_degrees / 2
            if np.isclose(
                self.x_center_coordinate, field2.x_center_coordinate + expected_distance
            ):
                position = Position.LEFT
            else:
                position = Position.RIGHT
        return position

    def is_contiguous_to(self, field2):
        """Whether this field is contiguous to field2."""
        return not (self._type_of_contiguity(field2) == Position.NONCONTIGUOUS)

    def join_with(self, field2):
        """
        Update attributes of this field to incorporate field2. Field2 is NOT changed.

        Args:
            field2: A second field object.
        """
        contiguity = self._type_of_contiguity(field2)
        if contiguity in [
            Position.ABOVE,
            Position.BELOW,
        ]:  # contiguous in y_center_coordinate axis
            # Compute some specific attributes
            if contiguity == Position.ABOVE:  # field2 is above/atop self
                new_y = field2.y_center_coordinate + (self.height_in_degrees / 2)
                output_yslices1 = [
                    slice(s.start + field2.height, s.stop + field2.height)
                    for s in self.output_yslices
                ]
                output_yslices2 = field2.output_yslices
            else:  # field2 is below self
                new_y = self.y_center_coordinate + (field2.height_in_degrees / 2)
                output_yslices1 = self.output_yslices
                output_yslices2 = [
                    slice(s.start + self.height, s.stop + self.height)
                    for s in field2.output_yslices
                ]
            # Set new attributes
            self.y_center_coordinate = new_y
            self.height += field2.height
            self.height_in_degrees += field2.height_in_degrees
            self.output_yslices = output_yslices1 + output_yslices2
            self.output_xslices = self.output_xslices + field2.output_xslices
        if contiguity in [
            Position.LEFT,
            Position.RIGHT,
        ]:  # contiguous in x_center_coordinate axis
            # Compute some specific attributes
            if contiguity == Position.LEFT:  # field2 is to the left of self
                new_x = field2.x_center_coordinate + (self.width_in_degrees / 2)
                output_xslices1 = [
                    slice(s.start + field2.width, s.stop + field2.width)
                    for s in self.output_xslices
                ]
                output_xslices2 = field2.output_xslices
            else:  # field2 is to the right of self
                new_x = self.x_center_coordinate + (field2.width_in_degrees / 2)
                output_xslices1 = self.output_xslices
                output_xslices2 = [
                    slice(s.start + self.width, s.stop + self.width)
                    for s in field2.output_xslices
                ]

            # Set new attributes
            self.x_center_coordinate = new_x
            self.width += field2.width
            self.width_in_degrees += field2.width_in_degrees
            self.output_yslices = self.output_yslices + field2.output_yslices
            self.output_xslices = output_xslices1 + output_xslices2

        # yslices and xslices just get appended regardless of the type of contiguity
        self.yslices = self.yslices + field2.yslices
        self.xslices = self.xslices + field2.xslices

        # Append roi ids and offsets
        self.roi_ids = self.roi_ids + field2.roi_ids
        self.offsets = self.offsets + field2.offsets


class Position:
    NONCONTIGUOUS = 0
    ABOVE = 1
    BELOW = 2
    LEFT = 3
    RIGHT = 4
