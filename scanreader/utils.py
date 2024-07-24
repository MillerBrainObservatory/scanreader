"""utils.py: general utilities"""
import numpy as np
from numpy import fft


def compute(frames: np.ndarray) -> int:
    """
    Returns the bidirectional phase offset, the offset between lines that sometimes occurs in line scanning.

    Parameters
    ----------
    frames : frames x Ly x Lx
        random subsample of frames in binary (frames x Ly x Lx)

    Returns
    -------
    bidiphase : int
        bidirectional phase offset in pixels

    """

    _, Ly, Lx = frames.shape

    # compute phase-correlation between lines in x-direction
    d1 = fft.fft(frames[:, 1::2, :], axis=2)
    d1 /= np.abs(d1) + 1e-5

    d2 = np.conj(fft.fft(frames[:, ::2, :], axis=2))
    d2 /= np.abs(d2) + 1e-5
    d2 = d2[:, :d1.shape[1], :]

    cc = np.real(fft.ifft(d1 * d2, axis=2))
    cc = cc.mean(axis=1).mean(axis=0)
    cc = fft.fftshift(cc)

    bidiphase = -(np.argmax(cc[-10 + Lx // 2:11 + Lx // 2]) - 10)
    return bidiphase


def shift(frames: np.ndarray, bidiphase: int) -> None:
    """
    Shift last axis of "frames" by bidirectional phase offset in-place, bidiphase.

    Parameters
    ----------
    frames : frames x Ly x Lx
    bidiphase : int
        bidirectional phase offset in pixels
    """
    if bidiphase > 0:
        frames[:, 1::2, bidiphase:] = frames[:, 1::2, :-bidiphase]
    else:
        frames[:, 1::2, :bidiphase] = frames[:, 1::2, -bidiphase:]


def fill_key(key, num_dimensions):
    """ Fill key with slice(None) (':') until num_dimensions size.

    Parameters
    ----------
    key: tuple
        Indices or single index. key as received by __getitem__().
    num_dimensions: int.
        Total number of dimensions needed.

    """

    # Deal with single valued keys, e.g., scan[:] or scan[0]
    if not isinstance(key, tuple):
        key = (key,)

    # Check key is not larger than num_dimensions
    if len(key) > num_dimensions:
        raise IndexError('too many indices for scan: {}'.format(len(key)))

    # Add missing dimensions
    missing_dimensions = num_dimensions - len(key)
    full_key = tuple(list(key) + [slice(None)] * missing_dimensions)

    return full_key


def check_index_type(axis, index):
    """
    Checks that index is an integer, slice or array/list/tuple of integers.

    Parameters
    ----------
    axis: int
        Axis of the specified index.
    index: int | tuple | np.ndarray
        Index to inspect.

    """
    if not _index_has_valid_type(index):  # raise error
        error_msg = ('index {} in axis {} is not an integer, slice or array/list/tuple '
                     'of integers'.format(index, axis))
        raise TypeError(error_msg)


def _index_has_valid_type(index):
    if np.issubdtype(type(index), np.signedinteger):  # integer
        return True
    if isinstance(index, slice):  # slice
        return True
    if (isinstance(index, (list, tuple)) and
            all(np.issubdtype(type(x), np.signedinteger) for x in index)):  # list or tuple
        return True
    if (isinstance(index, np.ndarray) and np.issubdtype(index.dtype, np.signedinteger)
            and index.ndim == 1):  # array
        return True

    return False


def check_index_is_in_bounds(axis, index, dim_size):
    """
    Check that an index is in bounds for the given dimension size.

    By python indexing rules, anything from -dim_size to dim_size-1 is valid.

    Parameters
    ----------
    axis: int
        Axis of the index.
    index: int | list | slice
        Index to check.
    dim_size: int
        Size of the dimension against which the index will be checked.

    """
    if not _is_index_in_bounds(index, dim_size):
        error_msg = ('index {} is out of bounds for axis {} with size '
                     '{}'.format(index, axis, dim_size))
        raise IndexError(error_msg)


def _is_index_in_bounds(index, dim_size):
    if np.issubdtype(type(index), np.signedinteger):
        return index in range(-dim_size, dim_size)
    elif isinstance(index, (list, tuple, np.ndarray)):
        return all(x in range(-dim_size, dim_size) for x in index)
    elif isinstance(index, slice):
        return True  # slices never go out of bounds, they are just cropped
    else:
        error_msg = ('index {} is not either integer, slice or array/list/tuple of '
                     'integers'.format(index))
        raise TypeError(error_msg)


def listify_index(index, dim_size):
    """ Generates the list representation of an index for the given dim_size."""
    if np.issubdtype(type(index), np.signedinteger):
        index_as_list = [index] if index >= 0 else [dim_size + index]
    elif isinstance(index, (list, tuple, np.ndarray)):
        index_as_list = [x if x >= 0 else (dim_size + x) for x in index]
    elif isinstance(index, slice):
        start, stop, step = index.indices(dim_size)  # transforms Nones and negative ints to valid slice
        index_as_list = list(range(start, stop, step))
    else:
        error_msg = ('index {} is not integer, slice or array/list/tuple of '
                     'integers'.format(index))
        raise TypeError(error_msg)

    return index_as_list


def compute_raster_phase(image, temporal_fill_fraction):
    """
    Compute raster correction for bidirectional resonant scanners by estimating the phase shift
    that best aligns even and odd rows of an image. This function assumes that the distortion
    is primarily due to the resonant mirror returning before reaching the nominal turnaround point.

    Parameters
    ----------
    image : np.array
        The image to be corrected. This should be a 2D numpy array where rows correspond to
        successive lines scanned by the resonant mirror.
    temporal_fill_fraction : float
        The fraction of the total line scan period during which the actual image acquisition occurs.
        This is used to calculate the effective scan range in angles. It typically ranges from 0 to 1,
        where 1 means the entire period is used for acquiring image data.

    Returns
    -------
    float
        An estimated phase shift angle (in radians) that indicates the discrepancy between the
        expected and actual initial angles of the bidirectional scan. Positive values suggest that
        even rows should be shifted to the right.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(100, 100)  # Simulated image
    >>> temporal_fill_fraction = 0.9
    >>> angle_shift = compute_raster_phase(image, temporal_fill_fraction)
    >>> print(f"Calculated angle shift: {angle_shift} radians")

    Notes
    -----
    - The function uses linear interpolation for shifting the pixel rows and assumes that even rows
      and odd rows should be symmetric around the central scan line.
    - The phase shift is found using a greedy algorithm that iteratively refines the estimate.
    - Artifacts near the edges of the image can significantly affect the accuracy of the phase
      estimation, hence a default 5% of the rows and 10% of the columns are skipped during the
      calculation.
    - This function depends on `numpy` for numerical operations and `scipy.interpolate` for
      interpolation of row data.

    """
    from scipy import interpolate as interp  # local import, this is a large dependency for this small package
    # Make sure image has even number of rows (so number of even and odd rows is the same)
    image = image[:-1] if image.shape[0] % 2 == 1 else image

    # Get some params
    image_height, image_width = image.shape
    skip_rows = round(image_height * 0.05)  # rows near the top or bottom have artifacts
    skip_cols = round(image_width * 0.10)  # so do columns

    # Create images with even and odd rows
    even_rows = image[::2][skip_rows:-skip_rows]
    odd_rows = image[1::2][skip_rows:-skip_rows]

    # Scan angle at which each pixel was recorded.
    max_angle = (np.pi / 2) * temporal_fill_fraction
    scan_angles = np.linspace(-max_angle, max_angle, image_width + 2)[1:-1]

    # Greedy search for the best raster phase: starts at coarse estimates and refines them
    even_interp = interp.interp1d(scan_angles, even_rows, fill_value="extrapolate")
    odd_interp = interp.interp1d(scan_angles, odd_rows, fill_value="extrapolate")
    angle_shift = 0
    for scale in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        angle_shifts = angle_shift + scale * np.linspace(-9, 9, 19)
        match_values = []
        for new_angle_shift in angle_shifts:
            shifted_evens = even_interp(scan_angles + new_angle_shift)
            shifted_odds = odd_interp(scan_angles - new_angle_shift)
            match_values.append(
                np.sum(
                    shifted_evens[:, skip_cols:-skip_cols]
                    * shifted_odds[:, skip_cols:-skip_cols]
                )
            )
        angle_shift = angle_shifts[np.argmax(match_values)]

    return angle_shift


def correct_raster(scan, raster_phase, temporal_fill_fraction, in_place=True):
    """
    Perform raster correction for resonant scans by adjusting even and odd lines based
    on a specified phase shift. This function is designed to correct geometric distortions
    in multi-photon images caused by the scanning mechanism's characteristics.

    Parameters
    ----------
    scan : np.array
        A numpy array representing the scan data, where the first two dimensions correspond to
        image height and width, respectively. The array can have additional dimensions,
        typically representing different frames or slices.
    raster_phase : float
        The phase shift angle (in radians) to be applied for correction. Positive values shift
        even lines to the left and odd lines to the right, while negative values shift even lines
        to the right and odd lines to the left.
    temporal_fill_fraction : float
        The ratio of the actual imaging time to the total time of one scan line. This parameter
        helps in determining the effective scan range that needs correction.
    in_place : bool, optional
        If True (default), modifies the input `scan` array in-place. If False, a corrected copy
        of the scan is returned, preserving the original data.

    Returns
    -------
    np.array
        The raster-corrected scan. The return type matches the input `scan` data type if it's a
        subtype of np.float. Otherwise, it is converted to np.float32 for processing.

    Raises
    ------
    PipelineException
        If input validations fail such as non-matching dimensions, incorrect data types, or if
        the `scan` does not have at least two dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> scan = np.random.rand(256, 256, 30)  # Simulate a 3D scan volume
    >>> raster_phase = 0.01  # Small phase shift
    >>> temporal_fill_fraction = 0.9
    >>> corrected_scan = correct_raster(scan, raster_phase, temporal_fill_fraction, in_place=False)
    >>> print(corrected_scan.shape)
    (256, 256, 30)

    Notes
    -----
    The raster correction is essential for improving the accuracy of image analyses and
    quantification in studies involving resonant scanning microscopy. Adjusting the phase
    shift accurately according to the resonant mirror's characteristics can significantly
    enhance the image quality.

    """
    from scipy import interpolate as interp  # local import, this is a large dependency for this small package
    # Basic checks
    if not hasattr(scan, 'shape'):
        raise TypeError("Scan needs to be np.array-like")
    if scan.ndim < 2:
        raise TypeError("Scan with less than 2 dimensions.")

    # Assert scan is float
    if not np.issubdtype(scan.dtype, np.floating):
        print("Warning: Changing scan type from", str(scan.dtype), "to np.float32")
        scan = scan.astype(np.float32, copy=(not in_place))
    elif not in_place:
        scan = scan.copy()  # copy it anyway preserving the original float dtype

    # Get some dimensions
    original_shape = scan.shape
    image_height = original_shape[0]
    image_width = original_shape[1]

    # Scan angle at which each pixel was recorded.
    max_angle = (np.pi / 2) * temporal_fill_fraction
    scan_angles = np.linspace(-max_angle, max_angle, image_width + 2)[1:-1]

    # We iterate over every image in the scan (first 2 dimensions). Same correction
    # regardless of what channel, slice or frame they belong to.
    reshaped_scan = np.reshape(scan, (image_height, image_width, -1))
    num_images = reshaped_scan.shape[-1]
    for i in range(num_images):
        # Get current image
        image = reshaped_scan[:, :, i]

        # Correct even rows of the image (0, 2, ...)
        interp_function = interp.interp1d(
            scan_angles,
            image[::2, :],
            bounds_error=False,
            fill_value=0,
            copy=(not in_place),
        )
        reshaped_scan[::2, :, i] = interp_function(scan_angles + raster_phase)

        # Correct odd rows of the image (1, 3, ...)
        interp_function = interp.interp1d(
            scan_angles,
            image[1::2, :],
            bounds_error=False,
            fill_value=0,
            copy=(not in_place),
        )
        reshaped_scan[1::2, :, i] = interp_function(scan_angles - raster_phase)
    return np.reshape(reshaped_scan, original_shape)


def return_scan_offset(image_in, dim, n_corr=3, run_napari=False):
    """
    Compute the scan offset correction between interleaved lines or columns in an image.

    Dimensions should be arranged in order by: [height, width], [height, width, time], or [height, width, time, channel/plane].
    The input array must be castable to numpy. e.g. np.shape, np.ravel.

    Parameters
    ----------
    image_in : ndarray | ndarray-like
        Input image or volume.
    dim : int
        Dimension along which to compute the scan offset correction.
    n_corr : int, optional
        The number of cross-correlation shifts to compare when searching for highest correlation.
        Default is 8 pixels shifts in the negative and positive direction.

    Returns
    -------
    int
        The pixel shift transform needed to obtain the highest line-phase correlation.

    Examples
    --------
    >>> img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    >>> return_scan_offset(img, 1)

    Notes
    -----
    This function assumes that the input image contains interleaved lines or columns that
    need to be analyzed for misalignment. The cross-correlation method is sensitive to
    the similarity in pattern between the interleaved lines or columns. Hence, a strong
    and clear peak in the cross-correlation result indicates a good alignment, and the
    corresponding lag value indicates the amount of misalignment.
    """
    from scipy import signal

    if len(image_in.shape) == 3:
        print(
            f'Array with 3 dimensions should be [time, height, width], averageing over the third dimension of array with shape: {image_in.shape}')
        image_in = np.mean(image_in, axis=0)
    elif len(image_in.shape) == 4:
        # image_in = np.mean(np.mean(image_in, axis=3), axis=2)
        return NotImplemented(
            "Input detected as 4D: Volumetric arrays are not yet supported. Check that you don't have an empty axis with np.squeeze()")
    n = n_corr

    Iv1 = None
    Iv2 = None
    if dim == 1:
        Iv1 = image_in[::2, :]
        Iv2 = image_in[1::2, :]

        min_len = min(Iv1.shape[0], Iv2.shape[0])
        Iv1 = Iv1[:min_len, :]
        Iv2 = Iv2[:min_len, :]

        buffers = np.zeros((Iv1.shape[0], n))

        Iv1 = np.hstack((buffers, Iv1, buffers))
        Iv2 = np.hstack((buffers, Iv2, buffers))

        Iv1 = Iv1.T.ravel(order='F')
        Iv2 = Iv2.T.ravel(order='F')

    elif dim == 2:
        raise NotImplementedError("Scan-phase correction does not yet support the 2nd dimension.")

    # Zero-center and clip negative values to zero
    Iv1 = Iv1 - np.mean(Iv1)
    Iv1[Iv1 < 0] = 0

    Iv2 = Iv2 - np.mean(Iv2)
    Iv2[Iv2 < 0] = 0

    Iv1 = Iv1[:, np.newaxis]
    Iv2 = Iv2[:, np.newaxis]

    r_full = signal.correlate(Iv1[:, 0], Iv2[:, 0], mode='full', method='auto')
    unbiased_scale = len(Iv1) - np.abs(np.arange(-len(Iv1) + 1, len(Iv1)))
    r = r_full / unbiased_scale

    mid_point = len(r) // 2
    lower_bound = mid_point - n
    upper_bound = mid_point + n + 1
    r = r[lower_bound:upper_bound]
    lags = np.arange(-n, n + 1)

    # Step 3: Find the correction value
    correction_index = np.argmax(r)
    return lags[correction_index]


def fix_scan_phase(data_in, offset, dim):
    """
    Corrects the scan phase of the data based on a given offset along a specified dimension.

    Parameters
    ----------
    data_in : ndarray
        The input data of shape (sy, sx, sc, sz).
    offset : int
        The amount of offset to correct for.
    dim : int
        Dimension along which to apply the offset.
        1 for vertical (along height/sy), 2 for horizontal (along width/sx).

    Returns
    -------
    ndarray
        The data with corrected scan phase, of shape (sy, sx, sc, sz).
    """
    # fill dims as none
    st, sy, sx, sc, sz = np.nan * np.ones(5)
    if len(data_in.shape) == 4:
        raise NotImplemented("Volumetric scan-phase corrections not yet supported.")
    elif len(data_in.shape) == 3:
        st, sy, sx = data_in.shape
    elif len(data_in.shape) == 2:
        sy, sx = data_in.shape

    # make sure this image is not wider than it is tall
    if sx > sy:
        raise Warning("Image is wider than it is tall, make sure you are correcting the right dimension.")

    if offset > 0:
        data_out = np.zeros_like(data_in)
        data_out[:, 0::2, :sx] = data_in[:, 0::2, ...]
        data_out[:, 1::2, offset:offset + sx, ...] = data_in[:, 1::2, ...]
    elif offset < 0:
        offset = abs(offset)
        data_out = np.zeros_like(data_in)
        data_out[:, offset:offset + sx, ...] = data_in[0::2, :, :, :]
        data_out[:, :sx, :, :] = data_in[1::2, :, :, :]
    else:
        half_offset = int(offset / 2)
        data_out = np.zeros((sy, sx + 2 * half_offset, sc, sz))
        data_out[:, half_offset:half_offset + sx, :, :] = data_in
    return data_out
