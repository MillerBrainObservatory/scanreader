"""utils.py: general utilities"""
import types

import numpy as np


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
    if isinstance(index, types.EllipsisType):
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
    if isinstance(index, types.EllipsisType):
        index_as_list = listify_index(slice(None, None, None), dim_size)
    elif np.issubdtype(type(index), np.signedinteger):
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

    input_in_ev = image_in[::2, :]
    input_in_od = image_in[1::2, :]

    min_len = min(input_in_ev.shape[0], input_in_od.shape[0])
    input_in_ev = input_in_ev[:min_len, :]
    input_in_od = input_in_od[:min_len, :]

    buffers = np.zeros((input_in_ev.shape[0], num_values))

    input_in_ev = np.hstack((buffers, input_in_ev, buffers))
    input_in_od = np.hstack((buffers, input_in_od, buffers))

    input_in_ev = input_in_ev.T.ravel(order="F")
    input_in_od = input_in_od.T.ravel(order="F")

    # Zero-center and clip negative values to zero
    input_in_ev = input_in_ev - np.mean(input_in_ev)
    input_in_ev[input_in_ev < 0] = 0

    input_in_od = input_in_od - np.mean(input_in_od)
    input_in_od[input_in_od < 0] = 0

    input_in_ev = input_in_ev[:, np.newaxis]

    input_in_od = input_in_od[:, np.newaxis]

    from scipy.ndimage import correlate
    r_full = correlate(input_in_ev[:, 0], input_in_od[:, 0], mode="full", method="auto")
    unbiased_scale = len(input_in_ev) - np.abs(np.arange(-len(input_in_ev) + 1, len(input_in_ev)))
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
            data_out[1::2, offset: offset + sx, :, :] = data_in[1::2, :, :, :]
        elif offset < 0:
            offset = abs(offset)
            data_out = np.zeros((sy, sx + offset, sc, sz))  # This initialization is key
            data_out[0::2, offset: offset + sx, :, :] = data_in[0::2, :, :, :]
            data_out[1::2, :sx, :, :] = data_in[1::2, :, :, :]
        else:
            half_offset = int(offset / 2)
            data_out = np.zeros((sy, sx + 2 * half_offset, sc, sz))
            data_out[:, half_offset: half_offset + sx, :, :] = data_in
    return data_out
