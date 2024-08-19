"""utils.py: general utilities"""
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


def return_scan_offset(image_in, dim, n_corr=3):
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


def fix_scan_phase(data_in, offset):
    """
    Corrects the scan phase of the data based on a given offset along a specified dimension.

    Parameters
    ----------
    data_in : ndarray
        The input data of shape (sy, sx, sc, sz).
    offset : int
        The amount of offset to correct for.

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
