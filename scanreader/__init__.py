from .core import read_scan
from .scans import ScanMultiROI, BaseScan, ScanLBM
from .multiroi import Field, ROI
from .utils import (fix_scan_phase, return_scan_offset, correct_raster, compute_raster_phase)

def get_size_of_objects():
    """Helper function to get the size of all objects in the current namespace."""
    import sys
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
    return sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if
                   not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1],
                  reverse=True)

__all__ = [
    "read_scan",
    "ScanMultiROI",
    "ScanLBM",
    "BaseScan",
    "Field",
    "ROI",
    "get_size_of_objects"
]
