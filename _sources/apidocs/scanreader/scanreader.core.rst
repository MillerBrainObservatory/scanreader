:py:mod:`scanreader.core`
=========================

.. py:module:: scanreader.core

.. autodoc2-docstring:: scanreader.core
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`read_scan <scanreader.core.read_scan>`
     - .. autodoc2-docstring:: scanreader.core.read_scan
          :summary:
   * - :py:obj:`expand_wildcard <scanreader.core.expand_wildcard>`
     - .. autodoc2-docstring:: scanreader.core.expand_wildcard
          :summary:
   * - :py:obj:`get_scanimage_version <scanreader.core.get_scanimage_version>`
     - .. autodoc2-docstring:: scanreader.core.get_scanimage_version
          :summary:
   * - :py:obj:`is_scan_multiROI <scanreader.core.is_scan_multiROI>`
     - .. autodoc2-docstring:: scanreader.core.is_scan_multiROI
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_scans <scanreader.core._scans>`
     - .. autodoc2-docstring:: scanreader.core._scans
          :summary:

API
~~~

.. py:data:: _scans
   :canonical: scanreader.core._scans
   :value: None

   .. autodoc2-docstring:: scanreader.core._scans

.. py:function:: read_scan(pathnames, dtype=np.int16, join_contiguous=False, lbm=False, x_cut=(), y_cut=())
   :canonical: scanreader.core.read_scan

   .. autodoc2-docstring:: scanreader.core.read_scan

.. py:function:: expand_wildcard(wildcard)
   :canonical: scanreader.core.expand_wildcard

   .. autodoc2-docstring:: scanreader.core.expand_wildcard

.. py:function:: get_scanimage_version(info)
   :canonical: scanreader.core.get_scanimage_version

   .. autodoc2-docstring:: scanreader.core.get_scanimage_version

.. py:function:: is_scan_multiROI(info)
   :canonical: scanreader.core.is_scan_multiROI

   .. autodoc2-docstring:: scanreader.core.is_scan_multiROI
