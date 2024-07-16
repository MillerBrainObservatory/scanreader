:py:mod:`scanreader.scans`
==========================

.. py:module:: scanreader.scans

.. autodoc2-docstring:: scanreader.scans
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`BaseScan <scanreader.scans.BaseScan>`
     - .. autodoc2-docstring:: scanreader.scans.BaseScan
          :summary:
   * - :py:obj:`ScanLegacy <scanreader.scans.ScanLegacy>`
     - .. autodoc2-docstring:: scanreader.scans.ScanLegacy
          :summary:
   * - :py:obj:`BaseScan5 <scanreader.scans.BaseScan5>`
     - .. autodoc2-docstring:: scanreader.scans.BaseScan5
          :summary:
   * - :py:obj:`Scan5Point1 <scanreader.scans.Scan5Point1>`
     - .. autodoc2-docstring:: scanreader.scans.Scan5Point1
          :summary:
   * - :py:obj:`Scan5Point2 <scanreader.scans.Scan5Point2>`
     - .. autodoc2-docstring:: scanreader.scans.Scan5Point2
          :summary:
   * - :py:obj:`NewerScan <scanreader.scans.NewerScan>`
     - .. autodoc2-docstring:: scanreader.scans.NewerScan
          :summary:
   * - :py:obj:`Scan5Point3 <scanreader.scans.Scan5Point3>`
     - .. autodoc2-docstring:: scanreader.scans.Scan5Point3
          :summary:
   * - :py:obj:`Scan5Point4 <scanreader.scans.Scan5Point4>`
     - .. autodoc2-docstring:: scanreader.scans.Scan5Point4
          :summary:
   * - :py:obj:`Scan5Point5 <scanreader.scans.Scan5Point5>`
     - .. autodoc2-docstring:: scanreader.scans.Scan5Point5
          :summary:
   * - :py:obj:`Scan5Point6 <scanreader.scans.Scan5Point6>`
     - .. autodoc2-docstring:: scanreader.scans.Scan5Point6
          :summary:
   * - :py:obj:`Scan5Point7 <scanreader.scans.Scan5Point7>`
     - .. autodoc2-docstring:: scanreader.scans.Scan5Point7
          :summary:
   * - :py:obj:`Scan2016b <scanreader.scans.Scan2016b>`
     - .. autodoc2-docstring:: scanreader.scans.Scan2016b
          :summary:
   * - :py:obj:`Scan2017a <scanreader.scans.Scan2017a>`
     - .. autodoc2-docstring:: scanreader.scans.Scan2017a
          :summary:
   * - :py:obj:`Scan2017b <scanreader.scans.Scan2017b>`
     - .. autodoc2-docstring:: scanreader.scans.Scan2017b
          :summary:
   * - :py:obj:`Scan2018a <scanreader.scans.Scan2018a>`
     - .. autodoc2-docstring:: scanreader.scans.Scan2018a
          :summary:
   * - :py:obj:`Scan2018b <scanreader.scans.Scan2018b>`
     - .. autodoc2-docstring:: scanreader.scans.Scan2018b
          :summary:
   * - :py:obj:`Scan2019a <scanreader.scans.Scan2019a>`
     - .. autodoc2-docstring:: scanreader.scans.Scan2019a
          :summary:
   * - :py:obj:`Scan2019b <scanreader.scans.Scan2019b>`
     - .. autodoc2-docstring:: scanreader.scans.Scan2019b
          :summary:
   * - :py:obj:`Scan2020 <scanreader.scans.Scan2020>`
     - .. autodoc2-docstring:: scanreader.scans.Scan2020
          :summary:
   * - :py:obj:`Scan2021 <scanreader.scans.Scan2021>`
     - .. autodoc2-docstring:: scanreader.scans.Scan2021
          :summary:
   * - :py:obj:`ScanMultiROI <scanreader.scans.ScanMultiROI>`
     - .. autodoc2-docstring:: scanreader.scans.ScanMultiROI
          :summary:
   * - :py:obj:`LBMScanMultiROI <scanreader.scans.LBMScanMultiROI>`
     - .. autodoc2-docstring:: scanreader.scans.LBMScanMultiROI
          :summary:

API
~~~

.. py:class:: BaseScan()
   :canonical: scanreader.scans.BaseScan

   .. autodoc2-docstring:: scanreader.scans.BaseScan

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.BaseScan.__init__

   .. py:property:: tiff_files
      :canonical: scanreader.scans.BaseScan.tiff_files

      .. autodoc2-docstring:: scanreader.scans.BaseScan.tiff_files

   .. py:property:: version
      :canonical: scanreader.scans.BaseScan.version

      .. autodoc2-docstring:: scanreader.scans.BaseScan.version

   .. py:property:: power_percent
      :canonical: scanreader.scans.BaseScan.power_percent

      .. autodoc2-docstring:: scanreader.scans.BaseScan.power_percent

   .. py:property:: is_slow_stack
      :canonical: scanreader.scans.BaseScan.is_slow_stack

      .. autodoc2-docstring:: scanreader.scans.BaseScan.is_slow_stack

   .. py:property:: is_multiROI
      :canonical: scanreader.scans.BaseScan.is_multiROI

      .. autodoc2-docstring:: scanreader.scans.BaseScan.is_multiROI

   .. py:property:: num_channels
      :canonical: scanreader.scans.BaseScan.num_channels

      .. autodoc2-docstring:: scanreader.scans.BaseScan.num_channels

   .. py:property:: requested_scanning_depths
      :canonical: scanreader.scans.BaseScan.requested_scanning_depths

      .. autodoc2-docstring:: scanreader.scans.BaseScan.requested_scanning_depths

   .. py:property:: num_scanning_depths
      :canonical: scanreader.scans.BaseScan.num_scanning_depths

      .. autodoc2-docstring:: scanreader.scans.BaseScan.num_scanning_depths

   .. py:property:: scanning_depths
      :canonical: scanreader.scans.BaseScan.scanning_depths

      .. autodoc2-docstring:: scanreader.scans.BaseScan.scanning_depths

   .. py:property:: num_requested_frames
      :canonical: scanreader.scans.BaseScan.num_requested_frames

      .. autodoc2-docstring:: scanreader.scans.BaseScan.num_requested_frames

   .. py:property:: num_frames
      :canonical: scanreader.scans.BaseScan.num_frames

      .. autodoc2-docstring:: scanreader.scans.BaseScan.num_frames

   .. py:property:: is_bidirectional
      :canonical: scanreader.scans.BaseScan.is_bidirectional

      .. autodoc2-docstring:: scanreader.scans.BaseScan.is_bidirectional

   .. py:property:: scanner_frequency
      :canonical: scanreader.scans.BaseScan.scanner_frequency

      .. autodoc2-docstring:: scanreader.scans.BaseScan.scanner_frequency

   .. py:property:: seconds_per_line
      :canonical: scanreader.scans.BaseScan.seconds_per_line

      .. autodoc2-docstring:: scanreader.scans.BaseScan.seconds_per_line

   .. py:property:: _num_pages
      :canonical: scanreader.scans.BaseScan._num_pages

      .. autodoc2-docstring:: scanreader.scans.BaseScan._num_pages

   .. py:property:: _page_height
      :canonical: scanreader.scans.BaseScan._page_height

      .. autodoc2-docstring:: scanreader.scans.BaseScan._page_height

   .. py:property:: _page_width
      :canonical: scanreader.scans.BaseScan._page_width

      .. autodoc2-docstring:: scanreader.scans.BaseScan._page_width

   .. py:property:: _num_averaged_frames
      :canonical: scanreader.scans.BaseScan._num_averaged_frames

      .. autodoc2-docstring:: scanreader.scans.BaseScan._num_averaged_frames

   .. py:property:: num_fields
      :canonical: scanreader.scans.BaseScan.num_fields
      :abstractmethod:

      .. autodoc2-docstring:: scanreader.scans.BaseScan.num_fields

   .. py:property:: field_depths
      :canonical: scanreader.scans.BaseScan.field_depths
      :abstractmethod:

      .. autodoc2-docstring:: scanreader.scans.BaseScan.field_depths

   .. py:property:: fps
      :canonical: scanreader.scans.BaseScan.fps

      .. autodoc2-docstring:: scanreader.scans.BaseScan.fps

   .. py:property:: spatial_fill_fraction
      :canonical: scanreader.scans.BaseScan.spatial_fill_fraction

      .. autodoc2-docstring:: scanreader.scans.BaseScan.spatial_fill_fraction

   .. py:property:: temporal_fill_fraction
      :canonical: scanreader.scans.BaseScan.temporal_fill_fraction

      .. autodoc2-docstring:: scanreader.scans.BaseScan.temporal_fill_fraction

   .. py:property:: scanner_type
      :canonical: scanreader.scans.BaseScan.scanner_type

      .. autodoc2-docstring:: scanreader.scans.BaseScan.scanner_type

   .. py:property:: motor_position_at_zero
      :canonical: scanreader.scans.BaseScan.motor_position_at_zero

      .. autodoc2-docstring:: scanreader.scans.BaseScan.motor_position_at_zero

   .. py:property:: initial_secondary_z
      :canonical: scanreader.scans.BaseScan.initial_secondary_z

      .. autodoc2-docstring:: scanreader.scans.BaseScan.initial_secondary_z

   .. py:property:: _initial_frame_number
      :canonical: scanreader.scans.BaseScan._initial_frame_number

      .. autodoc2-docstring:: scanreader.scans.BaseScan._initial_frame_number

   .. py:property:: _num_fly_back_lines
      :canonical: scanreader.scans.BaseScan._num_fly_back_lines

      .. autodoc2-docstring:: scanreader.scans.BaseScan._num_fly_back_lines

   .. py:property:: _num_lines_between_fields
      :canonical: scanreader.scans.BaseScan._num_lines_between_fields

      .. autodoc2-docstring:: scanreader.scans.BaseScan._num_lines_between_fields

   .. py:property:: is_slow_stack_with_fastZ
      :canonical: scanreader.scans.BaseScan.is_slow_stack_with_fastZ
      :abstractmethod:

      .. autodoc2-docstring:: scanreader.scans.BaseScan.is_slow_stack_with_fastZ

   .. py:property:: field_offsets
      :canonical: scanreader.scans.BaseScan.field_offsets
      :abstractmethod:

      .. autodoc2-docstring:: scanreader.scans.BaseScan.field_offsets

   .. py:method:: read_data(filenames, dtype)
      :canonical: scanreader.scans.BaseScan.read_data

      .. autodoc2-docstring:: scanreader.scans.BaseScan.read_data

   .. py:method:: __array__()
      :canonical: scanreader.scans.BaseScan.__array__

      .. autodoc2-docstring:: scanreader.scans.BaseScan.__array__

   .. py:method:: __str__()
      :canonical: scanreader.scans.BaseScan.__str__

   .. py:method:: __len__()
      :canonical: scanreader.scans.BaseScan.__len__

      .. autodoc2-docstring:: scanreader.scans.BaseScan.__len__

   .. py:method:: __getitem__(key)
      :canonical: scanreader.scans.BaseScan.__getitem__
      :abstractmethod:

      .. autodoc2-docstring:: scanreader.scans.BaseScan.__getitem__

   .. py:method:: __iter__()
      :canonical: scanreader.scans.BaseScan.__iter__

      .. autodoc2-docstring:: scanreader.scans.BaseScan.__iter__

   .. py:method:: _read_pages(slice_list, channel_list, frame_list, yslice=slice(None), xslice=slice(None))
      :canonical: scanreader.scans.BaseScan._read_pages

      .. autodoc2-docstring:: scanreader.scans.BaseScan._read_pages

   .. py:method:: _seconds_to_lines(seconds)
      :canonical: scanreader.scans.BaseScan._seconds_to_lines

      .. autodoc2-docstring:: scanreader.scans.BaseScan._seconds_to_lines

   .. py:method:: _compute_offsets(field_height, start_line)
      :canonical: scanreader.scans.BaseScan._compute_offsets

      .. autodoc2-docstring:: scanreader.scans.BaseScan._compute_offsets

.. py:class:: ScanLegacy()
   :canonical: scanreader.scans.ScanLegacy

   Bases: :py:obj:`scanreader.scans.BaseScan`

   .. autodoc2-docstring:: scanreader.scans.ScanLegacy

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.ScanLegacy.__init__

.. py:class:: BaseScan5()
   :canonical: scanreader.scans.BaseScan5

   Bases: :py:obj:`scanreader.scans.BaseScan`

   .. autodoc2-docstring:: scanreader.scans.BaseScan5

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.BaseScan5.__init__

   .. py:property:: num_fields
      :canonical: scanreader.scans.BaseScan5.num_fields

      .. autodoc2-docstring:: scanreader.scans.BaseScan5.num_fields

   .. py:property:: field_depths
      :canonical: scanreader.scans.BaseScan5.field_depths

      .. autodoc2-docstring:: scanreader.scans.BaseScan5.field_depths

   .. py:property:: image_height
      :canonical: scanreader.scans.BaseScan5.image_height

      .. autodoc2-docstring:: scanreader.scans.BaseScan5.image_height

   .. py:property:: image_width
      :canonical: scanreader.scans.BaseScan5.image_width

      .. autodoc2-docstring:: scanreader.scans.BaseScan5.image_width

   .. py:property:: shape
      :canonical: scanreader.scans.BaseScan5.shape

      .. autodoc2-docstring:: scanreader.scans.BaseScan5.shape

   .. py:property:: zoom
      :canonical: scanreader.scans.BaseScan5.zoom

      .. autodoc2-docstring:: scanreader.scans.BaseScan5.zoom

   .. py:property:: is_slow_stack_with_fastZ
      :canonical: scanreader.scans.BaseScan5.is_slow_stack_with_fastZ

      .. autodoc2-docstring:: scanreader.scans.BaseScan5.is_slow_stack_with_fastZ

   .. py:property:: field_offsets
      :canonical: scanreader.scans.BaseScan5.field_offsets

      .. autodoc2-docstring:: scanreader.scans.BaseScan5.field_offsets

   .. py:property:: _y_angle_scale_factor
      :canonical: scanreader.scans.BaseScan5._y_angle_scale_factor

      .. autodoc2-docstring:: scanreader.scans.BaseScan5._y_angle_scale_factor

   .. py:property:: _x_angle_scale_factor
      :canonical: scanreader.scans.BaseScan5._x_angle_scale_factor

      .. autodoc2-docstring:: scanreader.scans.BaseScan5._x_angle_scale_factor

   .. py:method:: __getitem__(key)
      :canonical: scanreader.scans.BaseScan5.__getitem__

      .. autodoc2-docstring:: scanreader.scans.BaseScan5.__getitem__

.. py:class:: Scan5Point1()
   :canonical: scanreader.scans.Scan5Point1

   Bases: :py:obj:`scanreader.scans.BaseScan5`

   .. autodoc2-docstring:: scanreader.scans.Scan5Point1

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.Scan5Point1.__init__

.. py:class:: Scan5Point2()
   :canonical: scanreader.scans.Scan5Point2

   Bases: :py:obj:`scanreader.scans.BaseScan5`

   .. autodoc2-docstring:: scanreader.scans.Scan5Point2

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.Scan5Point2.__init__

   .. py:property:: image_height_in_microns
      :canonical: scanreader.scans.Scan5Point2.image_height_in_microns

      .. autodoc2-docstring:: scanreader.scans.Scan5Point2.image_height_in_microns

   .. py:property:: image_width_in_microns
      :canonical: scanreader.scans.Scan5Point2.image_width_in_microns

      .. autodoc2-docstring:: scanreader.scans.Scan5Point2.image_width_in_microns

.. py:class:: NewerScan
   :canonical: scanreader.scans.NewerScan

   .. autodoc2-docstring:: scanreader.scans.NewerScan

   .. py:property:: is_slow_stack_with_fastZ
      :canonical: scanreader.scans.NewerScan.is_slow_stack_with_fastZ

      .. autodoc2-docstring:: scanreader.scans.NewerScan.is_slow_stack_with_fastZ

.. py:class:: Scan5Point3()
   :canonical: scanreader.scans.Scan5Point3

   Bases: :py:obj:`scanreader.scans.NewerScan`, :py:obj:`scanreader.scans.Scan5Point2`

   .. autodoc2-docstring:: scanreader.scans.Scan5Point3

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.Scan5Point3.__init__

.. py:class:: Scan5Point4()
   :canonical: scanreader.scans.Scan5Point4

   Bases: :py:obj:`scanreader.scans.Scan5Point3`

   .. autodoc2-docstring:: scanreader.scans.Scan5Point4

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.Scan5Point4.__init__

.. py:class:: Scan5Point5()
   :canonical: scanreader.scans.Scan5Point5

   Bases: :py:obj:`scanreader.scans.Scan5Point3`

   .. autodoc2-docstring:: scanreader.scans.Scan5Point5

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.Scan5Point5.__init__

.. py:class:: Scan5Point6()
   :canonical: scanreader.scans.Scan5Point6

   Bases: :py:obj:`scanreader.scans.Scan5Point3`

   .. autodoc2-docstring:: scanreader.scans.Scan5Point6

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.Scan5Point6.__init__

.. py:class:: Scan5Point7()
   :canonical: scanreader.scans.Scan5Point7

   Bases: :py:obj:`scanreader.scans.Scan5Point3`

   .. autodoc2-docstring:: scanreader.scans.Scan5Point7

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.Scan5Point7.__init__

.. py:class:: Scan2016b()
   :canonical: scanreader.scans.Scan2016b

   Bases: :py:obj:`scanreader.scans.Scan5Point3`

   .. autodoc2-docstring:: scanreader.scans.Scan2016b

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.Scan2016b.__init__

.. py:class:: Scan2017a()
   :canonical: scanreader.scans.Scan2017a

   Bases: :py:obj:`scanreader.scans.Scan5Point3`

   .. autodoc2-docstring:: scanreader.scans.Scan2017a

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.Scan2017a.__init__

.. py:class:: Scan2017b()
   :canonical: scanreader.scans.Scan2017b

   Bases: :py:obj:`scanreader.scans.Scan5Point3`

   .. autodoc2-docstring:: scanreader.scans.Scan2017b

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.Scan2017b.__init__

.. py:class:: Scan2018a()
   :canonical: scanreader.scans.Scan2018a

   Bases: :py:obj:`scanreader.scans.Scan5Point3`

   .. autodoc2-docstring:: scanreader.scans.Scan2018a

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.Scan2018a.__init__

.. py:class:: Scan2018b()
   :canonical: scanreader.scans.Scan2018b

   Bases: :py:obj:`scanreader.scans.Scan5Point3`

   .. autodoc2-docstring:: scanreader.scans.Scan2018b

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.Scan2018b.__init__

.. py:class:: Scan2019a()
   :canonical: scanreader.scans.Scan2019a

   Bases: :py:obj:`scanreader.scans.Scan5Point3`

   .. autodoc2-docstring:: scanreader.scans.Scan2019a

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.Scan2019a.__init__

.. py:class:: Scan2019b()
   :canonical: scanreader.scans.Scan2019b

   Bases: :py:obj:`scanreader.scans.Scan5Point3`

   .. autodoc2-docstring:: scanreader.scans.Scan2019b

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.Scan2019b.__init__

.. py:class:: Scan2020()
   :canonical: scanreader.scans.Scan2020

   Bases: :py:obj:`scanreader.scans.Scan5Point3`

   .. autodoc2-docstring:: scanreader.scans.Scan2020

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.Scan2020.__init__

.. py:class:: Scan2021()
   :canonical: scanreader.scans.Scan2021

   Bases: :py:obj:`scanreader.scans.Scan5Point3`

   .. autodoc2-docstring:: scanreader.scans.Scan2021

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.Scan2021.__init__

.. py:class:: ScanMultiROI(join_contiguous)
   :canonical: scanreader.scans.ScanMultiROI

   Bases: :py:obj:`scanreader.scans.NewerScan`, :py:obj:`scanreader.scans.BaseScan`

   .. autodoc2-docstring:: scanreader.scans.ScanMultiROI

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.ScanMultiROI.__init__

   .. py:property:: num_fields
      :canonical: scanreader.scans.ScanMultiROI.num_fields

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI.num_fields

   .. py:property:: num_rois
      :canonical: scanreader.scans.ScanMultiROI.num_rois

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI.num_rois

   .. py:property:: field_heights
      :canonical: scanreader.scans.ScanMultiROI.field_heights

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI.field_heights

   .. py:property:: field_widths
      :canonical: scanreader.scans.ScanMultiROI.field_widths

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI.field_widths

   .. py:property:: field_depths
      :canonical: scanreader.scans.ScanMultiROI.field_depths

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI.field_depths

   .. py:property:: field_slices
      :canonical: scanreader.scans.ScanMultiROI.field_slices

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI.field_slices

   .. py:property:: field_rois
      :canonical: scanreader.scans.ScanMultiROI.field_rois

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI.field_rois

   .. py:property:: field_masks
      :canonical: scanreader.scans.ScanMultiROI.field_masks

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI.field_masks

   .. py:property:: field_offsets
      :canonical: scanreader.scans.ScanMultiROI.field_offsets

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI.field_offsets

   .. py:property:: field_heights_in_microns
      :canonical: scanreader.scans.ScanMultiROI.field_heights_in_microns

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI.field_heights_in_microns

   .. py:property:: field_widths_in_microns
      :canonical: scanreader.scans.ScanMultiROI.field_widths_in_microns

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI.field_widths_in_microns

   .. py:property:: _num_fly_to_lines
      :canonical: scanreader.scans.ScanMultiROI._num_fly_to_lines

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI._num_fly_to_lines

   .. py:method:: _degrees_to_microns(degrees)
      :canonical: scanreader.scans.ScanMultiROI._degrees_to_microns

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI._degrees_to_microns

   .. py:method:: _microns_to_decrees(microns)
      :canonical: scanreader.scans.ScanMultiROI._microns_to_decrees

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI._microns_to_decrees

   .. py:method:: _degrees_to_pixels(degrees, num_pixels)
      :canonical: scanreader.scans.ScanMultiROI._degrees_to_pixels

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI._degrees_to_pixels

   .. py:method:: _pixels_to_degrees(pixels, num_pixels)
      :canonical: scanreader.scans.ScanMultiROI._pixels_to_degrees

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI._pixels_to_degrees

   .. py:method:: _pixels_to_microns(pixels, num_pixels)
      :canonical: scanreader.scans.ScanMultiROI._pixels_to_microns

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI._pixels_to_microns

   .. py:method:: read_data(filenames, dtype)
      :canonical: scanreader.scans.ScanMultiROI.read_data

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI.read_data

   .. py:method:: _create_rois()
      :canonical: scanreader.scans.ScanMultiROI._create_rois

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI._create_rois

   .. py:method:: _create_fields()
      :canonical: scanreader.scans.ScanMultiROI._create_fields

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI._create_fields

   .. py:method:: _join_contiguous_fields()
      :canonical: scanreader.scans.ScanMultiROI._join_contiguous_fields

      .. autodoc2-docstring:: scanreader.scans.ScanMultiROI._join_contiguous_fields

   .. py:method:: __getitem__(key)
      :canonical: scanreader.scans.ScanMultiROI.__getitem__

.. py:class:: LBMScanMultiROI(join_contiguous, x_cut=(0, 0), y_cut=(0, 0))
   :canonical: scanreader.scans.LBMScanMultiROI

   Bases: :py:obj:`scanreader.scans.ScanMultiROI`

   .. autodoc2-docstring:: scanreader.scans.LBMScanMultiROI

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.scans.LBMScanMultiROI.__init__

   .. py:method:: _create_fields()
      :canonical: scanreader.scans.LBMScanMultiROI._create_fields

      .. autodoc2-docstring:: scanreader.scans.LBMScanMultiROI._create_fields
