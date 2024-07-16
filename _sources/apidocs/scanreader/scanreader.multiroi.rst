:py:mod:`scanreader.multiroi`
=============================

.. py:module:: scanreader.multiroi

.. autodoc2-docstring:: scanreader.multiroi
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ROI <scanreader.multiroi.ROI>`
     - .. autodoc2-docstring:: scanreader.multiroi.ROI
          :summary:
   * - :py:obj:`Scanfield <scanreader.multiroi.Scanfield>`
     - .. autodoc2-docstring:: scanreader.multiroi.Scanfield
          :summary:
   * - :py:obj:`Field <scanreader.multiroi.Field>`
     - .. autodoc2-docstring:: scanreader.multiroi.Field
          :summary:
   * - :py:obj:`Position <scanreader.multiroi.Position>`
     - .. autodoc2-docstring:: scanreader.multiroi.Position
          :summary:

API
~~~

.. py:class:: ROI(roi_info)
   :canonical: scanreader.multiroi.ROI

   .. autodoc2-docstring:: scanreader.multiroi.ROI

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.multiroi.ROI.__init__

   .. py:property:: scanfields
      :canonical: scanreader.multiroi.ROI.scanfields

      .. autodoc2-docstring:: scanreader.multiroi.ROI.scanfields

   .. py:property:: is_discrete_plane_mode_on
      :canonical: scanreader.multiroi.ROI.is_discrete_plane_mode_on

      .. autodoc2-docstring:: scanreader.multiroi.ROI.is_discrete_plane_mode_on

   .. py:method:: _create_scanfields()
      :canonical: scanreader.multiroi.ROI._create_scanfields

      .. autodoc2-docstring:: scanreader.multiroi.ROI._create_scanfields

   .. py:method:: get_field_at(scanning_depth)
      :canonical: scanreader.multiroi.ROI.get_field_at

      .. autodoc2-docstring:: scanreader.multiroi.ROI.get_field_at

.. py:class:: Scanfield(height_px=None, width_px=None, depth=None, y_center_coordinate=None, x_center_coordinate=None, height_in_degrees=None, width_in_degrees=None)
   :canonical: scanreader.multiroi.Scanfield

   .. autodoc2-docstring:: scanreader.multiroi.Scanfield

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.multiroi.Scanfield.__init__

   .. py:method:: as_field()
      :canonical: scanreader.multiroi.Scanfield.as_field

      .. autodoc2-docstring:: scanreader.multiroi.Scanfield.as_field

.. py:class:: Field(height_px=None, width_px=None, depth=None, y_center_coordinate=None, x_center_coordinate=None, height_in_degrees=None, width_in_degrees=None, yslices=None, xslices=None, output_yslices=None, output_xslices=None, slice_id=None, roi_ids=None, offsets=None)
   :canonical: scanreader.multiroi.Field

   Bases: :py:obj:`scanreader.multiroi.Scanfield`

   .. autodoc2-docstring:: scanreader.multiroi.Field

   .. rubric:: Initialization

   .. autodoc2-docstring:: scanreader.multiroi.Field.__init__

   .. py:property:: has_contiguous_subfields
      :canonical: scanreader.multiroi.Field.has_contiguous_subfields

      .. autodoc2-docstring:: scanreader.multiroi.Field.has_contiguous_subfields

   .. py:property:: roi_mask
      :canonical: scanreader.multiroi.Field.roi_mask

      .. autodoc2-docstring:: scanreader.multiroi.Field.roi_mask

   .. py:property:: offset_mask
      :canonical: scanreader.multiroi.Field.offset_mask

      .. autodoc2-docstring:: scanreader.multiroi.Field.offset_mask

   .. py:method:: _type_of_contiguity(field2)
      :canonical: scanreader.multiroi.Field._type_of_contiguity

      .. autodoc2-docstring:: scanreader.multiroi.Field._type_of_contiguity

   .. py:method:: is_contiguous_to(field2)
      :canonical: scanreader.multiroi.Field.is_contiguous_to

      .. autodoc2-docstring:: scanreader.multiroi.Field.is_contiguous_to

   .. py:method:: join_with(field2)
      :canonical: scanreader.multiroi.Field.join_with

      .. autodoc2-docstring:: scanreader.multiroi.Field.join_with

.. py:class:: Position
   :canonical: scanreader.multiroi.Position

   .. autodoc2-docstring:: scanreader.multiroi.Position

   .. py:attribute:: NONCONTIGUOUS
      :canonical: scanreader.multiroi.Position.NONCONTIGUOUS
      :value: 0

      .. autodoc2-docstring:: scanreader.multiroi.Position.NONCONTIGUOUS

   .. py:attribute:: ABOVE
      :canonical: scanreader.multiroi.Position.ABOVE
      :value: 1

      .. autodoc2-docstring:: scanreader.multiroi.Position.ABOVE

   .. py:attribute:: BELOW
      :canonical: scanreader.multiroi.Position.BELOW
      :value: 2

      .. autodoc2-docstring:: scanreader.multiroi.Position.BELOW

   .. py:attribute:: LEFT
      :canonical: scanreader.multiroi.Position.LEFT
      :value: 3

      .. autodoc2-docstring:: scanreader.multiroi.Position.LEFT

   .. py:attribute:: RIGHT
      :canonical: scanreader.multiroi.Position.RIGHT
      :value: 4

      .. autodoc2-docstring:: scanreader.multiroi.Position.RIGHT
