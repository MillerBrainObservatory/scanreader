import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import napari
from napari.types import LayerDataTuple
import scanreader
import pandas as pd
import tifffile
from qtpy import QtCore
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QTableView,
    QWidget,
)


def reader_function(path: os.PathLike) -> List[LayerDataTuple]:
    """
    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer.

        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """

    print("Loading LBM directory")
    path = Path(os.path.abspath(path))
    data = scanreader.read_scan(path)

    layers: List[LayerDataTuple] = []

    layers.append(
        (
            tifffile.imread(path / "downsampled.tiff"),
            {"name": "Registered image", "metadata": metadata},
            "image",
        )
    )
    layers.append(
        (
            tifffile.imread(path / "boundaries.tiff"),
            {
                "name": "Boundaries",
                "blending": "additive",
                "opacity": 0.5,
                "visible": False,
            },
            "image",
        )
    )

    return layers


class TransformPoints(QWidget):
    def __init__(self, viewer: napari.viewer.Viewer):
        """
        Initialize the TransformPoints widget.

        Parameters
        ----------
        viewer : napari.viewer.Viewer
            The napari viewer instance.
            This will be passed when opening the widget.
        """
        super(self).__init__()
        self.viewer = viewer
        self.raw_data = None

        self.image_layer_names = self._get_layer_names(
            layer_type=napari.layers.Image
        )
        self.points_layer_names = self._get_layer_names(
            layer_type=napari.layers.Points
        )
        self.setup_main_layout()

        @self.viewer.layers.events.connect
        def update_layer_list(v: napari.viewer.Viewer) -> None:
            """
            Update internal list of layers whenever the napari layers list
            is updated.

            Parameters
            ----------
            v : napari.viewer.Viewer
                The napari viewer instance.
            """
            self.image_layer_names = self._get_layer_names(
                layer_type=napari.layers.Image
            )
            self.points_layer_names = self._get_layer_names(
                layer_type=napari.layers.Points
            )

            self._update_combobox_options(
                self.raw_data_choice, self.image_layer_names
            )

            self._update_combobox_options(
                self.points_layer_choice, self.points_layer_names
            )

    @staticmethod
    def _update_combobox_options(
            combobox: QComboBox, options_list: List[str]
    ) -> None:
        """
        Update the options in a QComboBox.

        Parameters
        ----------
        combobox : QComboBox
            The combobox to update.
        options_list : List[str]
            The list of options to set in the combobox.
        """

        original_text = combobox.currentText()
        combobox.clear()
        combobox.addItems(options_list)
        combobox.setCurrentText(original_text)

    def _get_layer_names(
            self,
            layer_type: napari.layers.Layer,
            default: str = "",
    ) -> List[str]:
        """
        Get list of layer names of a given layer type.

        Parameters
        ----------
        layer_type : napari.layers.Layer, optional
            The type of layer to get names for.
        default : str, optional
            Default values to include in the list. Default is an empty string.

        Returns
        -------
        List[str]
            A list of layer names.
        """
        layer_names = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, layer_type)
        ]

        if layer_names:
            return [default] + layer_names
        else:
            return [default]

    def setup_main_layout(self) -> None:
        """
        Set up the main layout of the widget.
        """
        self.layout = QGridLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(4)
        # self.add_raw_data_combobox(row=2, column=0)
        # self.add_transform_button(row=3, column=0)
        # self.add_brainrender_export_button(row=3, column=1)
        # self.add_points_summary_table(row=4, column=0)
        # self.add_save_all_points_button(row=6, column=0)
        # self.add_save_points_summary_button(row=6, column=1)
        self.add_status_label(row=7, column=0)

        self.setLayout(self.layout)

    def load_data(self, path, **kwargs) -> None:
        """
        Load the BrainGlobe atlas used for the initial brainreg registration.
        """
        self.data = scanreader.read_scan(path, kwargs)

    def add_points_combobox(self, row: int, column: int) -> None:
        """
        Add a combobox for selecting the points layer containing
        the points (e.g. cells) to transform to a BrainGlobe atlas.

        Parameters
        ----------
        row : int
            Row in the grid layout.
        column : int
            Column in the grid layout.
        """
        self.points_layer_choice, _ = add_combobox(
            self.layout,
            "Points layer",
            self.points_layer_names,
            column=column,
            row=row,
            callback=self.set_points_layer,
        )

    def add_raw_data_combobox(self, row: int, column: int) -> None:
        """
        Add a combobox for selecting the raw data layer. This defines
        the coordinate space for transforming points to a BrainGlobe atlas.

        Parameters
        ----------
        row : int
            Row in the grid layout.
        column : int
            Column in the grid layout.
        """
        self.raw_data_choice, _ = add_combobox(
            self.layout,
            "Raw data layer",
            self.image_layer_names,
            column=column,
            row=row,
            callback=self.set_raw_data_layer,
        )

    def add_transform_button(self, row: int, column: int) -> None:
        """
        Add a button to begin the transformation of points to atlas space.

        Parameters
        ----------
        row : int
            Row in the grid layout.
        column : int
            Column in the grid layout.
        """
        self.transform_button = add_button(
            "Transform points",
            self.layout,
            self.transform_points_to_atlas_space,
            row=row,
            column=column,
            visibility=True,
            tooltip="Transform points layer to atlas space",
        )

    def add_brainrender_export_button(self, row: int, column: int) -> None:
        """
        Add a button to export the points in atlas space in the brainrender
        format.

        Parameters
        ----------
        row : int
            Row in the grid layout.
        column : int
            Column in the grid layout.
        """
        self.brainrender_export_button = add_button(
            "Export to brainrender",
            self.layout,
            self.export_points_to_brainrender,
            row=row,
            column=column,
            visibility=False,
            tooltip="Export points in atlas space to brainrender",
        )

    def add_points_summary_table(self, row: int, column: int) -> None:
        """
        Add a table to display the summary of points per atlas region.

        Parameters
        ----------
        row : int
            Row in the grid layout.
        column : int
            Column in the grid layout.
        """
        self.points_per_region_table_title = QLabel(
            "Points distribution summary"
        )
        self.points_per_region_table_title.setVisible(False)
        self.layout.addWidget(self.points_per_region_table_title, row, column)
        self.points_per_region_table = QTableView()
        self.points_per_region_table.setVisible(False)
        self.layout.addWidget(self.points_per_region_table, row + 1, column)

    def add_save_all_points_button(self, row: int, column: int) -> None:
        """
        Add a button to save all points information (i.e. the list of
        all points, and their coordinates in raw data and atlas
        space, alongside assigned atlas region).

        Parameters
        ----------
        row : int
            Row in the grid layout.
        column : int
            Column in the grid layout.
        """
        self.save_all_points_button = add_button(
            "Save all points information",
            self.layout,
            self.save_all_points_csv,
            row=row,
            column=column,
            visibility=False,
            tooltip="Save all points information as a csv file",
        )

    def add_save_points_summary_button(self, row: int, column: int) -> None:
        """
        Add a button to save points summary (i.e. points per atlas region).

        Parameters
        ----------
        row : int
            Row in the grid layout.
        column : int
            Column in the grid layout.
        """
        self.save_points_summary_button = add_button(
            "Save points summary",
            self.layout,
            self.save_points_summary_csv,
            row=row,
            column=column,
            visibility=False,
            tooltip="Save points summary as a csv file",
        )

    def add_status_label(self, row: int, column: int) -> None:
        """
        Add a status label to inform the user of progress.

        Parameters
        ----------
        row : int
            Row in the grid layout.
        column : int
            Column in the grid layout.
        """
        self.status_label = QLabel()
        self.status_label.setText("Ready")
        self.layout.addWidget(self.status_label, row, column)

    def set_raw_data_layer(self) -> None:
        """
        Set background layer from current background text box selection.
        """
        if self.raw_data_choice.currentText() != "":
            self.raw_data = self.viewer.layers[
                self.raw_data_choice.currentText()
            ]

    def save_all_points_csv(self) -> None:
        """
        Save the coordinate and atlas region
        information for all points to a CSV file.
        """
        self.save_df_to_csv(self.all_points_df)

    def save_points_summary_csv(self) -> None:
        """
        Save the summary of the distribution of points
        in the atlas space to a CSV file.
        """
        self.save_df_to_csv(self.points_per_region_df)

    def save_df_to_csv(self, df: pd.DataFrame) -> None:
        """
        Save the given DataFrame to a CSV file.

        Prompts the user to choose a filename and ensures the file has a
        .csv extension.
        The DataFrame is then saved to the specified file.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be saved.

        Returns
        -------
        None
        """
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Choose filename",
            "",
            "CSV Files (*.csv)",
        )

        if path:
            path = ensure_extension(path, ".csv")
            df.to_csv(path, index=False)


class Paths:
    """
    A class to hold all brainreg-related file paths.

    N.B. this could be imported from brainreg, but it is copied here to
    prevent a circular dependency

    Attributes
    ----------
    brainreg_directory : Path
        Path to brainreg output directory (or brainmapper
        "registration" directory)
    brainreg_metadata_file : Path
        The path to the brainreg metadata (brainreg.json) file
    deformation_field_0 : Path
        The path to the deformation field (0th dimension)
    deformation_field_1 : Path
        The path to the deformation field (1st dimension)
    deformation_field_2 : Path
        The path to the deformation field (2nd dimension)
    downsampled_image : Path
        The path to the downsampled.tiff image file
    volume_csv_path : Path
        The path to the csv file containing region volumes

    Parameters
    ----------
    brainreg_directory : Union[str, Path]
        Path to brainreg output directory (or brainmapper
        "registration" directory)
    """

    def __init__(self, brainreg_directory: Union[str, Path]) -> None:
        """
        Set the paths based on the given brainreg directory

        Parameters
        ----------
        brainreg_directory : Union[str, Path]
            Path to brainreg output directory
            (or brainmapper "registration" directory).
        """
        self.brainreg_directory: Path = Path(brainreg_directory)
        self.brainreg_metadata_file: Path = self.make_filepaths(
            "brainreg.json"
        )
        self.deformation_field_0: Path = self.make_filepaths(
            "deformation_field_0.tiff"
        )
        self.deformation_field_1: Path = self.make_filepaths(
            "deformation_field_1.tiff"
        )
        self.deformation_field_2: Path = self.make_filepaths(
            "deformation_field_2.tiff"
        )
        self.downsampled_image: Path = self.make_filepaths("downsampled.tiff")
        self.volume_csv_path: Path = self.make_filepaths("volumes.csv")

    def make_filepaths(self, filename: str) -> Path:
        """
        Create a full file path by combining the directory with a filename.

        Parameters
        ----------
        filename : str
            The name of the file to create a path for.

        Returns
        -------
        Path
            The full path to the specified file.
        """
        return self.brainreg_directory / filename


class Metadata:
    """
    A class to represent brainreg registration metadata
    (loaded from brainreg.json)

    Attributes
    ----------
    orientation : str
        The orientation of the input data (in brainglobe-space format)
    atlas_string : str
        The BrainGlobe atlas used for brain registration.
    voxel_sizes : List[float]
        The voxel sizes of the input data

    Parameters
    ----------
    brainreg_metadata : Dict[str, Any]
        A dictionary containing metadata information,
        loaded from brainreg.json
    """

    def __init__(self, brainreg_metadata: Dict[str, Any]) -> None:
        """
        Initialize the Metadata instance with brainreg metadata.

        Parameters
        ----------
        brainreg_metadata : Dict[str, Any]
            A dictionary containing metadata information from brainreg.json
        """
        self.orientation: str = brainreg_metadata["orientation"]
        self.atlas_string: str = brainreg_metadata["atlas"]
        self.voxel_sizes: List[float] = brainreg_metadata["voxel_sizes"]

# def run_transform_points_to_downsampled_space(self) -> None:
#     """
#     Transform points fromm the raw data space (in which points
#     were detected) to the downsampled space defined by brainreg.
#     This space is the same as the raw data, but downsampled and realigned
#     to match the orientation and resolution of the atlas.
#     """
#     downsampled_space = self.get_downsampled_space()
#     raw_data_space = self.get_raw_data_space()
#     self.points_in_downsampled_space = raw_data_space.map_points_to(
#         downsampled_space, self.points_layer.data
#     )
#     self.viewer.add_points(
#         self.points_in_downsampled_space,
#         name="Points in downsampled space",
#         visible=False,
#     )

# def run_transform_downsampled_points_to_atlas_space(self) -> None:
#     """
#     Transform points from the downsampled space to atlas space. Uses
#     the deformation fields output by NiftyReg (via brainreg) as a look up.
#     """
#     deformation_field_paths = [
#         self.paths.deformation_field_0,
#         self.paths.deformation_field_1,
#         self.paths.deformation_field_2,
#     ]
#     self.points_in_atlas_space, points_out_of_bounds = (
#         transform_points_from_downsampled_to_atlas_space(
#             self.points_in_downsampled_space,
#             self.atlas,
#             deformation_field_paths,
#             warn_out_of_bounds=False,
#         )
#     )
#     self.viewer.add_points(
#         self.points_in_atlas_space,
#         name="Points in atlas space",
#         visible=True,
#     )

#     if len(points_out_of_bounds) > 0:
#         display_info(
#             self,
#             "Points outside atlas",
#             f"{len(points_out_of_bounds)} "
#             f"points fell outside the atlas space",
#         )

# def get_downsampled_space(self) -> AnatomicalSpace:
#     """
#     Get the anatomical space (as defined by brainglobe-space)
#     for the downsampled data.

#     Returns
#     -------
#     AnatomicalSpace
#         The downsampled anatomical space (as defined by brainglobe-space).
#     """
#     target_shape = tifffile.imread(self.paths.downsampled_image).shape

#     downsampled_space = AnatomicalSpace(
#         self.atlas.orientation,
#         shape=target_shape,
#         resolution=self.atlas.resolution,
#     )
#     return downsampled_space
