import os
import time
from pathlib import Path
import logging
from typing import TYPE_CHECKING, List
import numpy as np
import dask.array as da
import pandas as pd
from scipy import ndimage
from skimage.filters import gaussian
import zarr
import scanreader
from skimage.util import img_as_float

if TYPE_CHECKING:
    import napari

# import mesmerize_core as mc
# from mesmerize_core.caiman_extensions.cnmf import cnmf_cache

from qtpy import QtCore
from magicgui import magic_factory
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtWidgets import (
    QSpacerItem,
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QCheckBox,
    QFileDialog,
    QDialog,
    QComboBox,
    QLabel,
    QLineEdit,
    QGridLayout,
    QTabWidget,
    QSizePolicy,
    QSpinBox,
    # QTextEdit,
    QPlainTextEdit,
)


@magic_factory(auto_call=True)
def mean_image_widget(image: ImageData) -> LabelsData:
    """Threshold an image and return a mask."""
    if image is not None:
        return np.mean(image, axis=0)


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CHUNKS = {0: "auto", 1: -1, 2: -1}

projs = [
    "mean-projection",
    "max-projection",
    "std-projection",
]

VALID_DATA_OPTIONS = (
    "raw-movie",
    "registered-movie",
    "correlation-image",
    *projs,
)


def imreader(path):
    return zarr.open(path, mode="r")


@magic_factory(call_button="Compute")
def mean_image_widget(
        image: "napari.types.ImageData",
) -> "napari.types.LabelsData":
    """Generate thresholded image.

    This pattern uses magicgui.magic_factory directly to turn a function
    into a callable that returns a widget.
    """
    mean = np.mean(image, axis=0)
    update_layer(f'mean_image', mean)


# def get_mcorr_data_mapping(series: pd.Series) -> dict:
#     """
#     Returns dict that maps data option str to a callable that can return the corresponding data array.
#
#     For example, ``{"input": series.get_input_movie}`` maps "input" -> series.get_input_movie
#
#     Parameters
#     ----------
#     series: pd.Series
#         row/item to get mcorr mapping
#
#     Returns
#     -------
#     dict
#         {data label: callable}
#     """
#
#     projections = {k: partial(series.caiman.get_projection, k) for k in projs}
#     m = {
#         "input": series.caiman.get_input_movie,
#         "mcorr": series.mcorr.get_output,
#         "corr": series.caiman.get_corr_image,
#         **projections
#     }
#     return m

# Uncomment below for terminal log messages
# logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(name)s - %(levelname)s - %(message)s')


class QPlainTextEditLogger(logging.Handler):
    def __init__(self):
        super().__init__()
        self.widget = QPlainTextEdit()
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)


class MyDialog(QDialog, QPlainTextEditLogger):
    def __init__(self):
        super().__init__()

        self.logTextBox = QPlainTextEditLogger()
        self.logTextBox.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(self.logTextBox)
        logging.getLogger().setLevel(logging.DEBUG)

        layout = QVBoxLayout()
        # Add the new logging box widget to the layout
        layout.addWidget(self.logTextBox.widget)
        self.setLayout(layout)

    def closeEvent(self, event):
        self.logTextBox.closeEvent(event)
        super().closeEvent(event)


def current_layers():
    return list(current_viewer().layers)


def update_widget_value(widget, event):
    # layers.inserted event has the layer name as value
    widget.value = event.value


def get_zarr_files(directory):
    if not isinstance(directory, (str, os.PathLike)):
        logger.error("iter_zarr_dir requires a single string/path object")
    directory = Path(directory)
    # get directory contents
    contents = [x for x in directory.glob("*/*") if x.is_dir()]
    return [x for x in directory.glob("*") if x.suffix == ".zarr"]


def get_layer_by_name(name, **kwargs):
    """Return Layer() instance that matches input name."""
    layers = current_layers()
    if name not in layers:
        logger.info(f"No layer data found in layer {name}")
        for layer in layers:
            logger.info(f"{layer.name}")
        return None
    else:
        return layers[name]


def update_layer(name, data, **kwargs):
    """
    Update a layer in the viewer with new data.

    If data is None, then the layer is removed.
    If the layer is not present, it's added to the viewer.
    """
    viewer = current_viewer()
    if data is None:
        if name in viewer.layers:
            viewer.layers.pop(name)
        viewer.reset_view()
    elif name not in viewer.layers:
        viewer.add_image(
            data, name=name, rgb=False, multiscale=False, **kwargs
        )
        viewer.reset_view()
    else:
        viewer.layers[name].data = data


def FileDialog(directory="", forOpen=True, ext=""):
    if not directory:
        return None

    directory = Path(directory).expanduser()

    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    options |= QFileDialog.DontUseCustomDirectoryIcons
    dialog = QFileDialog()
    dialog.setOptions(options)

    dialog.setFileMode(QFileDialog.AnyFile)
    (
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        if forOpen
        else dialog.setAcceptMode(QFileDialog.AcceptSave)
    )

    # SET FORMAT, IF SPECIFIED
    if ext != "" and directory.is_dir() is False:
        dialog.setDefaultSuffix(ext)
        dialog.setNameFilters([f"{ext} (*.{ext})"])

    # SET THE STARTING DIRECTORY
    if directory != "":
        dialog.setDirectory(str(directory))
    else:
        dialog.setDirectory(str(ROOT_DIR))

    if dialog.exec_() == QDialog.Accepted:
        path = dialog.selectedFiles()  # returns a list
        return path
    else:
        return ""


# LBM_DEFAULT_PATH = os.environ.get('LBM_DEFAULT_PATH', Path().home() / 'caiman_data' / 'high_res')
LBM_DEFAULT_PATH = "C:/Users/RBO/caiman_data/"


class LBMWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "um"

        self.reg_results = {}
        self.path = LBM_DEFAULT_PATH
        self.scan = None
        self.metadata = {}
        self.selected_planes = []
        self.selected_files = []
        self.current_reg_selection = None
        self.save_path = None
        self.processed_dir = None
        self.data_store = {}

        # called when a layer is inserted
        # callback = self.viewer.layers.events.inserted.connect(partial(update_widget_value, self._image_layer))

        self.downsample_fact = 64
        self.trim_x = (0, 0)
        self.trim_y = (0, 0)
        self.plane = 1

        title = QLabel("Light Beads Microscopy Pipeline")
        title.setAlignment(QtCore.Qt.AlignCenter)

        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.tabs.addTab(self.rawTabUI(), "Raw Data")
        self.tabs.addTab(self.processedTabUI(), "Processed Data")
        self.tabs.addTab(self.registrationTabUI(), "Registration")
        self.currently_selected_tab = self.tabs.currentWidget()
        #
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.selection_widget = self.SelectionWidget()
        self.selection_widget.layer_combo.reset_choices()

        self.viewer.layers.events.inserted.connect(
            self.selection_widget.layer_combo.reset_choices
        )
        self.viewer.layers.events.removed.connect(
            self.selection_widget.layer_combo.reset_choices
        )
        #

        # self.dlg = MyDialog()
        # self.dlg.show()
        self.setLayout(layout)
        self.layout().addWidget(title)
        self.layout().addWidget(self.tabs)
        # self.layout().addWidget(self.dlg)

    def set_scan_phase(self):
        if self.fix_scan_phase_checkbox.isChecked():
            self.scan.fix_scan_phase = True
            update_layer(
                f"raw_data_plane_{self.plane}",
                self.scan[:: self.downsample_fact, self.plane, :, :],
            )
        else:
            self.scan.fix_scan_phase = False
            update_layer(
                f"raw_data_plane_{self.plane}",
                self.scan[:: self.downsample_fact, self.plane, :, :],
            )

    def rawTabUI(self):
        """Create the General page UI."""
        generalTab = QWidget()
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins to 0
        outer_layout.setSpacing(0)  # Reduce spacing to 0

        self.open_button = QPushButton(text="Load Data")
        self.open_button.clicked.connect(self.initialize_raw_scan)

        self.save_button = QPushButton(text="Save Data")
        self.save_button.clicked.connect(self.save_data)

        verticalSpacer = QSpacerItem(
            20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding
        )

        self.setWindowTitle("LBM Raw Processing Preview")

        dlgLayout = QVBoxLayout()
        # Create a form layout and add widgets

        self.trim_x_spinbox = QSpinBox()
        self.trim_x_spinbox.setSuffix(" pixels")
        self.trim_x_spinbox.setMinimum(0)
        self.trim_x_spinbox.setMaximum(8)
        self.trim_x_spinbox.valueChanged.connect(self.change_x_trim)

        self.trim_y_spinbox = QSpinBox()
        self.trim_y_spinbox.setSuffix(" pixels")
        self.trim_y_spinbox.setMinimum(0)
        self.trim_y_spinbox.setMaximum(8)
        self.trim_y_spinbox.valueChanged.connect(self.change_y_trim)

        self.ds_combobox = QComboBox()
        self.ds_combobox.addItems(["2x", "8x", "16x", "32x", "64x", "128x"])
        self.ds_combobox.setCurrentText(f"{self.downsample_fact}x")
        self.ds_combobox.currentTextChanged.connect(
            self.change_downsample_fact
        )

        self.plane_spinbox = QSpinBox()
        self.plane_spinbox.setMinimum(1)
        self.plane_spinbox.setMaximum(30)
        self.plane_spinbox.valueChanged.connect(self.change_plane)

        self.fix_scan_phase_checkbox = QCheckBox("")
        self.fix_scan_phase_checkbox.stateChanged.connect(self.set_scan_phase)

        formLayout = QFormLayout()

        formLayout.addRow(self.open_button)
        formLayout.addRow("Plane: ", self.plane_spinbox)
        formLayout.addRow("Trim X: ", self.trim_x_spinbox)
        formLayout.addRow("Trim Y: ", self.trim_y_spinbox)
        formLayout.addRow("Downsample Ratio: ", self.ds_combobox)
        formLayout.addRow("Fix Scan Phase: ", self.fix_scan_phase_checkbox)
        formLayout.addRow(self.save_button)

        self.mean_image_button = QPushButton(text="Compute Mean Image")
        self.mean_image_button.clicked.connect(self.compute_mean_image)

        self.fix_scan_phase_checkbox.setEnabled(False)
        self.trim_x_spinbox.setEnabled(False)
        self.trim_y_spinbox.setEnabled(False)
        self.ds_combobox.setEnabled(False)
        self.ds_combobox.setEnabled(False)
        self.plane_spinbox.setEnabled(False)
        self.save_button.setEnabled(False)
        self.mean_image_button.setEnabled(False)

        dlgLayout.addLayout(formLayout)

        outer_layout.addLayout(dlgLayout)

        generalTab.setLayout(outer_layout)
        return generalTab

    def save_data(self):
        if not self.scan:
            logging.info("File cannot be saved, scan not yet initialized.")
            return None

        # Get the directory to save the file
        selection = QFileDialog.getExistingDirectory(
            self, "Save File: Choose a Directory"
        )
        if not selection:
            logging.info("No selection made.")
            return None  # Exit if no directory was selected

        self.savepath = selection

        # Create and set up the dialog
        dialog = QDialog()
        dialog.setWindowTitle("Choose planes to save")

        def accept():
            self.selected_planes = [
                i + 1
                for i, checkbox in enumerate(self.plane_checkboxes)
                if checkbox.isChecked()
            ]
            dialog.accept()
            self.save_as_zarr()

        def reject():
            dialog.reject()

        dialog_layout = QVBoxLayout()
        dialog.setLayout(dialog_layout)

        message = QLabel("Select planes")
        dialog_layout.addWidget(message)

        # Create a layout for checkboxes
        plane_layout = QVBoxLayout()
        select_all_checkbox = QCheckBox("Select all")
        select_all_checkbox.stateChanged.connect(self.set_checkbox_checked)

        self.plane_checkboxes = []
        layout = QGridLayout()
        # add select all checkbox to the first row
        row = 0
        col = 0
        layout.addWidget(
            select_all_checkbox, row, col, 1, 5
        )  # span the checkbox across 5 columns

        # loop through planes and add checkboxes in columns
        for i, plane in enumerate(range(1, self.scan.num_planes + 1)):
            checkbox = QCheckBox(f"Plane {plane}")
            row = (i // 5) + 1  # start from the second row
            col = i % 5
            layout.addWidget(checkbox, row, col)
            self.plane_checkboxes.append(checkbox)

        dialog_layout.addLayout(layout)

        # Create Apply and Cancel buttons
        apply_button = QPushButton("Apply")
        cancel_button = QPushButton("Cancel")

        button_layout = QHBoxLayout()
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)
        dialog_layout.addLayout(button_layout)

        apply_button.clicked.connect(accept)
        cancel_button.clicked.connect(reject)

        dialog.exec_()

    def iter_planes(self, selected_planes):
        for plane in selected_planes:
            yield da.squeeze(self.scan[:, plane, :, :])

    def save_as_zarr(self, overwrite=False):
        iterator = self.iter_planes(self.selected_planes)
        logging.info(f"Selected planes: {self.selected_planes}")
        outer = time.time()
        for idx, array in enumerate(iterator):
            url = Path(self.savepath) / f'{idx + 1}'
            start = time.time()
            try:
                da.to_zarr(
                    arr=array,
                    url=self.savepath,
                    component=f"mov",
                    overwrite=overwrite,
                )
            except ContainsArrayError:
                logging.info(f"Plane {idx + 1} already exists. Skipping...")
                continue
            # root['preprocessed'][f'plane_{idx+1}'].attrs['fps'] = self.metadata['fps']
            logging.info(f"Plane saved in {time.time() - start} seconds...")
        logging.info(f"All z-planes saved in {time.time() - outer} seconds...")

    def set_checkbox_checked(self, state):
        for checkbox in self.plane_checkboxes:
            checkbox.setChecked(state == 2)

    def compute_mean_image(self):
        viwer = self.viewer
        name = f"mean_image_plane_{self.plane}"
        data = np.mean(
            self.scan[:: self.downsample_fact, self.plane, :, :], axis=0
        )
        return update_layer(name, data)

    def show_z_stack(self, t=0):
        shape = None
        if not self.data_store:
            # fill the store
            selection = QFileDialog.getExistingDirectory(
                self, "Set data directory: Choose a Directory"
            )
            files = [x for x in Path(selection).glob("*.zarr*")]
            for idx, filename in enumerate(files):
                self.data_store[filename] = zarr.open(filename)
                if idx == 0:
                    shape = da.squeeze(self.data_store[filename]).shape

        zstack = []
        for filename, array in self.data_store.items():
            zstack.append(array[t])
        zstack = da.stack(zstack, axis=0).squeeze()
        zstep = 16
        pixel_size = 1
        update_layer(f"zstack_timepoint_{t}", zstack, scale=[16, 1, 1])

    def processedTabUI(self):
        """Create the Network page UI."""

        processedTab = QWidget()
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins to 0
        outer_layout.setSpacing(0)  # Reduce spacing to 0

        self.p_open_button = QPushButton(text="Load Processed Data")
        self.p_open_button.clicked.connect(self.load_processed_data)
        verticalSpacer = QSpacerItem(
            20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding
        )

        self.setWindowTitle("LBM Raw Processing Preview")

        dlgLayout = QVBoxLayout()
        # Create a form layout and add widgets

        self.p_ds_combobox = QComboBox()
        self.p_ds_combobox.addItems(["2x", "8x", "16x", "32x", "64x", "128x"])
        self.p_ds_combobox.setCurrentText(f"{self.downsample_fact}x")
        self.p_ds_combobox.currentTextChanged.connect(
            self.change_downsample_fact
        )

        self.p_plane_spinbox = QSpinBox()
        self.p_plane_spinbox.setMinimum(1)
        self.p_plane_spinbox.setMaximum(30)
        self.p_plane_spinbox.valueChanged.connect(self.change_p_plane)

        ##%
        checkbox_layout = QHBoxLayout()

        zstack_compute_button = QCheckBox("Render Z-Stack")
        zstack_compute_button.stateChanged.connect(self.show_z_stack)

        zstack_spinbox = QSpinBox()
        zstack_spinbox.setMinimum(1)
        zstack_spinbox.setMaximum(200)
        # zstack_spinbox.valueChanged.connect(self.show_z_stack) # TODO:

        checkbox_layout.addWidget(zstack_spinbox)
        checkbox_layout.addWidget(zstack_compute_button)

        self.save_current_selection_button = QPushButton(
            text="Save Current Selection"
        )
        self.save_current_selection_button.clicked.connect(
            self.save_current_selection
        )

        self.p_ds_combobox.setEnabled(False)
        self.p_plane_spinbox.setEnabled(False)
        self.save_current_selection_button.setEnabled(False)

        formLayout = QFormLayout()

        formLayout.addRow(self.p_open_button)
        formLayout.addRow("Plane: ", self.p_plane_spinbox)
        formLayout.addRow("Downsample Ratio: ", self.p_ds_combobox)
        formLayout.addRow(self.save_current_selection_button)
        formLayout.addRow(checkbox_layout)

        dlgLayout.addLayout(formLayout)
        # dlgLayout.addWidget(self.mean_image_button)

        outer_layout.addLayout(dlgLayout)
        processedTab.setLayout(outer_layout)

        return processedTab

    def save_current_selection(self):
        current_selection = self.viewer.layers.selection
        if not current_selection:
            logging.info("No layers selected.")
            return None

        # Get the directory to save the file
        selection = QFileDialog.getExistingDirectory(
            self, "Save File: Choose a Directory"
        )

        if not selection:
            logging.info("No selection made.")
            return None

        # open dialog where user can enter savename and filetype
        dialog = QDialog()
        dialog.setWindowTitle("Save Selection")

        # 3 checkboxes, zarr, tiff, hdfy
        zarr_checkbox = QCheckBox("Save as Zarr")
        tiff_checkbox = QCheckBox("Save as Tiff")
        hdf5_checkbox = QCheckBox("Save as HDF5")

        # Create Apply and Cancel buttons
        apply_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")

        button_layout = QFormLayout()
        button_layout.addRow(zarr_checkbox)
        button_layout.addRow(tiff_checkbox)
        button_layout.addRow(hdf5_checkbox)
        button_layout.addRow(apply_button)
        button_layout.addRow(cancel_button)

        filename = QLineEdit()
        button_layout.addRow(f"Filename: ", filename)

    def registrationTabUI(self):
        """Create the Network page UI."""

        if os.name == "nt":
            # disable the cache on windows, this will be automatic in a future version
            cnmf_cache.set_maxsize(0)

        if not self.current_reg_selection:
            self.current_reg_selection = VALID_DATA_OPTIONS[0]

        registrationTab = QWidget()

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins to 0
        outer_layout.setSpacing(0)  # Reduce spacing to 0

        self.run_registration_button = QPushButton(text="Run Registration")
        self.run_registration_button.clicked.connect(self.run_reg)

        self.load_registration_button = QPushButton(
            text="Load Registration Results"
        )
        self.load_registration_button.clicked.connect(self.load_reg)

        self.setWindowTitle("Registration Results")

        dlgLayout = QVBoxLayout()
        # Create a form layout and add widgets

        self.reg_combobox = QComboBox()
        self.reg_combobox.addItems(VALID_DATA_OPTIONS)
        # self.reg_combobox.setCurrentText(f"{self.current_reg_selection}")
        self.reg_combobox.currentTextChanged.connect(self.change_reg_selection)

        self.reg_plane_spinbox = QSpinBox()
        self.reg_plane_spinbox.setMinimum(1)
        self.reg_plane_spinbox.setMaximum(30)
        self.reg_plane_spinbox.valueChanged.connect(self.change_p_plane)

        formLayout = QFormLayout()

        formLayout.addRow("Result :", self.reg_combobox)
        formLayout.addRow("Plane :", self.reg_plane_spinbox)

        self.reg_combobox.setEnabled(False)
        self.reg_plane_spinbox.setEnabled(False)

        outer_layout.addWidget(self.run_registration_button)
        outer_layout.addWidget(self.load_registration_button)
        dlgLayout.addLayout(formLayout)
        outer_layout.addLayout(dlgLayout)
        registrationTab.setLayout(outer_layout)

        return registrationTab

    def run_reg(self):
        pass

    def load_reg(self):

        selection = QFileDialog.getExistingDirectory(
            self,
            caption="Select registration output directory",
            directory=self.save_path,
        )

        if not selection:
            logging.info("No selection made.")
            return None  # Exit if no directory was selected

        self.batch_path = Path(selection)
        if not self.batch_path:
            return None

        batch_files = [x for x in self.batch_path.glob("*.pickle*")]

        if len(batch_files) == 0:
            logging.info("No batch files found in directory.")
            return None
        logger.info(f"Located batch files: {batch_files}")
        self.df = pd.DataFrame()
        # self.df = mc.load_batch(batch_files[0])
        logger.info(self.df.head())

        # self.input_movie = df.iloc[0].caiman.get_input_movie(reader=imreader)
        # self.mcorr_movie = df.iloc[0].mcorr.get_output()

        self.reg_results = {
            "mean-projection": self.df.iloc[0].caiman.get_projection("mean"),
            "max-projection": self.df.iloc[0].caiman.get_projection("max"),
            "std-projection": self.df.iloc[0].caiman.get_projection("std"),
            "correlation-image": self.df.iloc[0].caiman.get_corr_image(),
        }

        self.reg_combobox.setEnabled(True)
        self.reg_combobox.setEnabled(True)
        self.reg_plane_spinbox.setEnabled(True)

    def change_reg_selection(self, text):
        logger.debug(f"Changing reg selection to {text}")

        self.current_reg_selection = text
        # res = self.reg_results[text]
        if text == "raw-movie":
            data = self.df.iloc[0].caiman.get_input_movie(reader=imreader)[
                f"plane_{self.plane}"
            ]
            update_layer(f"raw_plane_{self.plane}", data)
        if text == "registered-movie":
            raise NotImplementedError()
            # update_layer(f'registered_plane_{self.plane}', self.df.iloc[0].mcorr.get_output_path())
        else:
            update_layer(f"{text}_plane_{self.plane}", self.reg_results[text])

    def load_processed_data(self):
        ext = "zarr"  # TODO: allow multiple save formats
        # Get the directory to save the file
        selection = QFileDialog.getExistingDirectory(
            self, "Set data directory: Choose a Directory"
        )
        if not selection:
            logging.info("No selection made.")
            return None  # Exit if no directory was selected

        self.processed_dir = Path(selection)
        if not self.processed_dir:
            return None

        files = [x for x in self.processed_dir.glob("*.zarr*")]

        # for the first load, sort files by the plane number
        files = sorted(files, key=lambda x: int(x.stem.split("_")[-1]))

        # if a single zarr file is selected, the path will be '/_plane_N.zarr/.zarray', so get the parent
        if files[0].parent.suffix == ".zarr":
            files = [x.parent for x in files]  # list comp likely not needed
        self.selected_files.extend(files)
        logger.debug(f"Located files: {files}")

        for idx, file in enumerate(files):
            if idx == 0:
                self.plane_spinbox.setMaximum(len(files))
            name = file.stem  # strip the .zarr to match the layer 'name'
            data = da.from_zarr(file)
            update_layer(f"processed_data_plane_{self.plane}", data)
            self.data_store[name] = data

        link_layers(self.viewer.layers)
        self.p_ds_combobox.setEnabled(True)
        self.p_plane_spinbox.setEnabled(True)
        self.save_current_selection_button.setEnabled(True)

    def initialize_raw_scan(self):
        path = FileDialog(self.path, ext="tif")
        if not path:
            logger.info("No files found... searching up one level.")
            # TODO: my god, make this recursive
            path = [
                x
                for x in self.path.glob("*.tif*")
                if "__plane__" not in str(x)
            ]
            if len(path) == 0:
                all = [
                    x
                    for x in self.path.glob("*.*")
                    if "__plane__" not in str(x)
                ]
                for x in all:
                    if Path(x).is_dir():
                        path = [
                            x
                            for x in self.path.glob("*.tif*")
                            if "__plane__" not in str(x)
                        ]
                        if len(path) == 0:
                            raise FileNotFoundError("Try again!")

        if not isinstance(path, (list, tuple)):
            path = [path]

        logger.info("Initializing scan...")
        self.scan = scanreader.ScanLBM(path)
        self.plane_spinbox.setMaximum(self.scan.num_planes)
        logger.info("Scan initialized. Updating napari...")
        self.scan.trim_x = self.trim_x
        self.scan.trim_y = self.trim_y
        self.metadata = self.scan.metadata

        self.fix_scan_phase_checkbox.setEnabled(True)
        self.trim_x_spinbox.setEnabled(True)
        self.trim_y_spinbox.setEnabled(True)
        self.ds_combobox.setEnabled(True)
        self.plane_spinbox.setEnabled(True)
        self.save_button.setEnabled(True)

        update_layer(
            f"raw_data_plane_{self.plane}",
            self.scan[:: self.downsample_fact, self.plane, :, :],
        )

    def update_scan(self):
        update_layer(
            f"raw_data_plane_{self.plane}",
            self.scan[:: self.downsample_fact, self.plane, :, :],
        )

    def change_x_trim(self, i):
        self.scan.trim_x = (i, i)
        if self.scan:
            self.update_scan()

    def change_y_trim(self, i):
        self.scan.trim_y = (i, i)
        if self.scan:
            self.update_scan()

    def change_downsample_fact(self, text):
        self.downsample_fact = int(text[:-1])
        if self.scan:
            self.update_scan()

    def change_p_plane(self, i):
        self.plane = i
        file = self.selected_files[i]
        update_layer(f"processed_data_plane_{self.plane}", da.from_zarr(file))

    def change_plane(self, i):
        # find the active tab

        self.plane = i
        if self.scan:
            self.update_scan()

    class SelectionWidget(QWidget):
        """A custom widget class."""

        def __init__(self) -> None:
            super().__init__()

            title = QLabel("Layer Selection")
            title.setAlignment(QtCore.Qt.AlignCenter)

            self.mean_image_button = QPushButton(text="Mean Image")
            self.mean_image_button.clicked.connect(self.run_mean_image)

            self.sobel_filter_button = QPushButton(text="Edge Detection")
            self.sobel_filter_button.clicked.connect(self.run_edge_detection)

            self.layer_combo = create_widget(annotation=Layer)

            formLayout = QFormLayout()
            formLayout.addRow(title)
            formLayout.addRow("Layer", self.layer_combo.native)
            formLayout.addRow(self.mean_image_button)
            formLayout.addRow(self.sobel_filter_button)

            self.setLayout(formLayout)

        def run_mean_image(self):
            layer = self.layer_combo.value
            update_layer(
                f"mean_image_plane_{int(layer.name[-1])}",
                da.mean(layer.data, axis=0),
            )

        def sobel_h(self, image):
            return ndimage.sobel(image)

        def run_edge_detection(self):
            self.layer_combo

            layer = self.layer_combo.value
            data = layer.data.astype("int32").squeeze()
            tile_map = da.map_blocks(self.sobel_h, data)
            name = f"edges_plane_{int(layer.name[-1])}"
            update_layer(name, tile_map)
            link_layers(current_viewer().layers)
