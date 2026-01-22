# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:14:30 2022

@author: Ruben
"""

import sys, functools, os, glob, h5py, pywt, traceback
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import pyplot as plt, ticker as mticker
import matplotlib as mpl
from pathlib import Path
from time import strftime, sleep

print = functools.partial(print, flush=True) # Re-implement print to fix issue where print statements do not show in console until after script execution completes

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools import get_paths, printProgressBar, get_logbook, get_logbook_data, get_cwt_scales

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

debug = False  # Set to True for detailed CWT debugging output

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # Setting for whether or not to display axes on the plotted CWT scalogram
        # Set to False for generating training data
        self.show_axes = False
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("CWT labeller")
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.setFixedWidth(1000)
        self.setFixedHeight(1000)
        
        # Create widgets in sublayout
        # Load data
        self.btnStart = QPushButton('Load data')

        # CWT settings - Row 1: Basic window settings
        pdLabel = QLabel('Photodiode channel:')
        wlLabel = QLabel('Window length:')
        msUnit1 = QLabel('ms')
        woLabel = QLabel('Window offset:')
        msUnit2 = QLabel('ms')
        nfLabel = QLabel('Frequency steps:')

        self.pdChannelSelector = QLineEdit('1')
        self.windowLengthSelector = QLineEdit('1')
        self.windowOffsetSelector = QLineEdit('1')
        self.freqsSelector = QLineEdit('256')

        # CWT settings - Row 2: Wavelet, computation mode, and COI
        wsLabel = QLabel('Wavelet:')
        cmLabel = QLabel('Colourmap:')
        cwtModeLabel = QLabel('CWT Mode:')

        self.waveletSelector = QComboBox()
        wavelet_names = get_cwt_scales(None)
        for n in sorted(wavelet_names):
            self.waveletSelector.addItem(n)

        self.cmapSelector = QComboBox()
        cmaps = ['grey', 'grey_r', 'jet', 'magma', 'plasma', 'viridis']
        for c in cmaps:
            self.cmapSelector.addItem(c)

        # CWT computation mode radio buttons
        self.cwtModeFullSignal = QRadioButton('Full Signal')
        self.cwtModePerWindow = QRadioButton('Per-Window')
        self.cwtModeFullSignal.setChecked(True)  # Default to full signal

        # COI masking checkbox
        self.coiMaskingCheckbox = QCheckBox('COI Masking')
        self.coiMaskingCheckbox.setToolTip('Set Cone of Influence edge artifacts to 0')

        # Global vmax calculation checkbox
        self.globalVmaxCheckbox = QCheckBox('Auto vmax')
        self.globalVmaxCheckbox.setToolTip('Calculate vmax automatically from dataset')

        # Setup Row 1: Load data + basic window parameters
        inputsRow1 = QGroupBox('Setup - Basic')
        inputsLayout1 = QHBoxLayout()
        inputsLayout1.addWidget(self.btnStart, stretch=3)
        inputsLayout1.addStretch(2)
        inputsLayout1.addWidget(pdLabel, stretch=3)
        inputsLayout1.addWidget(self.pdChannelSelector, stretch=2)
        inputsLayout1.addWidget(wlLabel, stretch=3)
        inputsLayout1.addWidget(self.windowLengthSelector, stretch=2)
        inputsLayout1.addWidget(msUnit1, stretch=1)
        inputsLayout1.addWidget(woLabel, stretch=3)
        inputsLayout1.addWidget(self.windowOffsetSelector, stretch=2)
        inputsLayout1.addWidget(msUnit2, stretch=1)
        inputsLayout1.addWidget(nfLabel, stretch=3)
        inputsLayout1.addWidget(self.freqsSelector, stretch=2)
        inputsRow1.setLayout(inputsLayout1)

        # Setup Row 2: Wavelet settings + computation mode + COI
        inputsRow2 = QGroupBox('Setup - Advanced')
        inputsLayout2 = QHBoxLayout()
        inputsLayout2.addWidget(wsLabel, stretch=2)
        inputsLayout2.addWidget(self.waveletSelector, stretch=3)
        inputsLayout2.addWidget(cmLabel, stretch=2)
        inputsLayout2.addWidget(self.cmapSelector, stretch=3)
        inputsLayout2.addWidget(cwtModeLabel, stretch=2)
        inputsLayout2.addWidget(self.cwtModeFullSignal, stretch=2)
        inputsLayout2.addWidget(self.cwtModePerWindow, stretch=2)
        inputsLayout2.addStretch(1)
        inputsLayout2.addWidget(self.coiMaskingCheckbox, stretch=2)
        inputsLayout2.addWidget(self.globalVmaxCheckbox, stretch=2)
        inputsRow2.setLayout(inputsLayout2)
        
        # Auto labelling inputs
        self.autoLabelPath = QLineEdit(r'F:\AlSi10Mg single layer ffc\CWT_labelled_windows\1.0ms-window_0.2ms_offset_data_labels.csv')
        self.btnAuto = QPushButton('Run auto')
        self.btnAuto.setEnabled(False)
        self.btnCancelAuto = QPushButton('Cancel')
        self.btnCancelAuto.setEnabled(False)

        autoRow = QGroupBox('Auto labelling')
        autoLayout = QHBoxLayout()
        autoLayout.addWidget(self.autoLabelPath, stretch=6)
        autoLayout.addWidget(self.btnAuto, stretch=1)
        autoLayout.addWidget(self.btnCancelAuto, stretch=1)
        autoRow.setLayout(autoLayout)
        
        # Radiograph image display    
        self.xrayFig = plt.figure(frameon=False)
        self.xrayCanvas = FigureCanvas(self.xrayFig)
        
        self.figRow = QGroupBox()
        figLayout = QHBoxLayout()
        figLayout.addWidget(self.xrayCanvas, stretch=1)
        self.figRow.setLayout(figLayout)
        
        # Controls
        self.btnRow = QGroupBox('Manual labelling')
        btnRowLayout = QHBoxLayout()
        self.buttons = {}
        for label in ['Previous', 'Go back', 'Label 0', 'Label 1', 'Skip', 'Next']:
            self.buttons[label] = QPushButton(label)
            self.buttons[label].setEnabled(False)
            btnRowLayout.addWidget(self.buttons[label])
        self.btnRow.setLayout(btnRowLayout)
        
        # Progress bar
        self.progressBar = QProgressBar()
        
        # Text readout
        self.readout = QLineEdit()
        self.readout.setReadOnly(True)
        
        # Combine GUI elements into main window layout
        layout = QVBoxLayout()
        layout.addWidget(inputsRow1, stretch=1)
        layout.addWidget(inputsRow2, stretch=1)
        layout.addWidget(autoRow, stretch=1)
        layout.addWidget(self.figRow, stretch=20)
        layout.addWidget(self.btnRow, stretch=1)
        layout.addWidget(self.progressBar, stretch=1)
        layout.addWidget(self.readout, stretch=1)
        self.centralWidget.setLayout(layout)
        
    def update_progress(self, perc, text):
        self.progressBar.setValue(perc)
        self.set_readout_text(text)
        
    def grey_out_load(self):
        self.btnStart.setEnabled(False)
        
    def enable_controls(self, mode=True):
        for label in self.buttons:
            self.buttons[label].setEnabled(mode)
        self.btnAuto.setEnabled(mode)
        self.btnCancelAuto.setEnabled(False)  # Cancel button managed separately
        
    def set_readout_text(self, text):
        self.readout.setText(text)
        
    def closeEvent(self, event):
        QApplication.closeAllWindows()
        event.accept()
        
class VideoPlayerWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player")
        self.setGeometry(0, 0, 800, 600)
        
        # Create main layout
        self.layout = QVBoxLayout(self)

        # Create video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)
        
        # Create controls layout
        controls_layout = QHBoxLayout()
        
        # Play/Pause button
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)
        
        # Previous frame button
        self.prev_frame_button = QPushButton()
        self.prev_frame_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        self.prev_frame_button.clicked.connect(self.previous_frame)
        controls_layout.addWidget(self.prev_frame_button)
        
        # Next frame button
        self.next_frame_button = QPushButton()
        self.next_frame_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.next_frame_button.clicked.connect(self.next_frame)
        controls_layout.addWidget(self.next_frame_button)
        
        # Frame counter label
        self.frame_label = QLabel("Frame: 0")
        controls_layout.addWidget(self.frame_label)
        
        # FPS selector
        self.fps_label = QLabel("Speed:")
        self.fps_selector = QComboBox()
        self.fps_selector.addItems(["5 FPS", "15 FPS", "30 FPS", "60 FPS"])
        self.fps_selector.setCurrentText("30 FPS")  # Default to 30 FPS
        self.fps_selector.currentTextChanged.connect(self.change_fps)
        controls_layout.addWidget(self.fps_label)
        controls_layout.addWidget(self.fps_selector)
        
        # Add stretch to push controls to the left
        controls_layout.addStretch()
        
        # Add controls layout to main layout
        self.layout.addLayout(controls_layout)
        
        # Initialize playback variables
        self.current_frame = 0
        self.playing = False
        self.fps = 30  # Default FPS
        self.frame_data = None
        
        # Setup playback timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Connect mouse tracking
        self.video_label.setMouseTracking(True)

    def change_fps(self, fps_text):
        """Update playback speed"""
        self.fps = int(fps_text.split()[0])
        if self.playing:
            self.timer.start(int(1000 / self.fps))
        
    def set_data(self, frame_data):
        """
        Set the video data from numpy array
        Args:
            frame_data (numpy.ndarray): Array of shape (frames, height, width) with uint8 type
        """
        assert frame_data.dtype == np.uint8, "Data must be uint8"
        assert len(frame_data.shape) == 3, "Data must be 3D array (frames, height, width)"
        
        self.frame_data = frame_data
        self.current_frame = 0
        
        # Start playing automatically
        self.playing = True
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.timer.start(int(1000 / self.fps))
        
        self.update_frame()
        
    def update_frame(self):
        """Display the current frame"""
        if self.frame_data is None:
            return
            
        # Handle end of video (loop back to start)
        if self.current_frame >= len(self.frame_data):
            self.current_frame = 0
            
        # Get current frame data
        frame = self.frame_data[self.current_frame]
        height, width = frame.shape
        
        # Convert numpy array to QImage
        qimg = QImage(frame.data, width, height, width, QImage.Format_Grayscale8)
        
        # Scale the image to fit the label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update frame counter
        self.frame_label.setText(f"Frame: {self.current_frame}")
        
        # Increment frame counter if playing
        if self.playing:
            self.current_frame += 1
            
    def toggle_play(self):
        """Toggle between play and pause"""
        self.playing = not self.playing
        if self.playing:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.timer.start(int(1000 / self.fps))
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.stop()
            
    def stop(self):
        """Stop playback"""
        self.playing = False
        self.timer.stop()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            
    def next_frame(self):
        """Advance to next frame"""
        if self.frame_data is not None:
            self.playing = False
            self.timer.stop()
            self.current_frame = (self.current_frame + 1) % len(self.frame_data)
            self.update_frame()
        
    def previous_frame(self):
        """Go back one frame"""
        if self.frame_data is not None:
            self.playing = False
            self.timer.stop()
            self.current_frame = (self.current_frame - 1) % len(self.frame_data)
            self.update_frame()

class CWTWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CWT Display")
        
        # Set a fixed, comfortable viewing size
        self.WINDOW_WIDTH = 400  # Larger for better visibility
        self.WINDOW_HEIGHT = 330
        self.setFixedSize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        
        # Create main layout with no margins
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)  # Small margins for aesthetics
        
        # Create the canvas for CWT
        self.cwtFig = plt.figure(frameon=False)
        self.cwtCanvas = FigureCanvas(self.cwtFig)
        
        # Make canvas expand to fill available space
        self.cwtCanvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.cwtCanvas, stretch=1)  # stretch=1 makes it expand
        
        # Info label with fixed height
        self.info_label = QLabel("CWT Scalogram")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setFixedHeight(25)  # Fixed height
        self.layout.addWidget(self.info_label, stretch=0)  # stretch=0 keeps it fixed size
        
    def resize_to_fill_window(self):
        """Make the figure fill the available canvas space, accounting for high DPI"""
        
        # Get the device pixel ratio (this is key for high DPI displays)
        device_pixel_ratio = self.devicePixelRatio()
        print(f"Device pixel ratio: {device_pixel_ratio}")
        
        # Get the actual canvas size in logical pixels
        canvas_size = self.cwtCanvas.size()
        logical_width = canvas_size.width()
        logical_height = canvas_size.height()
        
        print(f"Canvas logical size: {logical_width} x {logical_height}")
        
        # Calculate physical pixels
        physical_width = logical_width * device_pixel_ratio
        physical_height = logical_height * device_pixel_ratio
        
        print(f"Canvas physical size: {physical_width} x {physical_height}")
        
        # Method A: Set figure size to match physical canvas size
        # Get the canvas DPI (which should account for high DPI scaling)
        canvas_dpi = self.cwtCanvas.figure.dpi
        print(f"Canvas DPI: {canvas_dpi}")
        
        # Calculate figure size in inches to fill the canvas
        fig_width_inches = physical_width / canvas_dpi
        fig_height_inches = physical_height / canvas_dpi
        
        print(f"Setting figure size to: {fig_width_inches} x {fig_height_inches} inches")
        
        # Set the figure size
        self.cwtFig.set_size_inches(fig_width_inches, fig_height_inches)
        
        # Force redraw
        self.cwtCanvas.draw()
        
    def update_info(self, trackid, window_start, window_end):
        """Update the info label with current window information"""
        self.info_label.setText(f"Track: {trackid} | Window: {window_start}-{window_end} ms")
        
    
class Controller(QObject):
    updateReadout = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    
    def __init__(self, view, videoPlayer, cwtWindow):
        super().__init__()
        print(f'Controller running on thread: {int(QThread.currentThreadId())} (main)')
        self.view = view   # Define the view for access from within the Controller object
        self.video_player = videoPlayer
        self.cwt_window = cwtWindow
        self.connect_signals()
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        # Variables to track navigation through datasets
        self.nFiles = 0
        self.fIndex = -1
        self.trackid = ''
        self.wIndex = 0
        self.wStart = 0
        self.wEnd = 0
        self.label = ''
        # Get PD channel name
        self.pdChannelLong = f'Photodiode{self.view.pdChannelSelector.text()}Bits'
        self.pdChannelShort = f'PD{self.view.pdChannelSelector.text()}'
        # CWT settings
        self.windowLength = float(self.view.windowLengthSelector.text())
        self.windowOffset = float(self.view.windowOffsetSelector.text())
        self.cmap = self.view.cmapSelector.currentText()
        self.wavelet = self.view.waveletSelector.currentText()
        self.n_freqs = int(self.view.freqsSelector.text())
        self.freq_min = None
        self.freq_max = None
        self.n_points = None
        # CWT computation mode: 'full' or 'per-window'
        self.cwtMode = 'full'  # Default to full signal
        # COI masking setting
        self.coiMasking = False  # Default to no masking
        # Global vmax calculation setting
        self.useGlobalVmax = False  # Default to hardcoded vmax
        self.calculatedVmax = None  # Store calculated value
        # Percentile-based vmax (set to None for absolute max, or 99.5 for 99.5th percentile)
        self.vmaxPercentile = 99.9  # Saturate top 0.1% for better contrast
        # Initialise sampling rate variable
        self.samplingRate = 0 # Hz
        # Define file locations
        self.folder = get_paths()['hdf5']
        self.outputFolder = Path(self.folder, 'CWT_labelled_windows')
        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder)
        self.logbook = get_logbook()
        # Enable CSV logging for storing window information and labels (manual mode only)
        self.csv_logging_enabled = False
        # Define xray resolution
        self.xrayRes = 4.3 # um/px
        self.autoSave = False # Sets to True if using auto-labelling
        # Cancellation flag for auto labelling
        self.cancel_auto_labelling = False
        # Tracks to exclude due to corrupted PD signal
        self.exclude = ['0514_02',
           '0514_04',
           '0514_05',
           '0515_01',
           '0515_02',
           '0515_03',
           '0515_04',
           '0515_05',
           '0515_06',
           '0516_01',
           '0516_02',
           '0516_03',
           '0516_04',
           '0516_05',
           '0516_06']
    
    def connect_signals(self):
        self.view.btnStart.clicked.connect(self.get_data)
        self.view.btnStart.clicked.connect(self.view.grey_out_load)
        self.view.btnAuto.clicked.connect(lambda: self.view.enable_controls(mode=False))
        self.view.btnAuto.clicked.connect(self.auto_label)
        self.view.btnCancelAuto.clicked.connect(self.cancel_auto_label)
        self.view.pdChannelSelector.textEdited.connect(lambda: self.update_pd_channel(self.view.pdChannelSelector.text()))
        self.view.waveletSelector.currentTextChanged.connect(lambda: self.update_wavelet(self.view.waveletSelector.currentText()))
        self.view.cmapSelector.currentTextChanged.connect(lambda: self.update_cmap(self.view.cmapSelector.currentText()))
        self.view.windowLengthSelector.textEdited.connect(lambda: self.update_window_length(self.view.windowLengthSelector.text()))
        self.view.windowOffsetSelector.textEdited.connect(lambda: self.update_window_offset(self.view.windowOffsetSelector.text()))
        self.view.freqsSelector.textEdited.connect(lambda: self.update_n_freqs(self.view.freqsSelector.text()))
        # CWT mode radio buttons
        self.view.cwtModeFullSignal.toggled.connect(lambda: self.update_cwt_mode('full') if self.view.cwtModeFullSignal.isChecked() else None)
        self.view.cwtModePerWindow.toggled.connect(lambda: self.update_cwt_mode('per-window') if self.view.cwtModePerWindow.isChecked() else None)
        # COI masking checkbox
        self.view.coiMaskingCheckbox.stateChanged.connect(lambda: self.update_coi_masking(self.view.coiMaskingCheckbox.isChecked()))
        # Global vmax checkbox
        self.view.globalVmaxCheckbox.stateChanged.connect(lambda: self.update_global_vmax_setting(self.view.globalVmaxCheckbox.isChecked()))
        self.updateReadout.connect(self.view.set_readout_text)
        self.view.buttons['Next'].clicked.connect(lambda: self.navigate(fileDirection='+'))
        self.view.buttons['Previous'].clicked.connect(lambda: self.navigate(fileDirection='-'))
        self.view.buttons['Label 0'].clicked.connect(lambda: self.save_cwt('0'))
        self.view.buttons['Label 0'].clicked.connect(lambda: self.navigate(windowDirection='+'))
        self.view.buttons['Label 1'].clicked.connect(lambda: self.save_cwt('1'))
        self.view.buttons['Label 1'].clicked.connect(lambda: self.navigate(windowDirection='+'))
        self.view.buttons['Skip'].clicked.connect(lambda: self.navigate(windowDirection='+'))
        self.view.buttons['Go back'].clicked.connect(lambda: self.navigate(windowDirection='-'))
        self.progress.connect(self.view.update_progress)
        
    def print_settings(self):
        print()
        print(f'PD channel = {self.pdChannelShort}')
        print(f'wavelet = {self.wavelet}')
        print(f'cmap = {self.cmap}')
        print(f'n_freqs = {self.n_freqs}')
        print(f'window length = {self.windowLength}')
        print(f'window offset = {self.windowOffset}')
        print(f'CWT mode = {self.cwtMode}')
        print(f'COI masking = {self.coiMasking}')
        print(f'Use global vmax = {self.useGlobalVmax}')

    def update_pd_channel(self, value):
        """Update PD channel when selector text changes."""
        try:
            channel_num = int(value)
            self.pdChannelLong = f'Photodiode{channel_num}Bits'
            self.pdChannelShort = f'PD{channel_num}'
            self.print_settings()
        except ValueError:
            pass  # Invalid input, keep current values

    def update_wavelet(self, name):
        self.wavelet = name
        self.print_settings()

    def update_cmap(self, name):
        self.cmap = name
        self.print_settings()

    def update_window_length(self, value):
        try:
            self.windowLength = float(value)
        except:
            pass
        self.print_settings()

    def update_window_offset(self, value):
        try:
            self.windowOffset = float(value)
        except:
            pass
        self.print_settings()

    def update_n_freqs(self, value):
        try:
            self.n_freqs = int(value)
        except:
            pass
        self.print_settings()

    def update_cwt_mode(self, mode):
        self.cwtMode = mode
        self.print_settings()

    def update_coi_masking(self, enabled):
        self.coiMasking = enabled
        self.print_settings()

    def update_global_vmax_setting(self, enabled):
        """Update global vmax calculation setting."""
        self.useGlobalVmax = enabled
        self.print_settings()
        
    def show_viewer_windows(self):
        # Show the video player window directly to the right of the main window
        main_geo = self.view.geometry()
        player_x = main_geo.x() + main_geo.width() + 10
        player_y = main_geo.y()
        self.video_player.move(player_x, player_y)
        self.video_player.show()
        
        # Show CWT window below the video player
        player_geo = self.video_player.geometry()
        cwt_x = player_x
        cwt_y = player_geo.y() + player_geo.height() + 10
        self.cwt_window.move(cwt_x, cwt_y)
        self.cwt_window.show()
        
    def load_video(self):
        video_data = self.data.iloc[self.fIndex]['video']
        
        framerate = get_logbook_data(self.logbook, self.trackid)['framerate']
        start = int((self.wIndex + 1) * self.windowOffset * framerate / 1000)
        end = int(start + self.windowLength * framerate / 1000)
        
        trimmed_video = video_data[start:end]
        
        # Send data to video player
        self.video_player.set_data(trimmed_video)
    
    def filter_logbook(self):
        log = self.logbook
        
        if True:
            # filters for welding or powder melting
            welding = log['Powder material'] == 'None'
            powder = np.invert(welding)

            # filters for CW or PWM laser mode
            cw = log['Point jump delay [us]'] == 0
            pwm = np.invert(cw)

            # filter for Layer 1 tracks only
            L1 = log['Layer'] == 1
            
            # filter for presence of KH pores
            pores = log['n_pores'] > 2
            
            # filter by layer thickness
            thin_layer = log['measured_layer_thickness [um]'] <= 100
            very_thin_layer = log['measured_layer_thickness [um]'] <= 35
            
            # filter by scan speed
            speed = log['Scan speed [mm/s]'] == 400
            
            # filter by beamtime
            ltp1 = log['Beamtime'] == 1
            ltp2 = log['Beamtime'] == 2
            ltp3 = log['Beamtime'] == 3
            
            # filter by substrate
            s0514 = log['Substrate No.'] == '514'
            s0515 = log['Substrate No.'] == '515'
            s0504 = log['Substrate No.'] == '504'
            
            # filter by material
            AlSi10Mg = log['Substrate material'] == 'AlSi10Mg'
            Al7A77 = log['Substrate material'] == 'Al7A77'
            Al = log['Substrate material'] == 'Al'
            Ti64 = log['Substrate material'] == 'Ti64'
            lit = np.logical_or(Ti64, Al7A77)
            
            # filter by regime
            not_flickering = log['Melting regime'] != 'keyhole flickering'
            not_cond = log['Melting regime'] != 'conduction'
            
        # Apply combination of above filters to select parameter subset to plot
        # log_red = log[np.logical_or(AlSi10Mg, lit) & L1 & cw & powder]
        log_red = log[AlSi10Mg & L1]
        # log_red = log[s0514]
        
        return log_red

    def create_log(self):
        """Create CSV log file with proper schema for flat directory + CSV labels."""
        logDf = pd.DataFrame({
            'image_filename' : [],      # Added for flat directory structure
            'trackid' : [],
            'window_n' : [],
            'window_start_ms' : [],
            'window_end_ms' : [],
            'label' : []})
        logDf.set_index('trackid')
        now = strftime('%y%m%d_%H-%M-%S')
        self.logPath = Path(self.outputFolder, f'{now}_{self.windowLength}ms-window_{self.windowOffset}ms-offset_labels.csv')
        logDf.to_csv(self.logPath, index=False, encoding='utf-8')
    
    def background_write(self):
        worker = Worker(self.write_to_log)
        self.threadpool.start(worker)
    
    def write_to_log(self):
        """Write label entry to CSV log with image_filename for flat directory structure."""
        if debug: print(f"🔧 DEBUG: write_to_log called - csv_logging_enabled={self.csv_logging_enabled}, autoSave={self.autoSave}")

        # Generate image filename matching the saved file format
        image_filename = f'{self.trackid}_{round(self.wStart, 1)}-{round(self.wEnd, 1)}ms.png'

        logRow = pd.DataFrame({
            'image_filename' : [image_filename],
            'trackid' : [self.trackid],
            'window_n' : [self.wIndex],
            'window_start_ms' : [self.wStart],
            'window_end_ms' : [self.wEnd],
            'label' : [self.label]})
        logRow.to_csv(self.logPath, mode='a', index=False, header=False, encoding='utf-8')
    
    def get_data(self):
        # Data loading - CSV log creation happens later when mode is determined
        if debug: print(f"🔧 DEBUG: get_data() called - just loading data, no CSV creation yet")
        # Create file read job on worker thread
        print('Initialising Worker to read data')
        worker = Worker(self.read_files)
        worker.signals.output.connect(self.keep_data)
        worker.signals.finished.connect(lambda: self.view.set_readout_text('Done'))
        worker.signals.finished.connect(self.view.progressBar.reset)
        worker.signals.finished.connect(self.view.enable_controls)
        worker.signals.finished.connect(self.show_viewer_windows)
        # Execute
        self.threadpool.start(worker)
    
    def read_files(self):
        trackids_filt = self.filter_logbook()['trackid'].to_list()
        all_files = sorted(glob.glob(f'{self.folder}/*.hdf5'))
        files = []
        for t in trackids_filt:
            for f in all_files:
                if t in f and t not in self.exclude: files.append(f)
        
        self.nFiles = len(files)
        print(f'Reading {self.nFiles} files from \'{self.folder}\'')
        group, time, series, colour = ('AMPM', 'Time', self.pdChannelLong, 'r')
        data = {'trackid': [], 't': [], 'PD': [], 'xray': [], 'video': []} 
        for i, filepath in enumerate(sorted(files)):
            trackid = Path(filepath).name[:7]
            data['trackid'].append(trackid)
            with h5py.File(filepath, 'r') as file:
                data['t'].append(np.array(file[f'{group}/{time}'])[500:-500])
                data['PD'].append(np.array(file[f'{group}/{series}'])[500:-500])
                data['xray'].append(np.array(file['bs-f40'])[-1])
                data['video'].append(np.array(file['bs-f40_lagrangian']))
            
            self.progress.emit(int(100*(i+1)/self.nFiles), f'Reading {Path(filepath).name}')
        
        df = pd.DataFrame(data)
        return df
    
    def navigate(self, fileDirection='=', windowDirection='='):
        try:
            if fileDirection == '+':
                if self.fIndex >= self.nFiles - 1:
                    raise IndexError
                else:
                    self.fIndex += 1
                    self.wIndex = 0
            elif fileDirection == '-':
                if self.fIndex <= 0:
                    raise IndexError
                else:
                    self.fIndex -= 1
                    self.wIndex = 0
            elif windowDirection == '+':
                self.wIndex += 1
            elif windowDirection == '-':
                self.wIndex -= 1
            # self.wStart =  self.windowOffset + self.wIndex * self.windowLength
            self.wStart = round(self.windowOffset * (self.wIndex + 1), 3)  # Round to 3 decimal places
            self.wEnd = round(self.wStart + self.windowLength, 3)  # Round to 3 decimal places
            data_row = self.data.iloc[self.fIndex]
            self.trackid = data_row['trackid']
            print(self.trackid, ' window ', self.wIndex)
            worker = Worker(self.cwt,
                            data = data_row,
                            wavelet = self.wavelet,
                            n_freqs = self.n_freqs
                            )
            worker.signals.output.connect(self.cwt_plot)
            self.threadpool.start(worker)
            self.xray_plot()
            self.load_video()
            self.view.update_progress(int(100*(self.fIndex+1)/self.nFiles), self.trackid)
        except IndexError:
            print('No more files')

    def cancel_auto_label(self):
        """Request cancellation of auto labelling."""
        self.cancel_auto_labelling = True
        self.view.set_readout_text('Cancelling...')
        self.view.btnCancelAuto.setEnabled(False)
        print('AUTO-LABELLING CANCELLED BY USER')

    def calculate_global_vmax(self, window_definitions):
        """
        Calculate global vmax by processing all CWTs.

        Returns the maximum CWT coefficient value across all windows
        in the dataset for consistent normalization. Can use either absolute
        maximum or percentile-based maximum for better contrast.

        Args:
            window_definitions: DataFrame with trackid, window_n, window_start_ms, window_end_ms

        Returns:
            float: Maximum CWT coefficient value (absolute or percentile-based)
        """
        print(f'\n{"="*60}')
        print(f'CALCULATING GLOBAL VMAX: Pass 1/2')
        print(f'Processing {len(window_definitions)} windows to find maximum...')
        if self.vmaxPercentile is not None:
            print(f'Using {self.vmaxPercentile}th percentile (saturates top {100-self.vmaxPercentile:.1f}%)')
        else:
            print(f'Using absolute maximum (no saturation)')
        print(f'{"="*60}\n')

        window_max_values = []  # Store all window max values for percentile calculation
        windows_by_track = window_definitions.groupby('trackid')
        processed = 0
        start_time = pd.Timestamp.now()

        for trackid, track_windows in windows_by_track:
            # Check for cancellation
            if self.cancel_auto_labelling:
                return None

            # Skip excluded tracks
            if trackid in self.exclude:
                processed += len(track_windows)
                continue

            # Get track data
            try:
                data_row = self.data.loc[trackid]
            except KeyError:
                print(f'{trackid} not found, skipping')
                processed += len(track_windows)
                continue

            # Compute CWT for this track (same logic as auto_label)
            if self.cwtMode == 'full':
                # Compute full CWT once
                cwt_spec = self.cwt(data=data_row, wavelet=self.wavelet, n_freqs=self.n_freqs)

                # Process all windows from cached CWT
                for row in track_windows.itertuples():
                    # Extract window
                    t = cwt_spec['t']
                    t_ms = t * 1000
                    window_start_idx = np.argmin(np.abs(t_ms - row.window_start_ms))
                    window_end_idx = np.argmin(np.abs(t_ms - row.window_end_ms))
                    cwt_windowed = cwt_spec['cwtmatr'][:, window_start_idx:window_end_idx]

                    # Track maximum for this window
                    window_max = cwt_windowed.max()
                    window_max_values.append(window_max)

                    processed += 1

                    # Progress update
                    if processed % 100 == 0:
                        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
                        rate = processed / elapsed if elapsed > 0 else 0
                        current_max = max(window_max_values)
                        pct = 100 * processed / len(window_definitions)
                        self.view.update_progress(
                            int(pct),
                            f'Pass 1: {trackid} | {processed}/{len(window_definitions)} ({pct:.1f}%) | Max={current_max:.1f}'
                        )
            else:
                # Per-window mode: compute each separately
                for row in track_windows.itertuples():
                    # Set window parameters for per-window CWT
                    self.wStart = round(row.window_start_ms, 3)
                    self.wEnd = round(row.window_end_ms, 3)

                    # Compute CWT for this window
                    cwt_spec = self.cwt(data=data_row, wavelet=self.wavelet, n_freqs=self.n_freqs)

                    # Track maximum for this window (CWT already windowed in per-window mode)
                    window_max = cwt_spec['cwtmatr'].max()
                    window_max_values.append(window_max)

                    processed += 1

                    # Progress update
                    if processed % 100 == 0:
                        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
                        rate = processed / elapsed if elapsed > 0 else 0
                        current_max = max(window_max_values)
                        pct = 100 * processed / len(window_definitions)
                        self.view.update_progress(
                            int(pct),
                            f'Pass 1: {trackid} | {processed}/{len(window_definitions)} ({pct:.1f}%) | Max={current_max:.1f}'
                        )

        # Calculate vmax from collected values
        if not window_max_values:
            print('ERROR: No valid windows processed')
            return None

        window_max_array = np.array(window_max_values)
        absolute_max = window_max_array.max()

        # Calculate percentiles for comprehensive report
        percentiles = [95.0, 96.0, 97.0, 98.0, 98.5, 99.0, 99.5, 99.9, 100.0]
        percentile_values = {p: np.percentile(window_max_array, p) if p < 100 else absolute_max
                            for p in percentiles}

        # Use percentile or absolute max
        if self.vmaxPercentile is not None:
            vmax = np.percentile(window_max_array, self.vmaxPercentile)
            n_saturated = np.sum(window_max_array > vmax)
            saturation_pct = 100 * n_saturated / len(window_max_array)
        else:
            vmax = absolute_max
            n_saturated = 0
            saturation_pct = 0.0

        # Final console report (concise)
        elapsed_total = (pd.Timestamp.now() - start_time).total_seconds()
        print(f'\n{"="*60}')
        print(f'GLOBAL VMAX CALCULATION COMPLETE')
        print(f'Processed: {processed} windows | Time: {elapsed_total:.1f}s')
        print(f'')
        if self.vmaxPercentile is not None:
            print(f'CALCULATED VMAX ({self.vmaxPercentile}th percentile): {vmax:.6f}')
            print(f'Saturated: {n_saturated}/{len(window_max_array)} ({saturation_pct:.2f}%)')
        else:
            print(f'CALCULATED VMAX (absolute max): {vmax:.6f}')
        print(f'{"="*60}\n')

        # Save comprehensive percentile report to file
        output_folder = self.get_output_folder()
        report_path = Path(output_folder, 'vmax_percentile_report.txt')

        with open(report_path, 'w') as f:
            f.write('='*60 + '\n')
            f.write('GLOBAL VMAX PERCENTILE REPORT\n')
            f.write('='*60 + '\n\n')
            f.write(f'Dataset: {len(window_max_array)} windows processed\n')
            f.write(f'Wavelet: {self.wavelet}\n')
            f.write(f'CWT Mode: {self.cwtMode}\n')
            f.write(f'COI Masking: {self.coiMasking}\n')
            f.write(f'Calculation time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)\n')
            f.write(f'Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write('\n' + '-'*60 + '\n')
            f.write('VMAX VALUES BY PERCENTILE\n')
            f.write('-'*60 + '\n\n')

            for p in percentiles:
                val = percentile_values[p]
                if p < 100:
                    n_sat = np.sum(window_max_array > val)
                    sat_pct = 100 * n_sat / len(window_max_array)
                    f.write(f'{p:5.1f}th percentile: {val:12.6f}  (saturates {sat_pct:5.2f}%)\n')
                else:
                    f.write(f'Absolute maximum: {val:12.6f}  (saturates  0.00%)\n')

            f.write('\n' + '-'*60 + '\n')
            f.write('USAGE NOTES\n')
            f.write('-'*60 + '\n\n')
            f.write('To use a specific percentile for future runs:\n')
            f.write('1. Choose a percentile from the table above\n')
            f.write('2. Update tools.py vmax_dict with the value, e.g.:\n')
            f.write(f"   '{self.wavelet}': {percentile_values[99.5]:.6f}\n")
            f.write('\n')
            f.write('Or adjust self.vmaxPercentile in dataset_labeller.py\n')
            f.write('to automatically use a different percentile.\n')
            f.write('\n')
            f.write(f'Current setting: {self.vmaxPercentile}th percentile\n' if self.vmaxPercentile else 'Current setting: absolute maximum\n')
            f.write(f'Current vmax used: {vmax:.6f}\n')

        print(f'Percentile report saved: {report_path}')
        print()

        return vmax

    def auto_label(self):
        """
        Auto-labelling mode: Process all windows from CSV without GUI interaction.

        OPTIMIZATIONS:
        - No sleep() calls - processes windows directly
        - Groups windows by trackid to cache CWT computation
        - Direct numpy→image saving without matplotlib overhead
        - Batch progress updates (every 100 windows)
        - Cancellable via cancel button
        """
        # Reset cancellation flag and enable cancel button
        self.cancel_auto_labelling = False
        self.view.btnCancelAuto.setEnabled(True)

        self.autoSave = True
        self.csv_logging_enabled = False  # Don't write to CSV - labels already in input CSV
        plt.close('all')
        mpl.use('agg')

        labelPath = self.view.autoLabelPath.text()
        window_definitions = pd.read_csv(labelPath)
        n_windows = len(window_definitions)

        print(f'\n{"="*60}')
        print(f'AUTO-LABELLING MODE: Processing {n_windows} windows')
        print(f'CWT Mode: {self.cwtMode}')
        print(f'COI Masking: {self.coiMasking}')
        print(f'{"="*60}\n')

        # Optimize dataframe for faster lookup
        if len(self.data.keys()) > 2:
            self.data = self.data.drop(columns=['xray', 'video'])
            self.data = self.data.set_index('trackid')

        # Group windows by trackid for efficient processing
        windows_by_track = window_definitions.groupby('trackid')

        # PASS 1: Calculate global vmax if enabled
        if self.useGlobalVmax:
            self.calculatedVmax = self.calculate_global_vmax(window_definitions)

            if self.calculatedVmax is None:
                # Cancelled during vmax calculation
                self.view.set_readout_text('Cancelled during vmax calculation')
                self.view.progressBar.reset()
                self.view.btnCancelAuto.setEnabled(False)
                self.view.enable_controls()
                return

            print(f'Using calculated vmax: {self.calculatedVmax:.6f}')
            print(f'\nStarting Pass 2/2: Saving images...\n')
        else:
            print(f'Using hardcoded vmax from tools.py')

        # PASS 2: Now process and save images
        completed = 0
        start_time = pd.Timestamp.now()

        for trackid, track_windows in windows_by_track:
            # Check for cancellation request
            if self.cancel_auto_labelling:
                break

            if trackid in self.exclude:
                completed += len(track_windows)
                continue

            try:
                data_row = self.data.loc[trackid]
            except KeyError:
                print(f'{trackid} not found, check filter_logbook() parameters')
                completed += len(track_windows)
                continue

            # OPTIMIZATION: For 'full' mode, compute CWT once per track
            if self.cwtMode == 'full':
                # Compute full CWT once
                cwt_full = self.cwt(data=data_row, wavelet=self.wavelet, n_freqs=self.n_freqs)

                # Process all windows from this cached CWT
                for row in track_windows.itertuples():
                    # Check for cancellation request
                    if self.cancel_auto_labelling:
                        break

                    self.trackid = trackid
                    self.wIndex = row.window_n
                    self.wStart = round(row.window_start_ms, 3)
                    self.wEnd = round(row.window_end_ms, 3)
                    self.label = str(row.has_porosity)

                    # Save directly using cached CWT (no worker thread needed)
                    # Use calculated vmax if available, else use hardcoded from cwt_spec
                    vmax_to_use = self.calculatedVmax if self.useGlobalVmax else None
                    self._save_cwt_from_cached(cwt_full, override_vmax=vmax_to_use)
                    completed += 1

                    # Update progress every 100 windows
                    if completed % 100 == 0:
                        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
                        rate = completed / elapsed if elapsed > 0 else 0
                        pct = 100 * completed / n_windows
                        self.view.update_progress(
                            int(pct),
                            f'{trackid} win {row.window_n} | {completed}/{n_windows} ({pct:.1f}%) | {rate:.1f} win/s'
                        )
            else:
                # Per-window mode: must compute each separately (smart padding)
                for row in track_windows.itertuples():
                    # Check for cancellation request
                    if self.cancel_auto_labelling:
                        break

                    self.trackid = trackid
                    self.wIndex = row.window_n
                    self.wStart = round(row.window_start_ms, 3)
                    self.wEnd = round(row.window_end_ms, 3)
                    self.label = str(row.has_porosity)

                    # Compute CWT for this window
                    cwt_result = self.cwt(data=data_row, wavelet=self.wavelet, n_freqs=self.n_freqs)

                    # Save directly (no worker thread)
                    # Use calculated vmax if available, else use hardcoded from cwt_spec
                    vmax_to_use = self.calculatedVmax if self.useGlobalVmax else None
                    self._save_cwt_from_cached(cwt_result, override_vmax=vmax_to_use)
                    completed += 1

                    # Update progress every 100 windows
                    if completed % 100 == 0:
                        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
                        rate = completed / elapsed if elapsed > 0 else 0
                        pct = 100 * completed / n_windows
                        self.view.update_progress(
                            int(pct),
                            f'{trackid} win {row.window_n} | {completed}/{n_windows} ({pct:.1f}%) | {rate:.1f} win/s'
                        )

        # Final update
        elapsed_total = (pd.Timestamp.now() - start_time).total_seconds()
        final_rate = completed / elapsed_total if elapsed_total > 0 else 0

        # Check if cancelled or completed
        if self.cancel_auto_labelling:
            print(f'\n{"="*60}')
            print(f'AUTO-LABELLING CANCELLED')
            print(f'{completed}/{n_windows} windows ({100*completed/n_windows:.1f}%) | {elapsed_total:.1f}s | {final_rate:.2f} win/s')
            print(f'{"="*60}\n')
            self.view.set_readout_text(f'Cancelled - {completed}/{n_windows} windows')
        else:
            print(f'\n{"="*60}')
            print(f'AUTO-LABELLING COMPLETE')
            print(f'{completed} windows | {elapsed_total:.1f}s ({elapsed_total/60:.1f} min) | {final_rate:.2f} win/s')
            print(f'{"="*60}\n')
            self.view.set_readout_text(f'Done - {completed} windows in {elapsed_total:.1f}s')

        self.view.progressBar.reset()
        self.view.btnCancelAuto.setEnabled(False)
        self.view.enable_controls()
    
    def get_output_folder(self):
        """
        Get output directory for CWT images (flat directory structure).

        Directory structure:
        outputFolder/PD1/wavelet/window_ms/freq_range/cmap/[full_signal|per_window]/

        Images are organized by computation mode to separate:
        - full_signal: CWT computed on full signal, then window extracted
        - per_window: CWT computed only on windowed region with smart padding

        All images are saved here regardless of their classification label.
        Labels are tracked separately in CSV file.

        Returns:
            Path: Output directory path
        """
        # Determine computation mode subdirectory name
        mode_subdir = 'full_signal' if self.cwtMode == 'full' else 'per_window'

        output_folder = Path(self.outputFolder,
                            self.pdChannelShort,
                            self.wavelet.replace('.', '_'),
                            f'{self.windowLength}_ms',
                            f'{self.freq_min}-{self.freq_max}_Hz_{self.n_freqs}_steps',
                            self.cmap,
                            mode_subdir)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        return output_folder

    def cwt(self, data, wavelet='cmor1.5-1.0', n_freqs=256, printCWTSpec=False):
        """
        Compute CWT with selectable computation mode.

        Modes:
        - 'full': Compute CWT on full signal with symmetric padding, then extract window (default)
        - 'per-window': Extract window first, pad with actual signal where available

        vmax is hardcoded per-wavelet to preserve amplitude comparison across dataset for ML training.
        This ensures consistent scaling so the neural network can learn amplitude differences.
        """
        if debug: print(f'cwt() called with mode: {self.cwtMode}')

        # Get basic signal parameters
        samplingPeriod = round(data['t'][1]-data['t'][0], 9)
        samplingRate = round(1/samplingPeriod, 7)
        s = data['PD']
        t = data['t']
        self.n_points = len(t)
        self.samplingRate = int(round(1/(t[1]-t[0])))

        if debug:
            print(f"CWT: {len(s)} samples, {self.samplingRate} Hz")

        # Get scales and vmax
        scales, vmax = get_cwt_scales(wavelet, n_freqs, sampling_rate=self.samplingRate)

        # Branch based on computation mode
        if self.cwtMode == 'per-window':
            # Per-window mode: compute only on windowed region with smart padding
            cwtmatr, freqs = self._cwt_per_window(s, t, scales, wavelet, samplingPeriod)
        else:
            # Full signal mode (default): compute on full signal then extract window
            cwtmatr, freqs = self._cwt_full_signal(s, scales, wavelet, samplingPeriod)

        n_samples_window = int(self.windowLength * self.samplingRate / 1000)

        if printCWTSpec:
            print(f'Wavelet: {wavelet}')
            print(f'Frequency range: {round(freqs[-1], 0)}-{round(freqs[0], 0)} Hz')
            print(f'Period range: {round(1000/freqs[0],2)}-{round(1000/freqs[-1],2)} ms')

        # For use in folder naming
        self.freq_min = int(round(freqs[-1]))
        self.freq_max = int(round(freqs[0]))

        # For use by cwt_plot
        return {'t': t, 'freqs': freqs, 'cwtmatr': cwtmatr, 'vmax': vmax, 'n_samples_window': n_samples_window}

    def _cwt_full_signal(self, s, scales, wavelet, samplingPeriod):
        """
        Compute CWT on full signal with symmetric padding (improved from manual reversal).
        Returns: (cwtmatr, freqs) with cwtmatr cropped to original signal length.
        """
        # Use symmetric padding for better edge continuity (C1 continuous)
        s_pad = np.pad(s, len(s), mode='symmetric')

        if debug:
            print(f"Full signal mode: Padded signal length: {len(s_pad)} (3x original)")

        # Perform CWT on padded signal
        cwtmatr, freqs = pywt.cwt(s_pad, scales, wavelet, sampling_period=samplingPeriod)

        # Crop out the padding (extract middle section = original signal)
        cwtmatr = np.abs(cwtmatr[:, self.n_points:2*self.n_points])

        return cwtmatr, freqs

    def _cwt_per_window(self, s, t, scales, wavelet, samplingPeriod):
        """
        Compute CWT on windowed region with smart padding strategy.
        Uses actual signal as padding where available, synthetic only at edges.
        Returns: (cwtmatr, freqs) already extracted to window region.
        """
        t_ms = t * 1000  # Convert to milliseconds

        # Find window indices
        window_start_idx = np.argmin(np.abs(t_ms - self.wStart))
        window_end_idx = np.argmin(np.abs(t_ms - self.wEnd))
        window_length = window_end_idx - window_start_idx

        if debug:
            print(f"Per-window mode: Window [{window_start_idx}:{window_end_idx}] = {window_length} samples")

        # Determine padding length (same as full signal mode for consistency)
        pad_length = window_length

        # LEFT PADDING: Use actual signal where available
        if window_start_idx >= pad_length:
            # Sufficient signal before window - use it directly
            left_pad = s[window_start_idx - pad_length : window_start_idx]
            if debug: print(f"Left padding: {pad_length} samples from actual signal")
        else:
            # Near signal start - need some synthetic padding
            available_left = window_start_idx
            left_signal = s[0:window_start_idx]
            synthetic_needed = pad_length - available_left
            synthetic_left = np.pad(left_signal, (synthetic_needed, 0), mode='symmetric')[:synthetic_needed]
            left_pad = np.concatenate([synthetic_left, left_signal])
            if debug: print(f"Left padding: {synthetic_needed} synthetic + {available_left} actual")

        # WINDOW (actual data to analyze)
        window = s[window_start_idx:window_end_idx]

        # RIGHT PADDING: Use actual signal where available
        if window_end_idx + pad_length <= len(s):
            # Sufficient signal after window - use it directly
            right_pad = s[window_end_idx : window_end_idx + pad_length]
            if debug: print(f"Right padding: {pad_length} samples from actual signal")
        else:
            # Near signal end - need some synthetic padding
            available_right = len(s) - window_end_idx
            right_signal = s[window_end_idx : len(s)]
            synthetic_needed = pad_length - available_right
            synthetic_right = np.pad(right_signal, (0, synthetic_needed), mode='symmetric')[-synthetic_needed:]
            right_pad = np.concatenate([right_signal, synthetic_right])
            if debug: print(f"Right padding: {available_right} actual + {synthetic_needed} synthetic")

        # Combine: [left_pad][window][right_pad]
        s_pad = np.concatenate([left_pad, window, right_pad])

        if debug:
            print(f"Per-window padded signal length: {len(s_pad)} (window={window_length}, total_pad={2*pad_length})")

        # Perform CWT on padded window
        cwtmatr, freqs = pywt.cwt(s_pad, scales, wavelet, sampling_period=samplingPeriod)

        # Extract only the window portion (middle section)
        cwtmatr = np.abs(cwtmatr[:, pad_length : pad_length + window_length])

        # Store window indices for use by _cwt_plot_internal
        self._per_window_start_idx = window_start_idx
        self._per_window_end_idx = window_end_idx

        return cwtmatr, freqs

    def _apply_coi_masking(self, cwtmatr, freqs):
        """
        Apply Cone of Influence (COI) masking to CWT coefficients.
        Sets edge artifact regions to 0 based on wavelet support width.

        The COI represents the region where edge effects from padding are significant.
        For CWT, larger scales (lower frequencies) require more samples and have wider COI.

        Args:
            cwtmatr: CWT coefficient matrix, shape (n_scales, n_samples)
            freqs: Frequency array corresponding to scales

        Returns:
            Masked CWT matrix with COI regions set to 0
        """
        n_scales, n_samples = cwtmatr.shape
        cwtmatr_masked = cwtmatr.copy()

        # COI boundary calculation
        # For most wavelets, the e-folding time (support width) is approximately sqrt(2)*scale
        # At the edges, coefficients within e-folding distance are affected by boundary
        for i, freq in enumerate(freqs):
            # Convert frequency to scale (inverse relationship)
            scale = self.samplingRate / freq

            # e-folding distance in samples (approximate wavelet support width)
            # This is wavelet-dependent; sqrt(2)*scale is a reasonable approximation
            coi_width = int(np.ceil(np.sqrt(2) * scale))

            # Mask left edge
            cwtmatr_masked[i, :coi_width] = 0

            # Mask right edge
            cwtmatr_masked[i, -coi_width:] = 0

        if debug:
            n_masked = np.sum(cwtmatr_masked == 0) - np.sum(cwtmatr == 0)
            print(f"COI masking: Set {n_masked} coefficients to 0 (edge artifacts)")

        return cwtmatr_masked

    def cwt_plot(self, cwt_spec):
        """Process CWT result from worker thread - display or save based on mode."""
        try:
            self._cwt_plot_internal(cwt_spec)
        except Exception as e:
            print(f"Error in cwt_plot: {e}")
            traceback.print_exc()

    def _cwt_plot_internal(self, cwt_spec):
        """
        Process CWT scalogram and either display (manual mode) or save (auto mode).

        Uses instance variables set before worker execution:
        - self.trackid, self.wStart, self.wEnd: Window identification
        - self.label: Classification label (0 or 1)
        - self.autoSave: Mode flag (True=auto, False=manual)
        - self.cwtMode: 'full' or 'per-window' computation mode
        """
        # Define CWT figure attributes
        dpi = 100
        rect = [0, 0, 1, 1]  # Full figure for pixel-perfect output

        t = cwt_spec['t']
        t_ms = t * 1000  # Convert to milliseconds

        # Handle window extraction based on CWT mode
        if self.cwtMode == 'per-window':
            # Per-window mode: CWT is already windowed, just use stored indices
            window_start_idx = self._per_window_start_idx
            window_end_idx = self._per_window_end_idx
            n_samples = window_end_idx - window_start_idx

            if debug:
                print(f"=== PER-WINDOW MODE ===")
                print(f"Using pre-computed window [{window_start_idx}:{window_end_idx}] = {n_samples} samples")

            # CWT is already windowed - use directly
            t_windowed = t[window_start_idx:window_end_idx]
            cwt_windowed = cwt_spec['cwtmatr']  # Already extracted in _cwt_per_window
        else:
            # Full signal mode: Extract window from full CWT
            window_start_idx = np.argmin(np.abs(t_ms - self.wStart))
            window_end_idx = np.argmin(np.abs(t_ms - self.wEnd))
            n_samples = window_end_idx - window_start_idx

            # Check that window is full length, skip to next file if end of signal reached
            if n_samples < cwt_spec['n_samples_window']:
                self.navigate(fileDirection = '+')
                return

            if debug:
                print(f"=== FULL SIGNAL MODE - WINDOW EXTRACTION ===")
                print(f"Window start: {self.wStart} ms, idx: {window_start_idx}")
                print(f"Window end: {self.wEnd} ms, idx: {window_end_idx}")
                print(f"Window width in samples: {n_samples}")

            # Extract the windowed portion from full CWT
            t_windowed = t[window_start_idx:window_end_idx]
            cwt_windowed = cwt_spec['cwtmatr'][:, window_start_idx:window_end_idx]

        if debug:
            print(f"Windowed time array shape: {t_windowed.shape}")
            print(f"Windowed CWT matrix shape: {cwt_windowed.shape}")

        # Apply COI masking if enabled
        if self.coiMasking:
            cwt_windowed = self._apply_coi_masking(cwt_windowed, cwt_spec['freqs'])

        tAx, fAx = np.meshgrid(t_windowed*1000, cwt_spec['freqs']/1000) # convert to ms and kHz
        
        # Store windowed data for saving
        self.current_windowed_data = {
            't': t_windowed,
            'cwt': cwt_windowed,
            'freqs': cwt_spec['freqs'],
            'vmax': cwt_spec['vmax'],
            'tAx': tAx,
            'fAx': fAx,
        }
        
        # Define figure and axes depending on labelling mode
        if self.autoSave:
            if debug: print('auto labelling triggered')
            self.save_cwt(self.label)
            
        else:
            if debug: print('manual labelling triggered')

            # Enable CSV logging and create log file for manual mode (first time only)
            if not self.csv_logging_enabled:
                self.csv_logging_enabled = True
                if debug: print("🔧 DEBUG: First manual label - enabling CSV logging")
                self.create_log()
            
            self.cwt_window.cwtFig.clear()
            self.cwt_window.cwtFig.set_dpi(dpi)
            
            if self.view.show_axes:
                ax = self.cwt_window.cwtFig.add_axes([0.2, 0.1, 0.75, 0.85])
            else:
                ax = self.cwt_window.cwtFig.add_axes(rect)
                ax.set_axis_off()
        
            # Define figure borders depending on axis display mode
            if self.view.show_axes == True:
                ax.set_xlabel('Time [ms]')
                ax.set_ylabel('Freq. [kHz]')
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
            else:
                ax.set_axis_off()
        
            # Draw CWT scalogram on axes
            im = ax.pcolormesh(tAx, fAx,
                cwt_windowed,
                cmap=self.cmap,
                vmin=0,
                vmax=cwt_spec['vmax']
                )
            ax.set_yscale('log', base=2)
                    
            self.cwt_window.cwtCanvas.draw()
            self.cwt_window.resize_to_fill_window()
            self.cwt_window.cwtCanvas.flush_events()
            
            # Update window info
            self.cwt_window.update_info(self.trackid, self.wStart, self.wEnd)
            
            plt.close()
                 
    def _save_cwt_from_cached(self, cwt_spec, override_vmax=None):
        """
        OPTIMIZED: Save CWT image directly from cached cwt_spec without matplotlib overhead.

        This bypasses the cwt_plot → save_cwt pipeline used in manual mode.
        Direct numpy → PIL conversion is ~5-10x faster than matplotlib rendering.

        Args:
            cwt_spec: CWT result dict from cwt() function
            override_vmax: Optional override for vmax (used for global normalization)
        """
        from PIL import Image
        import matplotlib.cm as cm

        # Extract window from CWT based on mode
        t = cwt_spec['t']
        t_ms = t * 1000

        if self.cwtMode == 'per-window':
            # CWT already windowed
            cwt_windowed = cwt_spec['cwtmatr']
        else:
            # Full signal mode: extract window
            window_start_idx = np.argmin(np.abs(t_ms - self.wStart))
            window_end_idx = np.argmin(np.abs(t_ms - self.wEnd))
            cwt_windowed = cwt_spec['cwtmatr'][:, window_start_idx:window_end_idx]

        # Apply COI masking if enabled
        if self.coiMasking:
            cwt_windowed = self._apply_coi_masking(cwt_windowed, cwt_spec['freqs'])

        # Normalize to 0-1 range using vmax
        # Use override_vmax if provided (global calculation), else use per-wavelet hardcoded
        vmax = override_vmax if override_vmax is not None else cwt_spec['vmax']
        cwt_normalized = np.clip(cwt_windowed / vmax, 0, 1)

        # Apply colormap
        cmap_func = mpl.colormaps.get_cmap(self.cmap)
        cwt_colored = cmap_func(cwt_normalized)  # Returns RGBA (0-1)

        # Convert to RGB uint8 and flip vertically (matplotlib convention)
        cwt_rgb = (cwt_colored[:, :, :3] * 255).astype(np.uint8)

        # Save using PIL (much faster than matplotlib)
        output_folder = self.get_output_folder()
        output_path = Path(output_folder, f'{self.trackid}_{round(self.wStart, 1)}-{round(self.wEnd, 1)}ms.png')

        img = Image.fromarray(cwt_rgb, mode='RGB')
        img.save(output_path, optimize=True)

    def save_cwt(self, label):
        """
        Original save_cwt for manual mode - uses matplotlib for GUI consistency.
        Auto mode now uses _save_cwt_from_cached() for speed.
        """
        self.label = label
        output_folder = self.get_output_folder()

        outputFPath = Path(output_folder, f'{self.trackid}_{round(self.wStart, 1)}-{round(self.wEnd, 1)}ms.png')
        if debug: print(f"🔧 DEBUG: Attempting to save image to: {outputFPath}")

        # Only write to CSV when logging is enabled (manual mode)
        if debug: print(f"🔧 DEBUG: save_cwt called, autoSave={self.autoSave}, csv_logging_enabled={self.csv_logging_enabled}")
        if self.csv_logging_enabled:
            self.background_write()

        # Use the stored windowed data to create a clean save
        # Create a temporary figure for saving with exact dimensions
        windowed_data = self.current_windowed_data

        actual_time_points = windowed_data['cwt'].shape[1]
        actual_freq_points = windowed_data['cwt'].shape[0]

        temp_fig = plt.figure(frameon=False, dpi=100,
                            figsize=(actual_time_points/100, actual_freq_points/100))
        temp_ax = temp_fig.add_axes([0, 0, 1, 1])
        temp_ax.set_axis_off()

        if debug:
            print('=====================')
            print('Saved figure spec:')
            print('tAx: ', windowed_data['tAx'].shape)
            print('fAx: ', windowed_data['fAx'].shape)
            print('cwt_windowed_data: ', windowed_data['cwt'].shape)

        # Plot the windowed data
        temp_ax.pcolormesh(windowed_data['tAx'], windowed_data['fAx'], windowed_data['cwt'],
                          cmap=self.cmap, vmin=0, vmax=windowed_data['vmax'])
        temp_ax.set_yscale('log', base=2)

        temp_fig.savefig(outputFPath, dpi=100)
        plt.close(temp_fig)
        if debug:
            print(f"🔧 DEBUG: Image saved successfully to: {outputFPath}")
            # Verify file exists
            if outputFPath.exists():
                print(f"🔧 DEBUG: File exists, size: {outputFPath.stat().st_size} bytes")
            else:
                print(f"🔧 DEBUG: ERROR - File was not created!")

        if debug:
            # DEBUG: Check saved image dimensions
            try:
                import PIL.Image
                with PIL.Image.open(outputFPath) as img:
                    actual_size = img.size
                print(f"SAVED IMAGE DIMENSIONS: {actual_size}")  # (width, height)
            except ImportError:
                print("PIL not available for image size verification")
            print()

    def xray_plot(self):
        self.view.xrayFig.clear()
        image = self.data.iloc[self.fIndex]['xray']
        scanSpeed = get_logbook_data(self.logbook, self.trackid)['scan_speed']
        windowLengthPx = self.ms_to_px(self.windowLength, scanSpeed)
        ax = plt.Axes(self.view.xrayFig, [0, 0, 1, 1])
        ax.imshow(image, cmap='gray')
        laserStartOffset = 50 # px
        x1 = self.ms_to_px(self.wStart, scanSpeed) + laserStartOffset
        x2 = self.ms_to_px(self.wEnd, scanSpeed) + laserStartOffset
        y1 = 0
        y2 = 511
        ax.plot([x1, x1], [y1, y2], 'k--')
        ax.plot([x2, x2], [y1, y2], 'k--')
        ax.set_axis_off()
        ax = self.view.xrayFig.add_axes(ax)
        self.view.xrayCanvas.draw()
    
    def ms_to_px(self, t, v):
        l = v * t / self.xrayRes
        return int(l)

    def keep_data(self, data):
        self.data = data
        
    
class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        """Execute the worker function in a background thread."""
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.output.emit(result)
        finally:
            self.signals.finished.emit()  


class WorkerSignals(QObject):
    # Defines the signals available from a running worker thread.
    finished = pyqtSignal()
    output = pyqtSignal(object)
    error = pyqtSignal(tuple)


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True) # Enable highdpi scaling
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True) # Use highdpi icons
    Gui = QApplication(sys.argv)
    # Define Qt stylesheet for GUI
    Gui.setStyleSheet("""QLabel         {font-size: 10pt;}
                         QPushButton    {font-size: 10pt;}
                         QLineEdit      {font-size: 10pt;}
                         QRadioButton   {font-size: 10pt;}
                         QComboBox      {font-size: 10pt;}
                         QTabWidget     {font-size: 10pt;}
                         QGroupBox      {font: bold 10pt; color: gray}
                         QToolButton    {font-size: 10pt;}
                         """)
    view = Window() # Define and then show GUI window
    view.show()
    videoPlayer = VideoPlayerWindow()
    cwtWindow = CWTWindow()
    Controller(view, videoPlayer, cwtWindow) # Initialise controller with access to the view
    sys.exit(Gui.exec())
    
if __name__ == '__main__':
    main()
