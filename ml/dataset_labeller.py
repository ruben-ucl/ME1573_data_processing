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

debug = True

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
        
        # CWT settings
        wlLabel = QLabel('Window length:')
        msUnit1 = QLabel('ms')
        woLabel = QLabel('Window offset:')
        msUnit2 = QLabel('ms')
        nfLabel = QLabel('Frequency steps:')
        wsLabel = QLabel('Wavelet:')
        cmLabel = QLabel('Colourmap:')
        
        self.windowLengthSelector = QLineEdit('1')
        self.windowOffsetSelector = QLineEdit('1')
        self.freqsSelector = QLineEdit('256')
        
        self.waveletSelector = QComboBox()
        wavelet_names = get_cwt_scales(None)
        for n in sorted(wavelet_names):
            self.waveletSelector.addItem(n)
        
        self.cmapSelector = QComboBox()
        cmaps = ['grey', 'grey_r', 'jet', 'magma', 'plasma', 'viridis']
        for c in cmaps:
            self.cmapSelector.addItem(c)
            
        inputsRow = QGroupBox('Setup')
        inputsLayout = QHBoxLayout()
        inputsLayout.addWidget(self.btnStart, stretch=4)
        inputsLayout.addStretch(4)
        inputsLayout.addWidget(wlLabel, stretch=4)
        inputsLayout.addWidget(self.windowLengthSelector, stretch=2)
        inputsLayout.addWidget(msUnit1, stretch=8)
        inputsLayout.addWidget(woLabel, stretch=4)
        inputsLayout.addWidget(self.windowOffsetSelector, stretch=2)
        inputsLayout.addWidget(msUnit2, stretch=8)
        inputsLayout.addWidget(nfLabel, stretch=4)
        inputsLayout.addWidget(self.freqsSelector, stretch=4)
        inputsLayout.addWidget(wsLabel, stretch=4)
        inputsLayout.addWidget(self.waveletSelector, stretch=4)
        inputsLayout.addWidget(cmLabel, stretch=4)
        inputsLayout.addWidget(self.cmapSelector, stretch=4)
        inputsRow.setLayout(inputsLayout)
        
        # Auto labelling inputs
        self.autoLabelPath = QLineEdit(r'E:\AlSi10Mg single layer ffc\CWT_labelled_windows\250620_10-01-59_1.0ms-window_0.2ms-offset_labels.csv')
        self.btnAuto = QPushButton('Run auto')
        self.btnAuto.setEnabled(False)
        
        autoRow = QGroupBox('Auto labelling')
        autoLayout = QHBoxLayout()
        autoLayout.addWidget(self.autoLabelPath, stretch=6)
        autoLayout.addWidget(self.btnAuto, stretch=1)
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
        
        # Combine GUI elements into main wndow layout
        layout = QVBoxLayout()
        # layout.addWidget(self.btnStart, stretch=1)
        layout.addWidget(inputsRow, stretch=1)
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
        # CWT settings
        self.windowLength = float(self.view.windowLengthSelector.text())
        self.windowOffset = float(self.view.windowOffsetSelector.text())
        self.cmap = self.view.cmapSelector.currentText()
        self.wavelet = self.view.waveletSelector.currentText()
        self.n_freqs = int(self.view.freqsSelector.text())
        self.freq_min = None
        self.freq_max = None
        self.n_points = None
        # Initialise sampling rate variable
        self.samplingRate = 0 # Hz
        # Define file locations
        self.folder = get_paths()['hdf5']
        self.outputFolder = Path(self.folder, 'CWT_labelled_windows')
        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder)
        self.logbook = get_logbook()
        # Create log file for storing window information and labels (only in manual mode)
        self.logging = False
        # Define xray resolution
        self.xrayRes = 4.3 # um/px
        self.autoSave = False # Sets to True if using auto-labelling
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
        self.view.waveletSelector.currentTextChanged.connect(lambda: self.update_wavelet(self.view.waveletSelector.currentText()))
        self.view.cmapSelector.currentTextChanged.connect(lambda: self.update_cmap(self.view.cmapSelector.currentText()))
        self.view.windowLengthSelector.textEdited.connect(lambda: self.update_window_length(self.view.windowLengthSelector.text()))
        self.view.windowOffsetSelector.textEdited.connect(lambda: self.update_window_offset(self.view.windowOffsetSelector.text()))
        self.view.freqsSelector.textEdited.connect(lambda: self.update_n_freqs(self.view.freqsSelector.text()))       
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
        print(f'wavelet = {self.wavelet}')
        print(f'cmap = {self.cmap}')
        print(f'n_freqs = {self.n_freqs}')
        print(f'window length = {self.windowLength}')
        print(f'window offset = {self.windowOffset}')
   
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
        log_red = log[AlSi10Mg & cw & L1 & powder]
        # log_red = log[s0514]
        
        return log_red

    def create_log(self):
        logDf = pd.DataFrame({'trackid' : [],
            'window_n' : [],
            'window_start_ms' : [],
            'window_end_ms' : [],
            'label' : []})
        logDf.set_index('trackid')
        now = strftime('%y%m%d_%H-%M-%S')
        self.logPath = Path(self.outputFolder, f'{now}_{self.windowLength}ms-window_{self.windowOffset}ms-offset_labels.csv')
        logDf.to_csv(self.logPath, index=False)
    
    def background_write(self):
        worker = Worker(self.write_to_log)
        self.threadpool.start(worker)
    
    def write_to_log(self):
        if debug: print(f"🔧 DEBUG: write_to_log called - logging={self.logging}, autoSave={self.autoSave}")
        logRow = pd.DataFrame({'trackid' : [self.trackid],
            'window_n' : [self.wIndex],
            'window_start_ms' : [self.wStart],
            'window_end_ms' : [self.wEnd],
            'label' : [self.label]})
        logRow.to_csv(self.logPath, mode='a', index=False, header=False)
    
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
        group, time, series, colour = ('AMPM', 'Time', 'Photodiode2Bits', 'r')
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
    
    def auto_label(self):
        self.autoSave = True
        self.logging = False  # Disable logging in auto mode since we already have a labels file
        plt.close('all')
        mpl.use('agg')
        labelPath = self.view.autoLabelPath.text()
        window_definitions = pd.read_csv(labelPath)
        n_windows = len(window_definitions)
        
        # Optimise dataframe for faster lookup of PD data
        if len(self.data.keys()) > 2:
            self.data.drop(columns=['xray', 'video'], inplace=True)
            self.data.set_index('trackid', inplace=True)
            
        for i, row in enumerate(window_definitions.itertuples()):
            self.view.update_progress(int(100*(i+1)/n_windows), '')
            self.trackid = row.trackid
            if self.trackid in self.exclude:
                continue
            self.wIndex = row.window_n
            self.wStart = round(row.window_start_ms, 3)  # Round to 3 decimal places for consistency
            self.wEnd = round(row.window_end_ms, 3)  # Round to 3 decimal places for consistency
            self.label = str(row.label)
            
            print(f'\n{self.trackid} {self.wStart}-{self.wEnd} ms\n', '='*50)
            try:
                data_row = self.data.loc[self.trackid]
            except KeyError:
                print(self.trackid, ' not found, check filter_logbook() parameters')
                continue            
            
            # Prepare windowed_data placeholder (will be generated in cwt method)
            windowed_data_placeholder = None  # This will be set in _cwt_plot_internal
            
            worker = Worker(self.cwt,
                            data = data_row,
                            wavelet = self.wavelet,
                            n_freqs = self.n_freqs,
                            # Pass metadata for thread-safe processing (use rounded values)
                            trackid = row.trackid,
                            wStart = self.wStart,  # Use rounded values
                            wEnd = self.wEnd,      # Use rounded values
                            label = str(row.label),
                            windowed_data = windowed_data_placeholder
                            )
            
            # CRITICAL: Connect signals IMMEDIATELY after worker creation, before any other operations  
            connection_result = worker.signals.output.connect(self.cwt_plot_thread_safe, Qt.QueuedConnection)  # Force queued connection
            
            # Test: Also connect to a simple test slot to see if ANY signal comes through
            if debug:
                def test_slot(data):
                    print(f"🔧 DEBUG: *** TEST SLOT RECEIVED DATA *** {type(data)}")
                worker.signals.output.connect(test_slot, Qt.QueuedConnection)
                print(f"🔧 DEBUG: Connecting signal type: {type(worker.signals)}")
                print(f"🔧 DEBUG: Signal output type: {type(worker.signals.output)}")
                print(f"🔧 DEBUG: Signal connection result: {connection_result}")
            
            # WORKAROUND: Force immediate processing to avoid threading issues
            if debug: print("🔧 DEBUG: Starting worker with signals connected...")
            
            # Now start the worker AFTER signals are connected
            self.threadpool.start(worker)
            
            # Give worker time to complete and process Qt events before moving to next one
            sleep(0.1)
            QApplication.processEvents()  # Process queued signals
            sleep(0.4)
            QApplication.processEvents()  # Process any remaining signals
            
        self.view.set_readout_text('Done')    
        self.view.progressBar.reset()
        self.view.enable_controls()
            
        pass
    
    def get_label_folder(self):
        label_folder = Path(self.outputFolder,
                            self.wavelet.replace('.', '_'),
                            f'{self.windowLength}_ms',
                            f'{self.freq_min}-{self.freq_max}_Hz_{self.n_freqs}_steps',
                            self.cmap,
                            self.label)
        
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        
        return label_folder

    def cwt(self, data, wavelet='cmor1.5-1.0', n_freqs=256, printCWTSpec=False):
        if debug: print('cwt() called')
        
        # Perform CWT
        samplingPeriod = round(data['t'][1]-data['t'][0], 9)
        samplingRate = round(1/samplingPeriod, 7)
        s = data['PD']
        t = data['t']
        self.n_points = len(t)
        self.samplingRate = int(round(1/(t[1]-t[0])))
        
        if debug:
            print(f"CWT: {len(s)} samples, {self.samplingRate} Hz")
        
        s_r = s[::-1]
        s_pad = np.concatenate((s_r, s, s_r))
        # if debug: print(f"Padded signal length: {len(s_pad)}")  # Commented to reduce noise
        
        scales, vmax = get_cwt_scales(wavelet, n_freqs)
        cwtmatr, freqs = pywt.cwt(s_pad, scales, wavelet, sampling_period=samplingPeriod)
        
        # if debug: print(f"CWT matrix shape before cropping: {cwtmatr.shape}")  # Commented to reduce noise
        
        # Cropping out the padding
        cwtmatr = np.abs(cwtmatr[:, self.n_points:2*self.n_points])
        
        n_samples_window = int(self.windowLength * self.samplingRate / 1000)
        
        # if debug:  # Commented to reduce debug noise
        #     print(f"CWT matrix shape after cropping: {cwtmatr.shape}") # Should be (255, 999)
        #     print(f"Frequencies array shape: {freqs.shape}") # Should be (256,)
        #     print(f"Expected time points for {self.windowLength}ms window: {n_samples_window}")
        #     print(f"================================")
        
        if printCWTSpec:
            print(f'Wavelet: {wavelet}')
            print(f'Frequency range: {round(freqs[-1], 0)}-{round(freqs[0], 0)} Hz')
            print(f'Period range: {round(1000/freqs[0],2)}-{round(1000/freqs[-1],2)} ms')
        
        # For use in folder naming
        self.freq_min = int(round(freqs[-1]))
        self.freq_max = int(round(freqs[0]))
        
        # For use by cwt_plot
        return {'t': t, 'freqs': freqs, 'cwtmatr': cwtmatr, 'vmax': vmax, 'n_samples_window': n_samples_window}
        
    def cwt_plot_thread_safe(self, result_tuple):
        """Thread-safe version of cwt_plot that receives metadata from worker"""
        if debug: print(f"🔧 DEBUG: *** cwt_plot_thread_safe CALLED *** with {type(result_tuple)}")
        try:
            cwt_spec, metadata = result_tuple
            
            # Extract metadata
            trackid = metadata['trackid']
            wStart = metadata['wStart']
            wEnd = metadata['wEnd']
            label = metadata['label']
            windowed_data = metadata['windowed_data']
            
            # Temporarily store in instance variables for save_cwt compatibility
            original_trackid = getattr(self, 'trackid', None)
            original_wStart = getattr(self, 'wStart', None)
            original_wEnd = getattr(self, 'wEnd', None)
            original_label = getattr(self, 'label', None)
            original_windowed_data = getattr(self, 'current_windowed_data', None)
            
            self.trackid = trackid
            self.wStart = wStart
            self.wEnd = wEnd
            self.label = label
            self.current_windowed_data = windowed_data
            
            try:
                # Call the original cwt_plot logic
                self._cwt_plot_internal(cwt_spec)
            finally:
                # Restore original values (though this may not matter in auto mode)
                if original_trackid is not None:
                    self.trackid = original_trackid
                if original_wStart is not None:
                    self.wStart = original_wStart
                if original_wEnd is not None:
                    self.wEnd = original_wEnd
                if original_label is not None:
                    self.label = original_label
                if original_windowed_data is not None:
                    self.current_windowed_data = original_windowed_data
                    
        except Exception as e:
            print(f"Error in cwt_plot_thread_safe: {e}")
            traceback.print_exc()

    def cwt_plot(self, cwt_spec):
        """Original cwt_plot method - now calls internal implementation"""
        try:
            self._cwt_plot_internal(cwt_spec)
        except Exception as e:
            print(f"Error in cwt_plot: {e}")
            traceback.print_exc()
                 
    def _cwt_plot_internal(self, cwt_spec):
        """Internal implementation shared by both cwt_plot methods"""
        # Define CWT figure attributes
        dpi = 100
        rect = [0, 0, 1, 1]  # Full figure for pixel-perfect output
        
        t = cwt_spec['t']
        t_ms = t * 1000  # Convert to milliseconds
        
        # Find the indices corresponding to the window
        window_start_idx = np.argmin(np.abs(t_ms - self.wStart))
        window_end_idx = np.argmin(np.abs(t_ms - self.wEnd))
        n_samples = window_end_idx - window_start_idx
        
        # Check that window is full length, and skip to next file if end of signal has been reached
        if n_samples < cwt_spec['n_samples_window']:
            self.navigate(fileDirection = '+')
            return
        
        if debug:
            print(f"=== WINDOW EXTRACTION DEBUG ===")
            print(f"Window start: {self.wStart} ms, idx: {window_start_idx}")
            print(f"Window end: {self.wEnd} ms, idx: {window_end_idx}")
            print(f"Window width in samples: {n_samples}")
        
        # Extract the windowed portion of data
        t_windowed = t[window_start_idx:window_end_idx]
        cwt_windowed = cwt_spec['cwtmatr'][:, window_start_idx:window_end_idx]
        
        if debug:
            print(f"Windowed time array shape: {t_windowed.shape}")
            print(f"Windowed CWT matrix shape: {cwt_windowed.shape}")
        
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
            
            # Enable logging and create CSV log file for manual mode (first time only)
            if not self.logging:
                self.logging = True
                if debug: print("🔧 DEBUG: First manual label - enabling logging and creating CSV")
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
                 
    def save_cwt(self, label):
        self.label = label
        label_folder = self.get_label_folder()
        
        outputFPath = Path(label_folder, f'{self.trackid}_{round(self.wStart, 1)}-{round(self.wEnd, 1)}ms.png')
        if debug: print(f"🔧 DEBUG: Attempting to save image to: {outputFPath}")
        
        # Only write to CSV when logging is enabled (manual mode)
        if debug: print(f"🔧 DEBUG: save_cwt called, autoSave={self.autoSave}, logging={self.logging}, will write CSV: {self.logging}")
        if self.logging:
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
    def __init__(self, fn, *args, trackid=None, wStart=None, wEnd=None, label=None, windowed_data=None, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        # Store metadata for thread-safe processing (None means regular worker)
        self.trackid = trackid
        self.wStart = wStart
        self.wEnd = wEnd
        self.label = label
        self.windowed_data = windowed_data
        
        # Use metadata-aware signals if any metadata is provided
        self.has_metadata = any([trackid is not None, wStart is not None, wEnd is not None, 
                               label is not None, windowed_data is not None])
        
        if debug: print(f"🔧 DEBUG: === WORKER {trackid} ===\n🔧 DEBUG: Worker created with has_metadata={self.has_metadata}, trackid={trackid}")
        
        if self.has_metadata:
            self.signals = MetadataWorkerSignals()
        else:
            self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        # print(f'Worker instance running on thread: {int(QThread.currentThreadId())}')
        if debug: print(f"🔧 DEBUG: Worker.run() starting, has_metadata={self.has_metadata}")
        try:
            result = self.fn(*self.args, **self.kwargs)
            if debug: print(f"🔧 DEBUG: Worker.run() function completed successfully")
        except:
            if debug: print(f"🔧 DEBUG: Worker.run() ERROR during execution")
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            if self.has_metadata:
                if debug: print(f"🔧 DEBUG: Worker.run() emitting result with metadata")
                # Create metadata dict to pass with result
                metadata = {
                    'trackid': self.trackid,
                    'wStart': self.wStart,
                    'wEnd': self.wEnd,
                    'label': self.label,
                    'windowed_data': self.windowed_data
                }
                if debug: 
                    print(f"🔧 DEBUG: About to emit tuple: ({type(result)}, {type(metadata)})")
                    print(f"🔧 DEBUG: Metadata contents: {metadata}")
                
                self.signals.output.emit((result, metadata))  # Return result with metadata
                
                if debug: print(f"🔧 DEBUG: Signal emit completed")
            else:
                if debug: print(f"🔧 DEBUG: Worker.run() emitting result without metadata")
                self.signals.output.emit(result)  # Return result only
        finally:
            if debug: print(f"🔧 DEBUG: Worker.run() finished")
            self.signals.finished.emit()  # Done  


class WorkerSignals(QObject):
    # Defines the signals available from a running worker thread.
    finished = pyqtSignal()
    output = pyqtSignal(object)  # Original signal for regular workers
    error = pyqtSignal(tuple)

class MetadataWorkerSignals(QObject):
    # Defines the signals available from a metadata-aware worker thread.
    finished = pyqtSignal()
    output = pyqtSignal(object)  # Use object instead of tuple for better Qt compatibility
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
