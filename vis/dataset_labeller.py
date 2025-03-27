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
        self.autoLabelPath = QLineEdit(r'E:\AlSi10Mg single layer ffc\CWT_ML_training_data\CWT_labelled_cw_powder_in-situ_ss_double_sliced\250123_14-06-21_1ms-window_all_labels.csv')
        self.btnAuto = QPushButton('Run auto')
        self.btnAuto.setEnabled(False)
        
        autoRow = QGroupBox('Auto labelling')
        autoLayout = QHBoxLayout()
        autoLayout.addWidget(self.autoLabelPath, stretch=6)
        autoLayout.addWidget(self.btnAuto, stretch=1)
        autoRow.setLayout(autoLayout)
        
        # CWT and radiograph image display
        self.cwtFig = plt.figure(frameon=self.show_axes)
        self.cwtCanvas = FigureCanvas(self.cwtFig)
        
        self.xrayFig = plt.figure(frameon=False)
        self.xrayCanvas = FigureCanvas(self.xrayFig)
        
        self.figRow = QGroupBox()
        figLayout = QHBoxLayout()
        figLayout.addWidget(self.cwtCanvas, stretch=1)
        figLayout.addWidget(self.xrayCanvas, stretch=6)
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
    
    
class Controller(QObject):
    updateReadout = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    
    def __init__(self, view, videoPlayer):
        super().__init__()
        print(f'Controller running on thread: {int(QThread.currentThreadId())} (main)')
        self.view = view   # Define the view for access from within the Controller object
        self.video_player = videoPlayer
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
        self.windowLength = self.view.windowLengthSelector.text()
        self.windowOffset = self.view.windowOffsetSelector.text()
        self.cmap = self.view.cmapSelector.currentText()
        self.wavelet = self.view.waveletSelector.currentText()
        self.n_freqs = self.view.freqsSelector.text()
        # Initialise sampling rate variable
        self.samplingRate = 0 # Hz
        # Define file locations
        self.folder = get_paths()['hdf5']
        self.outputFolder = Path(self.folder, 'CWT_labelled_windows')
        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder)
        self.logbook = get_logbook()
        # Create log file for storing window information and labels
        self.logging = True
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
        self.windowLength = value
        self.print_settings()
        
    def update_window_offset(self, value):
        self.windowOffset = value
        self.print_settings()
        
    def update_n_freqs(self, value):
        self.n_freqs = value
        self.print_settings()
        
        
    
    def show_video_player(self):
        # Show the video player window directly to the right of the main window
        main_geo = self.view.geometry()
        player_x = main_geo.x() + main_geo.width() + 10
        player_y = main_geo.y()
        self.video_player.move(player_x, player_y)
        self.video_player.show()
        
    def load_video(self):
        video_data = self.data.iloc[self.fIndex]['video']
        
        framerate = get_logbook_data(self.logbook, self.trackid)['framerate']
        start = int((self.windowOffset + self.wIndex * self.windowLength) / 1000 * framerate)
        end = int(start + framerate * self.windowLength / 1000)
        print(f'start, end = {start}, {end}')
        
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
        logRow = pd.DataFrame({'trackid' : [self.trackid],
            'window_n' : [self.wIndex],
            'window_start_ms' : [self.wStart],
            'window_end_ms' : [self.wEnd],
            'label' : [self.label]})
        logRow.to_csv(self.logPath, mode='a', index=False, header=False)
    
    def get_data(self):
        # Create log for storing results
        if self.logging == True:
            self.create_log()
        # Create file read job on worker thread
        print('Initialising Worker to read data')
        worker = Worker(self.read_files)
        worker.signals.output.connect(self.keep_data)
        worker.signals.finished.connect(lambda: self.view.set_readout_text('Done'))
        worker.signals.finished.connect(self.view.progressBar.reset)
        worker.signals.finished.connect(self.view.enable_controls)
        worker.signals.finished.connect(self.show_video_player)
        # Execute
        self.threadpool.start(worker)
    
    def read_files(self):
        trackids_filt = self.filter_logbook()['trackid'].to_list()
        all_files = sorted(glob.glob(f'{self.folder}/*.hdf5'))
        files = []
        for t in trackids_filt:
            for f in all_files:
                if t in f: files.append(f)
        
        self.nFiles = len(files)
        print(f'Reading {self.nFiles} files from \'{self.folder}\'')
        group, time, series, colour = ('AMPM', 'Time', 'Photodiode1Bits', 'r')
        data = {'trackid': [], 't': [], 'PD': [], 'xray': [], 'video': []} 
        for i, filepath in enumerate(sorted(files)):
            trackid = Path(filepath).name[:7]
            if trackid in self.exclude:
                continue
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
            self.wStart =  self.windowOffset + self.wIndex * self.windowLength
            self.wEnd = self.windowOffset + (self.wIndex + 1) * self.windowLength
            row = self.data.iloc[self.fIndex]
            self.trackid = row['trackid']
            print(self.trackid, ' window ', self.wIndex)
            worker = Worker(self.cwt, row)
            worker.signals.output.connect(self.cwt_plot)
            self.threadpool.start(worker)
            self.xray_plot()
            self.load_video()
            self.view.update_progress(int(100*(self.fIndex+1)/self.nFiles), self.trackid)
        except IndexError:
            print('No more files')
    
    def auto_label(self):
        self.autoSave = True
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
            self.wStart = row.window_start_ms
            self.wEnd = row.window_end_ms
            self.label = str(row.label)
            
            print(f'{self.trackid} {self.wStart}-{self.wEnd} ms')
            try:
                data_row = self.data.loc[self.trackid]
            except KeyError:
                print(self.trackid, ' not found, check filter_logbook() parameters')
                continue            
            
            worker = Worker(self.cwt, data = data_row,
                                      wavelet = self.wavelet,
                                      cmap = self.cmap,
                                      n_freqs = self.n_freqs,
                                      label = self.label,
                                      xmin = self.wStart,
                                      xmax = self.wEnd,
                                      auto_save = self.autoSave,
                                      outputFolder = self.outputFolder
                                      )
            sleep(0.2)
            self.threadpool.start(worker)
            
        self.view.set_readout_text('Done')    
        self.view.progressBar.reset()
        self.view.enable_controls()
            
        pass
    
    def cwt(self,
            data,
            wavelet = 'cmor1.5-1.0',
            cmap = 'jet',
            n_freqs = 256,
            printCWTSpec = False,
            label = None,
            xmin = None,
            xmax = None,
            auto_save = None,
            outputFolder = None,
            ):
            
        # Perform CWT
        samplingPeriod = round(data['t'][1]-data['t'][0], 9)
        samplingRate = round(1/samplingPeriod, 7)
        s = data['PD']
        t = data['t']
        n_points = len(t)
        
        s_r = s[::-1]
        s_pad = np.concatenate((s_r, s, s_r))
        
        scales, vmax = get_cwt_scales(wavelet, n_freqs)
        cwtmatr, freqs = pywt.cwt(s_pad, scales, wavelet, sampling_period=samplingPeriod)
        cwtmatr = np.abs(cwtmatr[:-1, n_points:2*n_points-1])
        
        if printCWTSpec:
            print(f'Wavelet: {wavelet}')
            print(f'Frequency range: {round(freqs[-1], 0)}-{round(freqs[0], 0)} Hz')
            print(f'Period range: {round(1000/freqs[0],2)}-{round(1000/freqs[-1],2)} ms')
            printCWTSpec = False # Only print these details once
        
        if not auto_save:
            return (t, freqs, cwtmatr)
        
        # Generate and save plot in the background
        dpi = 30
        rect = [0, 0, 1, 1]
        cwtFig = plt.figure(frameon=False)
        ax = plt.Axes(cwtFig, rect)
        tAx, fAx = np.meshgrid(t*1000, freqs/1000)
        ax.pcolormesh(tAx, fAx, cwtmatr, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_yscale('log', base=2)
        ax.set_xlim(xmin, xmax)
        cwtFig.set_figwidth(0.001*self.windowLength*samplingRate/100)
        cwtFig.set_figheight(len(fAx)/100)
        ax.set_axis_off()
        ax = cwtFig.add_axes(ax)
        
        label_folder = Path(outputFolder,
                            wavelet.replace('.', '_'),
                            f'{self.windowLength}_ms',
                            f'{int(round(freqs[-1]))}-{int(round(freqs[0]))}_Hz_{n_freqs}_steps',
                            cmap,
                            label)
                            
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
            
        outputFPath = Path(label_folder, f'{self.trackid}_{xmin}-{xmax}ms.png')
        cwtFig.savefig(outputFPath)
        
        plt.close()
        
        
    def cwt_plot(self, data, cmap='jet'):
        dpi = 30
        t, freqs, cwtmatr = data
        self.view.cwtFig.clear()
        rect = [0.2, 0.1, 0.75, 0.85] if self.view.show_axes == True else [0, 0, 1, 1]
        ax = plt.Axes(self.view.cwtFig, rect)
        self.tAx, self.fAx = np.meshgrid(t*1000, freqs/1000)
        ax.pcolormesh(self.tAx, self.fAx, cwtmatr, cmap=cmap, vmin=0, vmax=200)
        ax.set_yscale('log', base=2)
        ax.set_xlim(self.wStart, self.wEnd)
        
        if self.view.show_axes == True:
            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('Freq. [kHz]')
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        else:
            self.view.cwtFig.set_figwidth(0.001*self.windowLength*self.samplingRate/100)
            self.view.cwtFig.set_figheight(len(self.fAx)/100)
            ax.set_axis_off()
        ax = self.view.cwtFig.add_axes(ax)
        self.view.cwtCanvas.draw()

    def save_cwt(self, label):
        self.label = label
        label_folder = Path(self.outputFolder, label)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        outputFPath = Path(label_folder, f'{self.trackid}_w{self.wIndex}_{self.windowLength}ms_{self.windowOffset}ms-offset.png')
        self.background_write()
        self.view.cwtFig.savefig(outputFPath)

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
        # print(f'Worker instance running on thread: {int(QThread.currentThreadId())}')
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.output.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done  


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
    Controller(view, videoPlayer) # Initialise controller with access to the view
    sys.exit(Gui.exec())
    
if __name__ == '__main__':
    main()
