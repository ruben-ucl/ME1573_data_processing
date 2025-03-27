import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QHBoxLayout, QStyle, QComboBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap

class VideoPlayerWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
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
            
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Application")
        self.setGeometry(100, 100, 400, 200)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create title label
        title_label = QLabel("Grayscale Video Player")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Create buttons layout
        button_layout = QHBoxLayout()
        
        # Create button to open video player window
        self.open_player_button = QPushButton("Open Video Player")
        self.open_player_button.clicked.connect(self.show_video_player)
        button_layout.addWidget(self.open_player_button)
        
        # Add button to load test data
        self.load_test_button = QPushButton("Load Test Data")
        self.load_test_button.clicked.connect(self.load_test_data)
        button_layout.addWidget(self.load_test_button)
        
        # Add button layout to main layout
        layout.addLayout(button_layout)
        
        # Initialize video player window but don't show it yet
        self.video_player = VideoPlayerWindow()
        
    def show_video_player(self):
        """Show the video player window directly to the right of the main window"""
        main_geo = self.geometry()
        player_x = main_geo.x() + main_geo.width() + 10
        player_y = main_geo.y()
        self.video_player.move(player_x, player_y)
        self.video_player.show()
        
    def load_test_data(self):
        """Load test data: moving gradient pattern"""
        frames = 100
        height = 256
        width = 256
        
        # Create base gradient pattern
        x = np.linspace(0, 255, width, dtype=np.uint8)
        y = np.linspace(0, 255, height, dtype=np.uint8)
        base_pattern = np.zeros((height, width), dtype=np.uint8)
        
        # Generate moving pattern
        test_data = np.zeros((frames, height, width), dtype=np.uint8)
        for i in range(frames):
            # Create diagonal gradient with moving offset
            offset = i * 2
            pattern = (x[np.newaxis, :] + y[:, np.newaxis] + offset) % 256
            test_data[i] = pattern.astype(np.uint8)
        
        # Send data to video player
        self.video_player.set_data(test_data)
        
    def set_video_data(self, data):
        """
        Set video data from external source
        Args:
            data (numpy.ndarray): Array of shape (frames, height, width) with uint8 type
        """
        self.video_player.set_data(data)

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
    