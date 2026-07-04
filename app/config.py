import cv2
"""
config.py

Central configuration file for the Vision System.
Modify project parameters here instead of changing code throughout the project.
"""
MIN_PERSON_WIDTH = 80
MIN_PERSON_HEIGHT = 80

# ==========================================================
# CAMERA SETTINGS
# ==========================================================

# Camera Source
# Use 0 for webcam or RTSP_URL for IP camera.

RTSP_URL = (
    "rtsp://192.168.10.100:8554/webcam"
)

FRAME_WIDTH = 704
FRAME_HEIGHT = 480
# Camera

USE_RTSP = True

WEBCAM_INDEX = 0


# ==========================================================
# YOLO
# ==========================================================

YOLO_MODEL = "models/yolo11n.pt"

DEVICE = "cpu"

PERSON_CLASS_ID = 0

TRACKER_CONFIG = "bytetrack.yaml"

# ==========================================================
# TRACKING SETTINGS
# ==========================================================

TRACKER_CONFIG = "bytetrack.yaml"

TRACK_TIMEOUT = 30        # frames

# ==========================================================
# FACE RECOGNITION SETTINGS
# ==========================================================

FACE_DATABASE_PATH = "faces"

FACE_MODEL_NAME = "buffalo_s"

FACE_DETECTION_SIZE = (640, 640)

SIMILARITY_THRESHOLD = 0.55

RECHECK_INTERVAL = 30          # frames

# ==========================================================
# DISPLAY
# ==========================================================

WINDOW_NAME = "Vision System"

FONT = cv2.FONT_HERSHEY_SIMPLEX

FONT_SCALE = 0.55

FONT_THICKNESS = 2

KNOWN_COLOR = (0, 255, 0)

UNKNOWN_COLOR = (0, 0, 255)

FPS_COLOR = (0, 0, 255)

TEXT_COLOR = (255, 255, 0)

PERSON_COUNT_COLOR = (255, 0, 0)

# ==========================================================
# PERFORMANCE
# ==========================================================

PRINT_STATS_INTERVAL = 5        # seconds

# ==========================================================
# COLORS (BGR)
# ==========================================================

PERSON_BOX_COLOR = (0, 255, 0)

UNKNOWN_BOX_COLOR = (0, 0, 255)

TEXT_COLOR = (255, 255, 255)

# ==========================================================
# FONT SETTINGS
# ==========================================================

FONT_SCALE = 0.6

FONT_THICKNESS = 2

# ==========================================================
# PATHS
# ==========================================================

LOG_DIRECTORY = "logs"

MODEL_DIRECTORY = "models"

# ==========================================================
# FUTURE OPTIONS
# ==========================================================

SAVE_LOGS = False

SAVE_VIDEO = False

VIDEO_OUTPUT_PATH = "output/output.mp4"

ENABLE_MULTIPROCESSING = False

ENABLE_FFMPEG_CAPTURE = False
