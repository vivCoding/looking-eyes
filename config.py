"""looking-eyes configuration — every knob lives here."""

# --- MediaPipe server (WebRTC) ---
MEDIAPIPE_SERVER_URL = "http://10.0.0.22:8080"  # mediapipe-server signal+result endpoint
CAMERA_ID = 0
RECONNECT_DELAY = 3.0            # seconds between reconnect attempts
RENDER_INTERVAL = 0.01           # main render-loop sleep (seconds)

# --- Camera capture ---
# The Pi encodes frames in software (aiortc), so capping resolution/fps keeps
# encode cost low and results near-realtime; person detection is unaffected at
# VGA resolutions. Set to 0 to leave the camera's own default.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 25

# --- LCD / hardware ---
WIDTH = 320
HEIGHT = 240
BL_PIN = 18                      # backlight GPIO (BCM)
SPI_PORT = 0
SPI_DEVICE = 0
SPI_GPIO_DC = 25
SPI_GPIO_RST = 24
SPI_BUS_SPEED_HZ = 40_000_000
DISPLAY_ROTATE = 2               # ST7789 rotation used by the standalone script

# --- Look / gaze ---
HEAD_FRACTION = 0.25             # look point = box.y + box.h * HEAD_FRACTION
TRACK_CENTER = True              # True: look at box center instead of head point
INVERT_LOOK_X = False            # flip horizontal look (reversed camera/LCD mount)
LOOK_SMOOTHING = 0.25            # EMA factor per frame toward the look target

# Last-spot / sleep / wake
LAST_SPOT_DWELL = 5.0            # hold last known spot after person leaves (s)
NO_PERSON_SLEEP = 10.0           # no person this long -> backlight off (s)
WAKE_DELAY = 0.4                 # eyes stay closed when waking (s)
WAKE_OPEN = 1.0                  # sleepy blink-open duration (s)

# Saccade (dart-y) wander
SACCADE_INTERVAL_MIN = 0.3
SACCADE_INTERVAL_MAX = 1.2
SACCADE_DURATION = 0.15          # ease-in-out time toward a wander target (s)
SACCADE_EDGE_BIAS = 0.3          # chance a wander target lands on an edge
SACCADE_REBLINK_CHANCE = 0.3     # chance of a micro-blink when a saccade lands

# --- Eye / mouth geometry (copied from eyes_backlight.py) ---
LEFT_EYE_CENTER = (90, 120)
RIGHT_EYE_CENTER = (230, 120)
EYE_RADIUS_X = 50
EYE_RADIUS_Y = 72
EYE_OUTLINE_WIDTH = 8
PUPIL_RADIUS_X = 16
PUPIL_RADIUS_Y = 20
PUPIL_MARGIN = 8
LID_LINE_WIDTH = 5
LID_LINE_HALF_LENGTH = 46
MOUTH_Y = 200
MOUTH_HALF_LENGTH = 14
MOUTH_LINE_WIDTH = 3
MOUTH_TILT_DEGREES = 18.0
MOUTH_GROW_MAX = 1.3

# --- Blink (copied from eyes_backlight.py) ---
BLINK_INTERVAL_MIN = 2.0
BLINK_INTERVAL_MAX = 6.0
BLINK_DURATION_MIN = 0.05
BLINK_DURATION_MAX = 0.15
DOUBLE_BLINK_GAP_MIN = 0.05
DOUBLE_BLINK_GAP_MAX = 0.1
DOUBLE_BLINK_CHANCE = 0.22
TRIPLE_BLINK_CHANCE = 0.04

# --- Web debug view ---
WEB_VIEW_ENABLED = True
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
