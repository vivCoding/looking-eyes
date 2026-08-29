# looking-eyes

A Raspberry Pi + ST7789 LCD that *looks at people*. It streams the camera to
a remote MediaPipe person-detection server over WebRTC, gets person bounding
boxes back, and moves cartoon eyes on the LCD to track the biggest person in
frame — with last-spot memory, dart-y idle wandering, a 30 s backlight-off
sleep, and a sleepy blink-open wake.

## Hardware

- Raspberry Pi (64-bit), camera, ST7789 320×240 SPI LCD (DC=25, RST=24,
  backlight on GPIO 18 — see `config.py`).
- A running
  `mediapipe-server` reachable at
  `MEDIAPIPE_SERVER_URL` (default `http://10.0.0.22:8080`), serving the
  WebRTC offer endpoint `POST /webrtc-offer` and routing person detections to
  a `"persons"` data channel.

## Setup (Pi)

```bash
cd looking-eyes
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# edit config.py: DISPLAY_KIND="st7789", MEDIAPIPE_SERVER_URL
python main.py
```

## Dev machine (sim mode)

Keep `DISPLAY_KIND="sim"`; install `opencv-python` (GUI build) instead of
`opencv-python-headless`. `python main.py` opens an OpenCV window showing
what the LCD would show, plus the debug web view at
`http://localhost:5000` (mirrored feed, tracked-person highlight, status).

## Behavior

- Multiple people: the **biggest bounding box** is tracked.
- Person leaves: eyes hold the last spot ~5 s, then dart around randomly
  (saccades, occasionally re-blinking) until someone reappears.
- No person for 30 s: backlight off. On reappearance the eyes wake with a
  sleepy blink-open (~1 s).
- The mouth tilts opposite the gaze and grows with look deflection.

## Config

Everything lives in `config.py`: server URL, camera id, display geometry and
pins, look mapping (`HEAD_FRACTION`/`TRACK_CENTER`/`INVERT_LOOK_X`),
smoothing, saccade timings, last-spot/sleep/wake durations, eye/mouth/blink
geometry, and the web view toggle (`WEB_VIEW_ENABLED`).

## Files

| File | Job |
|---|---|
| `config.py` | all knobs |
| `person_tracker.py` | WebRTC person tracker (thread-safe shared state) |
| `gaze.py` | pure gaze/sleep/wake logic |
| `eyes_renderer.py` | PIL eyes/mouth/blink rendering |
| `hardware.py` | ST7789 + backlight (or sim window) glue |
| `web_view.py` | debug web page |
| `main.py` | wiring + render loop |