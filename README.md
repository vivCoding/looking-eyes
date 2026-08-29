# looking-eyes

A Raspberry Pi + ST7789 LCD that *looks at people*. It streams the camera to
a remote MediaPipe person-detection server over WebRTC, gets person bounding
boxes back, and moves cartoon eyes on the LCD to track the biggest person in
frame — with last-spot memory, dart-y idle wandering, a 30 s backlight-off
sleep, and a sleepy blink-open wake.

**Raspberry Pi-only.** This app runs on the Pi driving the ST7789 LCD;
`hardware.py` requires `luma.lcd` and `RPi.GPIO`, both installed from
`requirements.txt`. There is no desktop preview mode.

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
# edit config.py: MEDIAPIPE_SERVER_URL
python main.py
```

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
| `hardware.py` | ST7789 LCD + backlight GPIO glue (Pi-only) |
| `web_view.py` | debug web page |
| `main.py` | wiring + render loop |