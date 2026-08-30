# looking-eyes — Design

A Raspberry Pi + ST7789 LCD device that _looks at people_. It streams the
camera feed to a remote MediaPipe person-detection server over WebRTC,
receives person bounding boxes back, and moves cartoon eyes on the LCD to
face the person — biggest person in frame wins.

Combines two existing pieces:

- `opencv-mediapipe-stream-test/` — WebRTC client that sends camera frames to
  a `mediapipe-server` and receives `{box, label, score}` person detections
  over a `"persons"` data channel (normalized coords, mirrored feed).
- `eyes_backlight.py` — ST7789 (luma) + RPi.GPIO cartoon eyes: ovals with
  pupils, periodic (double/triple) blinks, and a mouth that tilts opposite
  the gaze.

## Behavior decisions (agreed)

| Topic                       | Decision                                                                                                  |
| --------------------------- | --------------------------------------------------------------------------------------------------------- |
| Multiple people             | Track the **biggest bounding box** (largest area).                                                        |
| No person in frame          | Hold **last known spot** for 5 s, then **dart-y random wander**.                                          |
| No person for 30 s          | **Backlight off** (LCD black, render loop pauses). Clock runs **regardless of connection health** (Q4/B). |
| Person appears while asleep | **Wake**: backlight on → brief closed (`>_<`) → ~3 rapid blinks → tracking.                   |
| Web view                    | Keep the demo's debug web view (camera + boxes + tracked highlight), **on by default, configurable off**. |
| Testing                     | **Manual only.** No automated test suite.                                                                 |
| Wander style                | Replace the deterministic cos/sin wander with random **saccades** (dart-y, unpredictable, snappy ease-out ~0.06 s).                |

## Architecture

Single process, three threads:

1. **Main thread** — LCD render loop: gaze state machine → blink → draw →
   display, plus backlight/GPIO control. Runs at ~100 Hz (`sleep 0.01`).
2. **Tracker thread** — WebRTC person tracker (`aiortc`), publishing shared
   state instead of an MJPEG generator.
3. **Web thread** — (optional) Flask debug page reading the same shared state.

Matches the demo's proven thread+loop pattern; one process, one command.

```
Camera ──► PersonTracker (WebRTC) ──► mediapipe-server
                │  latest_persons / latest_frame / connection_state
                ▼
            Gaze state machine ──► EyesRenderer ──► ST7789 LCD (+ backlight GPIO)
                │                        ▲
                └── tracked person ──────┘ (highlighted on web view)
```

## File layout (new `looking-eyes/` folder; demo untouched)

| File                | Job                                                                                                                                                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`         | Every knob (see below).                                                                                                                                                                                                         |
| `person_tracker.py` | WebRTC client refactored from `remote_person_webrtc.py` into a `PersonTracker` class running in its own thread; exposes `latest_persons`, `latest_frame`, `connection_state` (thread-safe). Auto-reconnects on connection loss. |
| `gaze.py`           | Pure logic, no hardware: person selection, box→look mapping, saccade wander, last-spot memory, wake/sleep state machine, timings.                                                                                               |
| `eyes_renderer.py`  | PIL line-art face: translating eye strokes, frowning/tilting eyebrows, `>_<` blink/closed shapes, mouth, and the blink state machine (incl. wake burst).                              |
| `web_view.py`       | Flask app: `/` HTML page, `/feed` MJPEG stream (mirrored feed + boxes + highlight on tracked person + status text), `/status` JSON (`persons`, gaze state, `look_x/y`, connection state).                                       |
| `hardware.py`       | Thin glue: `init_display()`, `set_backlight(on)`, display power-on/power-off. Only file importing `luma`/`RPi.GPIO`. Pi-only; requires luma + RPi.GPIO at import time.                                                                                                                   |
| `main.py`           | Wires the threads, runs the render loop, handles KeyboardInterrupt shutdown (clear LCD, backlight off, `GPIO.cleanup()`).                                                                                                       |
| `requirements.txt`  | aiortc, av, opencv-python-headless, httpx, flask, luma.lcd, pillow, RPi.GPIO.                                                                                                                                                   |
| `README.md`         | Setup + run instructions on the Pi, config pointers.                                                                                                                                                                            |

## Gaze state machine

States, in priority order. `look_x/look_y ∈ [-1, 1]`.

1. **TRACKING** — person detected → look at biggest box. Target point is the
   **head region** by default (`box.y + box.h * HEAD_FRACTION`,
   `HEAD_FRACTION = 0.25`), or box center (`TRACK_CENTER = True` config).
   Direct horizontal mapping is safe because the feed is mirrored
   (`look_x = (cx*2 - 1)`); `INVERT_LOOK_X` config flag covers a backwards
   camera/LCD mount. `look_y` from head/center as chosen.
2. **LAST_SPOT** — no person: hold last tracking direction for
   `LAST_SPOT_DWELL = 5.0` s. Blinking continues.
3. **WANDER** — random dart-y saccades (below) until a person reappears.
4. **ASLEEP** — `NO_PERSON_SLEEP = 30.0` s after the last detection (clock
   never pauses, per decision): backlight off, LCD black, render loop
   pauses, tracker keeps running.
5. **WAKING** — on next detection: backlight on, a rapid blink burst
   (`WAKE_BLINK_COUNT` = 3, `WAKE_BLINK_GAP` 0.15 s) plays via
   `wake_pending`, then TRACKING after `WAKE_STATE_DURATION` (0.7 s).

**Smoothing**: exponential moving average on the look target
(`LOOK_SMOOTHING = 0.25` per frame, configurable) — softens flicker when two
similar-size people trade the biggest-box spot.

**Dart-y wander** (replaces the deterministic cosine wander):

- Saccade every `SACCADE_INTERVAL_MIN..MAX = 0.3..1.2` s (random).
- Target: uniform random point in [-1,1]², but ~30% of the time biased to an
  **edge/corner** (eyes scanning the room).
- Ease-in-out toward the target over ~0.15 s, hold briefly, re-pick.
- Occasionally (config) **re-blink** right after a saccade lands, like real
  eyes micro-blink.

## Eyes / mouth / blink (reused from `eyes_backlight.py`)

- 320×240 portrait ST7789, `rotate=2`, SPI on the existing pins
  (DC=25, RST=24, `bus_speed_hz=40 MHz`), backlight `GPIO 18`.
- Line-art face: open eyes are vertical line strokes that translate with
  the gaze; eyebrows keep an inward frown and tilt with the gaze; blinks
  and the closed state draw `>_<` angle brackets; mouth tilt/grow
  behavior and blink intervals/double/triple odds live in `config.py`.
- Mouth keeps reacting to gaze (tilts opposite, grows with deflection).

## Config (`config.py`) — everything listed

Server/stream: `MEDIAPIPE_SERVER_URL` (default the demo's
`http://10.0.0.22:8080`), `CAMERA_ID`, `RECONNECT_DELAY`, frames per sec.
LCD/hardware: `WIDTH`, `HEIGHT`, `BL_PIN`, SPI pins/speed, rotation.
Look: `HEAD_FRACTION` / `TRACK_CENTER`, `INVERT_LOOK_X`, `LOOK_SMOOTHING`,
`SACCADE_*`, `LAST_SPOT_DWELL`, `NO_PERSON_SLEEP`, `WAKE_STATE_DURATION`,
`WAKE_BLINK_COUNT`, `WAKE_BLINK_GAP`, edge-bias odds.
Eyes/mouth/blink: line-art face geometry + blink interval/timing odds
(see above).
Web: `WEB_VIEW_ENABLED` (default True), `WEB_PORT`.

## Error handling

- **Camera fails to open** at startup → clear error and exit (same as demo).
- **WebRTC connection drops** → tracker logs and retries every
  `RECONNECT_DELAY`; persons are empty meanwhile, so the sleep clock keeps
  counting (per decision B) and the screen may go dark.
- **KeyboardInterrupt / SIGINT** → clear LCD, backlight off,
  `GPIO.cleanup()`, stop threads.
- No person tracking IDs from the server (only box/label/score) — never
  assume them; smoothing handles switcheroo flicker.

## Success criteria (manual test checklist)

1. With a person in frame: eyes track them; biggest box wins when several
   stand in front of the camera.
2. Person leaves: eyes hold the spot ~5 s, then dart around; after 30 s the
   backlight turns off.
3. Person reappears: backlight on, rapid blink burst, then tracking.
4. Mirror view in browser shows boxes with the tracked person highlighted;
   `/status` shows coherent state+liveness.
5. Blinks/double-blinks and the mouth still behave like the standalone
   script (eyes move, mouth tilts opposite).
6. Ctrl-C cleans up (screen cleared, backlight off, pins released).
