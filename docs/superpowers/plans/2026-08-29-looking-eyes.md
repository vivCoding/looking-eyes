# looking-eyes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `looking-eyes/` — a Raspberry Pi + ST7789 LCD device that streams the camera to a remote MediaPipe person-detection server over WebRTC and moves cartoon eyes to track the biggest person in frame, with last-spot memory, dart-y idle wander, a 30 s backlight-off sleep, and a sleepy blink-open wake.

**Architecture:** Single process, three threads. Main thread runs the LCD render loop (gaze state machine → blink → draw → display, ~100 Hz). A tracker thread runs the WebRTC client (aiortc) publishing shared state (`persons`, `frame`, `connection_state`). An optional Flask thread serves a debug web view from the same shared state. Pure logic (`gaze.py`) is separated from hardware (`hardware.py`) and rendering (`eyes_renderer.py`).

**Tech Stack:** Python 3.12, aiortc, av, opencv, httpx, Flask, Pillow, luma.lcd + RPi.GPIO (Pi only), ST7789 320×240 SPI LCD.

**Testing note:** All verification is **manual** by design (user decision). Every task ends with a "Verify" step with exact commands and expected output; there is no pytest. Mind the new project is a git repo: `/mnt/d/projects/security-eyes/looking-eyes/.git`. Commit after **every task**.

## Global Constraints

- New folder: `looking-eyes/` next to the existing demo; the demo folder is **untouched**.
- Config lives in `looking-eyes/config.py` — every knob listed in the spec must be there, with the exact constant names used below.
- Track the **biggest bounding box** (largest area = `w*h`).
- No person in frame → hold last spot `LAST_SPOT_DWELL` (5.0 s) → dart-y saccade wander.
- No person for `NO_PERSON_SLEEP` (30.0 s) → backlight off, LCD black, render loop pauses; the clock **never pauses** for connection health (Q4/B).
- Person appears while asleep → backlight on → eyes closed `WAKE_DELAY` (0.4 s) → openness eases 0→1 over `WAKE_OPEN` (1.0 s) → tracking.
- Feed is mirrored (`cv2.flip(frame, 1)`); direct horizontal mapping — look below in `box_to_look` — is the default; `INVERT_LOOK_X` flips it.
- All look values are normalized to [-1, 1].
- Reuse `eyes_backlight.py` geometry/blink/mouth behavior; only the wander and the blink-open animation are new.
- Web debug view on by default (`WEB_VIEW_ENABLED = True`, port `WEB_PORT = 5000`), configurable off.
- Pip deps: direct dependencies only; numpy left unpinned (mediapipe on arm64 needs numpy<2).

---

### Task 1: Project scaffold — config + requirements

**Files:**
- Create: `looking-eyes/config.py`
- Create: `looking-eyes/requirements.txt`
- Create: `looking-eyes/.gitignore` (exists: `.venv`, `__pycache__`, `.env*`)

**Interfaces:**
- Produces: every constant name the later tasks import (listed below exactly).

- [ ] **Step 1: Write `requirements.txt`**

```text
# Direct dependencies only — let pip resolve transitive deps per platform.
# mediapipe constraints on arm64 need numpy<2, so numpy is intentionally
# left unpinned here.
opencv-python-headless>=4.8
aiortc>=1.9
av>=12
httpx>=0.27
flask>=3.1
pillow>=10
# LCD hardware (Pi only — imports guarded in hardware.py)
luma.lcd>=2.16
RPi.GPIO>=0.7.1
```

- [ ] **Step 2: Write `config.py`**

```python
"""looking-eyes configuration — every knob lives here."""

# --- MediaPipe server (WebRTC) ---
MEDIAPIPE_SERVER_URL = "http://10.0.0.22:8080"  # mediapipe-server signal+result endpoint
CAMERA_ID = 0
RECONNECT_DELAY = 3.0            # seconds between reconnect attempts
RENDER_INTERVAL = 0.01           # main render-loop sleep (seconds)

# --- LCD / hardware ---
DISPLAY_KIND = "sim"             # "st7789" on the Pi, "sim" for dev machines
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
TRACK_CENTER = False             # True: look at box center instead of head point
INVERT_LOOK_X = False            # flip horizontal look (reversed camera/LCD mount)
LOOK_SMOOTHING = 0.25            # EMA factor per frame toward the look target

# Last-spot / sleep / wake
LAST_SPOT_DWELL = 5.0            # hold last known spot after person leaves (s)
NO_PERSON_SLEEP = 30.0           # no person this long -> backlight off (s)
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
```

- [ ] **Step 3: Verify**

Run: `cd looking-eyes && /mnt/d/projects/security-eyes/.venv/bin/python -c "import config; print(config.WIDTH, config.NO_PERSON_SLEEP, config.WEB_PORT)"`
Expected: `320 30.0 5000` (no import errors).

- [ ] **Step 4: Commit**

```bash
cd looking-eyes && git add config.py requirements.txt .gitignore && git commit -m "feat: add config and requirements scaffold"
```

---

### Task 2: `eyes_renderer.py` — eyes, mouth, blinks

**Files:**
- Create: `looking-eyes/eyes_renderer.py`

**Interfaces:**
- Consumes: all geometry/color/timing constants from `config.py`.
- Produces (later tasks rely on these exact names):
  - `clamp(value, minimum, maximum) -> float`
  - `get_pupil_center(eye_center: tuple[int,int], look_x: float, look_y: float, openness: float) -> tuple[float,float]`
  - `draw_eyes_image(look_x: float, look_y: float, openness: float = 1.0) -> Image.Image`
  - `class BlinkState` with `__init__(now: float)`, `update(now: float) -> bool` (True = closed), `force(now: float)`.

`openness` semantics: `0.0` = fully closed (lid line only, like `closed=True` in the standalone); `1.0` = full open; values in between squash the eye/pupil vertically (sleepy blink-open). During a blink, pass `0.0`.

- [ ] **Step 1: Write `eyes_renderer.py`**

```python
"""PIL rendering of the cartoon eyes + mouth, plus the blink state machine."""
import math
import random

from PIL import Image, ImageDraw

from config import (
    WIDTH, HEIGHT,
    LEFT_EYE_CENTER, RIGHT_EYE_CENTER,
    EYE_RADIUS_X, EYE_RADIUS_Y, EYE_OUTLINE_WIDTH,
    PUPIL_RADIUS_X, PUPIL_RADIUS_Y, PUPIL_MARGIN,
    LID_LINE_WIDTH, LID_LINE_HALF_LENGTH,
    MOUTH_Y, MOUTH_HALF_LENGTH, MOUTH_LINE_WIDTH,
    MOUTH_TILT_DEGREES, MOUTH_GROW_MAX,
    BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX,
    BLINK_DURATION_MIN, BLINK_DURATION_MAX,
    DOUBLE_BLINK_GAP_MIN, DOUBLE_BLINK_GAP_MAX,
    DOUBLE_BLINK_CHANCE, TRIPLE_BLINK_CHANCE,
)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def get_pupil_center(
    eye_center: tuple[int, int],
    look_x: float,
    look_y: float,
    openness: float = 1.0,
) -> tuple[float, float]:
    """Pupil position for a look direction, inside a squashed (open) eye."""
    o = clamp(openness, 0.0, 1.0)
    rx = EYE_RADIUS_X - PUPIL_RADIUS_X - PUPIL_MARGIN
    ry = (EYE_RADIUS_Y - PUPIL_RADIUS_Y - PUPIL_MARGIN) * o
    offset_x = look_x * rx
    offset_y = look_y * ry
    return eye_center[0] + offset_x, eye_center[1] + offset_y


def _draw_lid_lines(draw: ImageDraw.ImageDraw) -> None:
    for x, y in [LEFT_EYE_CENTER, RIGHT_EYE_CENTER]:
        draw.line(
            (x - LID_LINE_HALF_LENGTH, y, x + LID_LINE_HALF_LENGTH, y),
            fill="white",
            width=LID_LINE_WIDTH,
        )


def draw_eyes_image(
    look_x: float,
    look_y: float,
    openness: float = 1.0,
) -> Image.Image:
    """Render one 320x240 frame of the eyes.

    openness 0.0 = shut (lid lines only, matching the standalone blink),
    1.0 = fully open, intermediate = vertically squashed eye (waking).
    """
    img = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(img)

    o = clamp(openness, 0.0, 1.0)
    if o <= 0.0:
        _draw_lid_lines(draw)
        # Mouth still reacts to gaze even with closed eyes (small).
        _draw_mouth(draw, look_x, look_y)
        return img

    ry = EYE_RADIUS_Y * o
    py = PUPIL_RADIUS_Y * o

    for x, y in [LEFT_EYE_CENTER, RIGHT_EYE_CENTER]:
        draw.ellipse(
            (x - EYE_RADIUS_X, y - ry, x + EYE_RADIUS_X, y + ry),
            fill="white",
            outline="black",
            width=EYE_OUTLINE_WIDTH,
        )
    for eye_center in [LEFT_EYE_CENTER, RIGHT_EYE_CENTER]:
        px, pyc = get_pupil_center(eye_center, look_x, look_y, o)
        draw.ellipse(
            (px - PUPIL_RADIUS_X, pyc - py, px + PUPIL_RADIUS_X, pyc + py),
            fill="black",
        )

    _draw_mouth(draw, look_x, look_y)
    return img


def _draw_mouth(draw: ImageDraw.ImageDraw, look_x: float, look_y: float) -> None:
    """Neutral mouth that tilts opposite the gaze and grows with deflection."""
    look_mag = min(1.0, math.hypot(look_x, look_y))
    half = MOUTH_HALF_LENGTH * (1.0 + (MOUTH_GROW_MAX - 1.0) * look_mag)
    angle = math.radians(-MOUTH_TILT_DEGREES * look_x)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    mx, my = 160, MOUTH_Y
    draw.line(
        (mx - half * cos_a, my - half * sin_a,
         mx + half * cos_a, my + half * sin_a),
        fill="white",
        width=MOUTH_LINE_WIDTH,
    )


class BlinkState:
    """Periodic blinks; occasionally a double (rarely triple) blink."""

    def __init__(self, now: float) -> None:
        self.closed = False
        self.close_until = 0.0
        self.next_blink = now + random.uniform(BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX)
        self.cluster_remaining = 0
        self.cluster_gap = 0.2

    def schedule_next_cluster(self, now: float) -> None:
        self.next_blink = now + random.uniform(BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX)
        roll = random.random()
        if roll < DOUBLE_BLINK_CHANCE:
            self.cluster_remaining = 1   # double blink
        elif roll < DOUBLE_BLINK_CHANCE + TRIPLE_BLINK_CHANCE:
            self.cluster_remaining = 2   # rare triple blink
        else:
            self.cluster_remaining = 0
        self.cluster_gap = random.uniform(DOUBLE_BLINK_GAP_MIN, DOUBLE_BLINK_GAP_MAX)

    def update(self, now: float) -> bool:
        """Advance the blink state machine; returns True while eyes are closed."""
        if self.closed:
            if now >= self.close_until:
                self.closed = False
                if self.cluster_remaining > 0:
                    self.cluster_remaining -= 1
                    self.next_blink = now + self.cluster_gap
                else:
                    self.schedule_next_cluster(now)
        else:
            if now >= self.next_blink:
                self.closed = True
                self.close_until = now + random.uniform(BLINK_DURATION_MIN, BLINK_DURATION_MAX)
        return self.closed

    def force(self, now: float) -> None:
        """Trigger an immediate blink (used for saccade micro-blinks)."""
        self.closed = True
        self.close_until = now + random.uniform(BLINK_DURATION_MIN, BLINK_DURATION_MAX)
```

- [ ] **Step 2: Verify by rendering sample frames to PNG files**

Run:

```bash
cd looking-eyes
/mnt/d/projects/security-eyes/.venv/bin/python - <<'EOF'
from eyes_renderer import draw_eyes_image, BlinkState
import time
draw_eyes_image(0.0, 0.0, 1.0).save("/tmp/eyes_open.png")
draw_eyes_image(0.8, -0.5, 1.0).save("/tmp/eyes_look_ur.png")
draw_eyes_image(0.0, 0.0, 0.0).save("/tmp/eyes_closed.png")
draw_eyes_image(0.0, 0.0, 0.4).save("/tmp/eyes_half.png")
b = BlinkState(time.monotonic()); b.force(time.monotonic())
print("closed=True ->", b.update(time.monotonic()))
EOF
ls -la /tmp/eyes_*.png
```

Expected: four PNG files exist; visually inspect `/tmp/eyes_look_ur.png` — pupils sit up-right and the mouth tilts down-left; `/tmp/eyes_closed.png` shows only lid lines; `/tmp/eyes_half.png` shows squash-oval eyes. `closed=True -> True`.

- [ ] **Step 3: Commit**

```bash
cd looking-eyes && git add eyes_renderer.py && git commit -m "feat: eyes/mouth/blink renderer with blink-open openness"
```

---

### Task 3: `gaze.py` — person selection, look mapping, state machine

**Files:**
- Create: `looking-eyes/gaze.py`

**Interfaces:**
- Consumes: `config.py` constants; raw persons = list of dicts `{"box": {"x","y","w","h"}, "label", "score"}` in normalized coordinates (same shape as the WebRTC `"persons"` data channel).
- Produces (later tasks rely on these exact names):
  - `class Person` — dataclass with `x, y, w, h, score, label`; properties `area`, `cx`, `cy`.
  - `parse_persons(raw) -> list[Person]` (skips malformed dicts).
  - `pick_target_person(persons: list[Person]) -> Person | None` (largest **area**).
  - `box_to_look(person: Person, invert_x: bool = INVERT_LOOK_X, head_fraction: float = HEAD_FRACTION, track_center: bool = TRACK_CENTER) -> tuple[float, float]`
  - `class LookSmoother` — `__init__(factor)`, `update(tx, ty) -> (x, y)`, `reset()`.
  - `class SaccadeWander` — `__init__()`, `update(now, dt) -> ((x, y), reblink: bool)`.
  - `class GazeController` — `__init__(now)`, `update(now, dt, raw_persons)`; readable attributes: `state`, `look_x`, `look_y`, `openness`, `reblink`, `backlight_on`, `tracked`, `persons`.

Coordinate convention: normalized [0,1], feed is mirrored so screen-left == physical-left; `look_x = cx*2 - 1` (negative = left), `look_y = cy*2 - 1` (negative = up, since screen-y grows downward and the pupil offset is `look_y * radius`).

- [ ] **Step 1: Write `gaze.py`**

```python
"""Pure gaze logic: person selection, look mapping, and the wake/sleep
state machine. No hardware, no rendering — fully testable by hand."""
import math
import random
from dataclasses import dataclass

from config import (
    HEAD_FRACTION, TRACK_CENTER, INVERT_LOOK_X, LOOK_SMOOTHING,
    LAST_SPOT_DWELL, NO_PERSON_SLEEP, WAKE_DELAY, WAKE_OPEN,
    SACCADE_INTERVAL_MIN, SACCADE_INTERVAL_MAX, SACCADE_DURATION,
    SACCADE_EDGE_BIAS, SACCADE_REBLINK_CHANCE,
)


@dataclass
class Person:
    x: float
    y: float
    w: float
    h: float
    score: float = 0.0
    label: str = "person"

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


def parse_persons(raw) -> list[Person]:
    """Convert raw server JSON into Person objects, skipping malformed ones."""
    out: list[Person] = []
    for p in raw or []:
        try:
            box = p["box"]
            out.append(Person(
                x=float(box["x"]), y=float(box["y"]),
                w=float(box["w"]), h=float(box["h"]),
                score=float(p.get("score", 0.0)),
                label=str(p.get("label", "person")),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def pick_target_person(persons: list[Person]) -> Person | None:
    """Biggest bounding box wins."""
    if not persons:
        return None
    return max(persons, key=lambda p: p.area)


def box_to_look(
    person: Person,
    invert_x: bool = INVERT_LOOK_X,
    head_fraction: float = HEAD_FRACTION,
    track_center: bool = TRACK_CENTER,
) -> tuple[float, float]:
    """Map a person box (normalized, mirrored feed) to a look direction."""
    cx = person.cx
    cy = person.y + person.h * head_fraction if not track_center else person.cy
    look_x = (cx * 2.0 - 1.0) * (-1.0 if invert_x else 1.0)
    look_y = cy * 2.0 - 1.0
    return look_x, look_y


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class LookSmoother:
    """Exponential moving average toward a target; softens box switcheroo."""

    def __init__(self, factor: float = LOOK_SMOOTHING) -> None:
        self.factor = factor
        self.x = 0.0
        self.y = 0.0
        self._has = False

    def update(self, tx: float, ty: float) -> tuple[float, float]:
        if not self._has:
            self.x, self.y = tx, ty
            self._has = True
        else:
            self.x += self.factor * (tx - self.x)
            self.y += self.factor * (ty - self.y)
        return self.x, self.y

    def reset(self) -> None:
        self._has = False


class SaccadeWander:
    """Random dart-y gaze: ease to a target, hold briefly, re-pick.
    Targets are uniform-random, with a configurable bias toward screen edges
    (eyes scanning the room). Occasionally flags a micro-blink on landing."""

    def __init__(self) -> None:
        self.pos = (0.0, 0.0)
        self.target = (0.0, 0.0)
        self.start = (0.0, 0.0)
        self.phase = 1.0
        self.hold_until = 0.0
        self._reblink_pending = False
        self.reblink = False

    def _pick_target(self) -> None:
        if random.random() < SACCADE_EDGE_BIAS:
            side = random.choice(["left", "right", "top", "bottom"])
            if side in ("left", "right"):
                tx = -0.95 if side == "left" else 0.95
                ty = random.uniform(-0.8, 0.8)
            else:
                ty = -0.95 if side == "top" else 0.95
                tx = random.uniform(-0.8, 0.8)
        else:
            tx = random.uniform(-1.0, 1.0)
            ty = random.uniform(-1.0, 1.0)
        self.start = self.pos
        self.target = (tx, ty)
        self.phase = 0.0
        self._reblink_pending = random.random() < SACCADE_REBLINK_CHANCE

    def update(self, now: float, dt: float) -> tuple[tuple[float, float], bool]:
        """Returns ((x, y), reblink). reblink fires once when a saccade lands."""
        self.reblink = False
        if now >= self.hold_until:
            self.hold_until = now + random.uniform(
                SACCADE_INTERVAL_MIN, SACCADE_INTERVAL_MAX)
            self._pick_target()
        was_below = self.phase < 1.0
        self.phase = min(1.0, self.phase + dt / SACCADE_DURATION)
        landed = was_below and self.phase >= 1.0
        if landed and self._reblink_pending:
            self.reblink = True
            self._reblink_pending = False
        t = self.phase * self.phase * (3.0 - 2.0 * self.phase)  # smoothstep
        self.pos = (
            self.start[0] + (self.target[0] - self.start[0]) * t,
            self.start[1] + (self.target[1] - self.start[1]) * t,
        )
        return self.pos, self.reblink


class GazeController:
    """State machine driving gaze + sleep/wake.

    States: "tracking" | "last_spot" | "wander" | "asleep" | "waking".
    Attributes read by main.py / web_view.py:
      state, look_x, look_y, openness, reblink, backlight_on, tracked, persons
    """

    def __init__(self, now: float) -> None:
        self.state = "wander"
        self.look_x = 0.0
        self.look_y = 0.0
        self.openness = 1.0
        self.reblink = False
        self.backlight_on = True
        self.tracked: Person | None = None
        self.persons: list[Person] = []

        self.last_detection_time = now   # so startup doesn't sleep immediately
        self.last_spot_look = (0.0, 0.0)
        self.wake_until = 0.0
        self.smoother = LookSmoother(LOOK_SMOOTHING)
        self.wander = SaccadeWander()

    def update(self, now: float, dt: float, raw_persons) -> None:
        persons = parse_persons(raw_persons)
        person = pick_target_person(persons)
        self.tracked = person
        self.persons = persons
        self.reblink = False

        if person is not None:
            self.last_detection_time = now
            tx, ty = box_to_look(person)
            self.look_x, self.look_y = self.smoother.update(tx, ty)
            self.last_spot_look = (self.look_x, self.look_y)

            if self.state in ("asleep",):
                self.state = "waking"
                self.wake_until = now + WAKE_DELAY
            elif self.state == "waking":
                pass  # wake timing handled below; keep tracking target
            else:
                self.state = "tracking"
        else:
            # No person: pick a non-tracking state.
            if self.state == "asleep":
                pass  # stay asleep; clock keeps counting
            elif self.state == "waking":
                self.state = "last_spot"   # abort the wake
            elif self.state in ("tracking", "last_spot"):
                if now - self.last_detection_time > LAST_SPOT_DWELL:
                    self.state = "wander"
                else:
                    self.state = "last_spot"
            else:  # wander
                self.state = "wander"

        # Sleep clock (never pauses, per design).
        if now - self.last_detection_time > NO_PERSON_SLEEP:
            self.state = "asleep"

        # Fill in per-state look/openness/reblink.
        if self.state in ("tracking", "waking"):
            pass  # look already set above; blinks override openness in renderer
        elif self.state == "last_spot":
            self.look_x, self.look_y = self.last_spot_look
        elif self.state == "wander":
            (wx, wy), reblink = self.wander.update(now, dt)
            self.look_x, self.look_y = _clamp(wx, -1.0, 1.0), _clamp(wy, -1.0, 1.0)
            self.reblink = reblink
        elif self.state == "asleep":
            pass  # render loop pauses; values irrelevant

        # Openness: 1.0 except during the sleepy wake ramp.
        if self.state == "waking":
            elapsed = now - (self.wake_until - WAKE_DELAY)
            if elapsed < WAKE_DELAY:
                self.openness = 0.0
            else:
                self.openness = min(1.0, (elapsed - WAKE_DELAY) / WAKE_OPEN)
                if self.openness >= 1.0:
                    self.state = "tracking"
        else:
            self.openness = 1.0

        self.backlight_on = self.state != "asleep"
```

- [ ] **Step 2: Verify the state machine by driving it with scripted persons**

Run (a 70 s synthetic timeline covering every state):

```bash
cd looking-eyes
/mnt/d/projects/security-eyes/.venv/bin/python - <<'EOF'
import time
from gaze import GazeController

now = 1000.0
g = GazeController(now)
dt = 0.05
P1 = [{"box": {"x": 0.2, "y": 0.1, "w": 0.3, "h": 0.6}, "score": 0.9}]
P2 = [{"box": {"x": 0.7, "y": 0.2, "w": 0.2, "h": 0.5}, "score": 0.8}]
# 1) person appears -> tracking, biggest wins
for _ in range(20):
    now += dt; g.update(now, dt, P1)
assert g.state == "tracking", g.state
assert -0.35 < g.look_x < -0.25, g.look_x    # P1 cx=0.35 -> look_x ~ -0.3
# P1 then P2: P2 bigger (0.2*0.5=0.10 > 0.3*0.6=0.18? no) -> use a bigger P2
P2 = [{"box": {"x": 0.7, "y": 0.2, "w": 0.5, "h": 0.5}, "score": 0.8}]  # area 0.25
for _ in range(200):
    now += dt; g.update(now, dt, P2)
assert g.state == "tracking"
assert g.look_x > 0.3, g.look_x               # swept right toward P2 cx=0.95
# 2) person leaves -> last_spot for 5s then wander
now += dt; g.update(now, dt, [])
assert g.state == "last_spot", g.state
while now < g.last_detection_time + 5.5:
    now += dt; g.update(now, dt, [])
assert g.state == "wander", g.state
v = (g.look_x, g.look_y)
for _ in range(60):
    now += dt; g.update(now, dt, [])
assert (g.look_x, g.look_y) != v, "wander should move"
# 3) 30s no person -> asleep
while now < g.last_detection_time + 32.0:
    now += dt; g.update(now, dt, [])
assert g.state == "asleep" and not g.backlight_on
# 4) person returns -> waking (closed then opening) -> tracking
now += dt; g.update(now, dt, P1)
assert g.state == "waking" and g.openness == 0.0
wake_start = g.wake_until - 0.4
while now < wake_start + 0.45:
    now += dt; g.update(now, dt, P1)
assert g.openness < 0.2
while now < wake_start + 0.4 + 1.2:
    now += dt; g.update(now, dt, P1)
assert g.state == "tracking" and g.openness == 1.0 and g.backlight_on
print("gaze state machine OK; final look:", round(g.look_x, 2), round(g.look_y, 2))
EOF
```

Expected: prints `gaze state machine OK; ...` with no assertion failures. This exercises tracking, biggest-box switching, last-spot → wander, sleep, and wake — all offline, no hardware.

- [ ] **Step 3: Verify `box_to_look` mapping (mirror + inversion)**

Run:

```bash
cd looking-eyes
/mnt/d/projects/security-eyes/.venv/bin/python - <<'EOF'
from gaze import Person, box_to_look
p = Person(0.5, 0.5, 0.1, 0.2)
x, y = box_to_look(p)                    # center -> 0,0
assert abs(x) < 1e-6 and abs(y) < 1e-6
p = Person(0.0, 0.0, 0.1, 0.2)
x, y = box_to_look(p)                    # head near top-left -> look left+up
assert x < 0 and y < 0, (x, y)
x, y = box_to_look(p, invert_x=True)     # inverted -> right
assert x > 0, x
p = Person(0.4, 0.2, 0.2, 0.4)           # same box, center vs head
xc, _ = box_to_look(p, track_center=True)
xh, yh = box_to_look(p, track_center=False)
assert abs(xc - xh) < 1e-6 and yh < 0, (xc, xh, yh)  # head is higher than center
print("box_to_look mapping OK")
EOF
```

Expected: prints `box_to_look mapping OK`.

- [ ] **Step 4: Commit**

```bash
cd looking-eyes && git add gaze.py && git commit -m "feat: gaze state machine, biggest-person selection, saccade wander"
```

---

### Task 4: `hardware.py` — display + backlight glue

**Files:**
- Create: `looking-eyes/hardware.py`

**Interfaces:**
- Consumes: `config.py` display constants; a PIL image from `eyes_renderer`.
- Produces: `class Display` — `__init__(kind=DISPLAY_KIND)`, `.is_on: bool`, `.set_backlight(on: bool)`, `.show(pil_image)`, `.clear()`, `.cleanup()`. Only this file may import `luma` / `RPi.GPIO`, and only when `kind == "st7789"`.

Sim mode renders to an OpenCV window (needs a display — WSLg works on Windows 11; on a headless box use st7789).

- [ ] **Step 1: Write `hardware.py`**

```python
"""Display glue: ST7789 LCD + backlight GPIO, or an OpenCV sim window.

This is the ONLY module that touches luma / RPi.GPIO, and only when
DISPLAY_KIND == "st7789". Sim mode lets you run everything on a dev machine.
"""
import cv2
import numpy as np

from config import (
    DISPLAY_KIND, WIDTH, HEIGHT, BL_PIN,
    SPI_PORT, SPI_DEVICE, SPI_GPIO_DC, SPI_GPIO_RST,
    SPI_BUS_SPEED_HZ, DISPLAY_ROTATE,
)


class Display:
    def __init__(self, kind: str = DISPLAY_KIND) -> None:
        self.kind = kind
        self._lcd = None
        self.is_on = True
        if kind == "st7789":
            import RPi.GPIO as GPIO
            from luma.core.interface.serial import spi
            from luma.lcd.device import st7789
            self._GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(BL_PIN, GPIO.OUT)
            serial = spi(
                port=SPI_PORT,
                device=SPI_DEVICE,
                gpio_DC=SPI_GPIO_DC,
                gpio_RST=SPI_GPIO_RST,
                bus_speed_hz=SPI_BUS_SPEED_HZ,
            )
            self._lcd = st7789(serial, width=WIDTH, height=HEIGHT, rotate=DISPLAY_ROTATE)
        else:
            self._window = "looking-eyes (sim)"
            cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)

    def set_backlight(self, on: bool) -> None:
        if on == self.is_on:
            return
        self.is_on = on
        if self.kind == "st7789":
            self._GPIO.output(BL_PIN, self._GPIO.HIGH if on else self._GPIO.LOW)
        else:
            print(f"[sim] backlight {'ON' if on else 'OFF'}")

    def show(self, img) -> None:
        """Display a PIL image."""
        if self.kind == "st7789":
            self._lcd.display(img)
        else:
            cv2.imshow(self._window, np.array(img.convert("RGB")))
            cv2.waitKey(1)

    def clear(self) -> None:
        from PIL import Image
        self.show(Image.new("RGB", (WIDTH, HEIGHT), "black"))

    def cleanup(self) -> None:
        if self.kind == "st7789":
            self._GPIO.cleanup()
        else:
            cv2.destroyAllWindows()
```

- [ ] **Step 2: Verify in sim mode**

Run:

```bash
cd looking-eyes
/mnt/d/projects/security-eyes/.venv/bin/python - <<'EOF'
import time
from hardware import Display
from eyes_renderer import draw_eyes_image
d = Display("sim")
d.set_backlight(True)
for i in range(60):
    d.show(draw_eyes_image(1.0 - (i / 30.0), 0.5, 1.0))
    time.sleep(0.02)
print("backlight off test")
d.set_backlight(False)
d.show(draw_eyes_image(0.0, 0.0, 1.0))
time.sleep(1)
d.set_backlight(True)
d.clear()
d.cleanup()
EOF
```

Expected: a window shows the eyes sweeping right→left, then a `[sim] backlight OFF` line is printed, then the window clears and closes. (Close the window with the X if it lingers.)

- [ ] **Step 3: Commit**

```bash
cd looking-eyes && git add hardware.py && git commit -m "feat: display/backlight glue with sim mode"
```

---

### Task 5: `person_tracker.py` — WebRTC person client

**Files:**
- Create: `looking-eyes/person_tracker.py`

**Interfaces:**
- Consumes: `config.py` (`MEDIAPIPE_SERVER_URL`, `CAMERA_ID`, `RECONNECT_DELAY`); `httpx`, `aiortc`, `av`, `cv2` (same approach as the demo).
- Produces: `class PersonTracker` — `__init__(server_url=MEDIAPIPE_SERVER_URL, camera_id=CAMERA_ID, reconnect_delay=RECONNECT_DELAY)`, `start()`, `stop()`, and read-only properties `persons` (list of dicts as received from the `"persons"` data channel, empty when none/offline), `frame` (mirrored BGR ndarray copy or `None`), `connection_state` (one of `"idle"`, `"connecting"`, `"connected"`, `"failed"`). Thread-safe.

The camera track class from the demo (`CameraVideoTrack`) is **copied in** (small, self-contained) so this folder has no cross-folder import.

- [ ] **Step 1: Write `person_tracker.py`**

```python
"""WebRTC person tracker: streams camera frames to mediapipe-server and
publishes person detections as shared state (thread-safe)."""
import asyncio
import json
import logging
import threading
import time

import cv2
import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

from config import MEDIAPIPE_SERVER_URL, CAMERA_ID, RECONNECT_DELAY

log = logging.getLogger("looking-eyes.tracker")


class CameraVideoTrack(VideoStreamTrack):
    """Captures camera frames (mirrored) and yields them as a video track."""

    def __init__(self, camera_id: int):
        super().__init__()
        self._cap = cv2.VideoCapture(camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        self._running = True
        self.latest_frame = None
        self.frames_sent = 0

    async def recv(self):
        if not self._running:
            raise StopAsyncIteration()
        success, frame = self._cap.read()
        if not success:
            raise StopAsyncIteration()
        frame = cv2.flip(frame, 1)
        self.latest_frame = frame.copy()
        self.frames_sent += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pts, time_base = await self.next_timestamp()
        video_frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

    def stop(self):
        self._running = False
        self._cap.release()
        super().stop()


class PersonTracker:
    """Runs the WebRTC person loop in a background thread and publishes state."""

    def __init__(
        self,
        server_url: str = MEDIAPIPE_SERVER_URL,
        camera_id: int = CAMERA_ID,
        reconnect_delay: float = RECONNECT_DELAY,
    ) -> None:
        self.server_url = server_url
        self.camera_id = camera_id
        self.reconnect_delay = reconnect_delay
        self._running = False
        self._lock = threading.Lock()
        self._persons = []
        self._frame = None
        self._connection_state = "idle"
        self._loop = None
        self._thread = None
        self._pc = None

    # --- thread-safe state ---
    @property
    def persons(self) -> list:
        with self._lock:
            return list(self._persons)

    @property
    def frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def connection_state(self) -> str:
        with self._lock:
            return self._connection_state

    def _set_persons(self, value) -> None:
        with self._lock:
            self._persons = value

    def _set_frame(self, frame) -> None:
        with self._lock:
            self._frame = frame

    def _set_state(self, state: str) -> None:
        with self._lock:
            if state != self._connection_state:
                self._connection_state = state
                log.info("connection state -> %s", state)

    # --- lifecycle ---
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, args=(self._loop,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._stop_async)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._set_persons([])
        self._set_state("idle")

    def _run_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        try:
            loop.run_until_complete(self._run())
        except Exception:
            log.exception("tracker loop crashed")
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    def _stop_async(self) -> None:
        async def _close():
            if self._pc is not None:
                await self._pc.close()
        asyncio.ensure_future(_close(), loop=self._loop)

    async def _run(self) -> None:
        while self._running:
            self._set_state("connecting")
            try:
                await self._session()
            except Exception as exc:
                log.warning("session ended: %s", exc)
                self._set_persons([])
                self._set_state("failed")
            if self._running:
                await asyncio.sleep(self.reconnect_delay)

    async def _session(self) -> None:
        pc = RTCPeerConnection()
        self._pc = pc
        persons = []

        @pc.on("iceconnectionstatechange")
        def on_ice():
            log.info("ICE: %s", pc.iceConnectionState)

        @pc.on("connectionstatechange")
        def on_conn():
            log.info("conn: %s", pc.connectionState)

        camera_track = CameraVideoTrack(self.camera_id)
        pc.addTrack(camera_track)

        dc = pc.createDataChannel("persons")

        @dc.on("open")
        def on_dc_open():
            log.info("persons data channel open")

        @dc.on("message")
        def on_dc_message(message):
            nonlocal persons
            try:
                persons = json.loads(message)
            except json.JSONDecodeError:
                log.warning("malformed persons message ignored")
                return
            self._set_persons(persons)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.server_url}/webrtc-offer",
                json={"sdp": offer.sdp, "type": "offer"},
                timeout=15,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"signaling failed: {resp.status_code}")
            answer = RTCSessionDescription(
                sdp=resp.json()["sdp"], type=resp.json()["type"])
            await pc.setRemoteDescription(answer)

        for _ in range(50):
            if pc.iceConnectionState in ("connected", "completed"):
                break
            await asyncio.sleep(0.2)
        else:
            log.warning("ICE not connected after 10 s (%s)", pc.iceConnectionState)

        self._set_state("connected")
        try:
            while self._running:
                frame = camera_track.latest_frame
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                self._set_frame(frame)
                await asyncio.sleep(0.01)
        finally:
            await pc.close()
```

- [ ] **Step 2: Verify against the real server (on the Pi, or dev machine reachable to 10.0.0.22)**

Run:

```bash
cd looking-eyes
/mnt/d/projects/security-eyes/.venv/bin/python - <<'EOF'
import logging, time
logging.basicConfig(level=logging.INFO)
from person_tracker import PersonTracker
t = PersonTracker()
t.start()
try:
    time.sleep(15)
    print("state:", t.connection_state)
    print("persons:", len(t.persons))
    print("frame:", None if t.frame is None else t.frame.shape)
finally:
    t.stop()
EOF
```

Expected: logs show `conn -> connected`, `persons data channel open`; final print shows `state: connected`, and with someone in front of the camera `persons: >= 1` with a box dict, `frame: (H, W, 3)`. If the server is unreachable, expect `state: failed` and reconnect logs every `RECONNECT_DELAY` s.

- [ ] **Step 3: Commit**

```bash
cd looking-eyes && git add person_tracker.py && git commit -m "feat: WebRTC person tracker as thread-safe shared state"
```

---

### Task 6: `web_view.py` — debug web page

**Files:**
- Create: `looking-eyes/web_view.py`

**Interfaces:**
- Consumes: `PersonTracker` (`.persons`, `.frame`, `.connection_state`) and `GazeController` (`.state`, `.look_x`, `.look_y`, `.tracked`, `.persons`) from earlier tasks; `config.py` web constants.
- Produces: `start_web_view(tracker, gaze, enabled: bool = WEB_VIEW_ENABLED, host: str = WEB_HOST, port: int = WEB_PORT) -> None` — no-op when `enabled` is False; otherwise starts a daemon Flask thread serving:
  - `GET /` — HTML page with an `<img src="/feed">` and a status panel.
  - `GET /feed` — MJPEG stream: mirrored frame, every person box, the **tracked** box highlighted + `"TRACKED"` label, and a status line.
  - `GET /status` — JSON: `{state, look_x, look_y, connection_state, num_persons, tracked}`.

Person boxes are drawn from `tracker.persons` (raw dicts, same normalized format the demo used); the tracked highlight matches the box that `gaze.tracked` selected (compare rounded center + area).

- [ ] **Step 1: Write `web_view.py`**

```python
"""Debug web view: mirrored camera feed + person boxes + gaze status."""
import json
import threading
import time

import cv2
from flask import Flask, Response

from config import WEB_VIEW_ENABLED, WEB_HOST, WEB_PORT

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>looking-eyes debug</title></head><body style="background:#111;color:#eee;
font-family:monospace">
<h2>looking-eyes</h2>
<img src="/feed" style="max-width:640px;border:1px solid #444">
<div id="st">status…</div>
<script>
setInterval(async () => {
  const s = await (await fetch('/status')).json();
  document.getElementById('st').textContent =
    'state=' + s.state +
    ' | look=' + s.look_x.toFixed(2) + ',' + s.look_y.toFixed(2) +
    ' | conn=' + s.connection_state +
    ' | persons=' + s.num_persons + ' | tracked=' + s.tracked;
}, 500);
</script></body></html>"""

TRACK_COLOR = (80, 220, 80)     # BGR green: the person being looked at
OTHER_COLOR = (120, 120, 120)   # dim gray: other people
STATUS_COLOR = (0, 255, 255)


def _box_key(box: dict) -> tuple:
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    return (round((x + w / 2) * 100), round((y + h / 2) * 100), round(w * h * 10000))


def _tracked_key(tracker, gaze):
    p = gaze.tracked
    if p is None:
        return None
    return (round(p.cx * 100), round(p.cy * 100), round(p.area * 10000))


def _draw_snapshot(tracker, gaze) -> bytes:
    frame = tracker.frame
    if frame is None:
        return None
    h, w = frame.shape[:2]
    tracked_key = _tracked_key(tracker, gaze)
    for person in tracker.persons:
        try:
            box = person["box"]
            x1 = max(0, min(w, int(box["x"] * w)))
            y1 = max(0, min(h, int(box["y"] * h)))
            x2 = max(0, min(w, int((box["x"] + box["w"]) * w)))
            y2 = max(0, min(h, int((box["y"] + box["h"]) * h)))
        except (KeyError, TypeError, ValueError):
            continue
        is_tracked = _box_key(box) == tracked_key
        color = TRACK_COLOR if is_tracked else OTHER_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if is_tracked:
            cv2.putText(frame, "TRACKED", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    status = (f"state={gaze.state} conn={tracker.connection_state} "
              f"persons={len(tracker.persons)} look=({gaze.look_x:.2f},{gaze.look_y:.2f})")
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, STATUS_COLOR, 1, cv2.LINE_AA)
    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        return None
    return buf.tobytes()


def _feed(tracker, gaze):
    while True:
        jpeg = _draw_snapshot(tracker, gaze)
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(0.05)


def start_web_view(
    tracker,
    gaze,
    enabled: bool = WEB_VIEW_ENABLED,
    host: str = WEB_HOST,
    port: int = WEB_PORT,
) -> None:
    if not enabled:
        return
    app = Flask(__name__)

    @app.route("/")
    def index():
        return PAGE

    @app.route("/feed")
    def feed():
        return Response(_feed(tracker, gaze),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/status")
    def status():
        return json.dumps({
            "state": gaze.state,
            "look_x": round(gaze.look_x, 3),
            "look_y": round(gaze.look_y, 3),
            "connection_state": tracker.connection_state,
            "num_persons": len(tracker.persons),
            "tracked": None if gaze.tracked is None else {
                "cx": round(gaze.tracked.cx, 3),
                "cy": round(gaze.tracked.cy, 3),
                "area": round(gaze.tracked.area, 4),
            },
        })

    t = threading.Thread(target=lambda: app.run(host=host, port=port,
                                                debug=False, use_reloader=False),
                         daemon=True)
    t.start()
    print(f"[web] debug view at http://{host}:{port}")
```

- [ ] **Step 2: Verify**

Run the tracker against the real server, then start the web view and check it in a browser (needs camera + server reachable):

```bash
cd looking-eyes
/mnt/d/projects/security-eyes/.venv/bin/python - <<'EOF'
import logging, time
logging.basicConfig(level=logging.INFO)
from person_tracker import PersonTracker
from gaze import GazeController
from web_view import start_web_view
t = PersonTracker(); t.start()
g = GazeController(time.monotonic())
start_web_view(t, g)
try:
    for _ in range(120):
        g.update(time.monotonic(), 0.05, t.persons)
        time.sleep(0.05)
finally:
    t.stop()
EOF
```

Expected: with the camera live, `http://<host>:5000/` shows the mirrored feed; the biggest person's box is green with `TRACKED`, others are dim gray; the status panel shows `state=tracking conn=connected persons=N tracked=…` and `look` values that move as the person moves. `curl -s localhost:5000/status` returns the JSON object.

- [ ] **Step 3: Commit**

```bash
cd looking-eyes && git add web_view.py && git commit -m "feat: debug web view with tracked-person highlight"
```

---

### Task 7: `main.py` — integration + render loop

**Files:**
- Create: `looking-eyes/main.py`

**Interfaces:**
- Consumes: `Display`, `PersonTracker`, `GazeController`, `BlinkState`, `draw_eyes_image`, `start_web_view` — exact names from Tasks 2-6.
- Produces: runnable entry point `main()` and `if __name__ == "__main__": main()`.

Render loop per iteration (~100 Hz):
1. `now = time.monotonic()`; `dt = now - last`.
2. `gaze.update(now, dt, tracker.persons)`.
3. Backlight transitions only on change (`gaze.backlight_on`).
4. If asleep → skip drawing (LCD is black; backlight off), sleep, continue.
5. `closed = blink.update(now)`; if `gaze.reblink: blink.force(now)`.
6. `openness = 0.0 if closed else gaze.openness`.
7. `display.show(draw_eyes_image(gaze.look_x, gaze.look_y, openness))`.

Shutdown (KeyboardInterrupt): stop tracker, clear display, backlight off, `display.cleanup()`.

- [ ] **Step 1: Write `main.py`**

```python
"""looking-eyes entry point: wires WebRTC tracker + gaze + LCD render loop."""
import time

from config import DISPLAY_KIND, RENDER_INTERVAL
from person_tracker import PersonTracker
from gaze import GazeController
from eyes_renderer import BlinkState, draw_eyes_image
from hardware import Display
from web_view import start_web_view


def main() -> None:
    display = Display(DISPLAY_KIND)
    tracker = PersonTracker()
    tracker.start()
    gaze = GazeController(time.monotonic())
    blink = BlinkState(time.monotonic())
    start_web_view(tracker, gaze)

    display.set_backlight(True)
    last = time.monotonic()
    print("[looking-eyes] running — Ctrl-C to stop")
    try:
        while True:
            now = time.monotonic()
            dt = max(1e-4, now - last)
            last = now

            gaze.update(now, dt, tracker.persons)

            if gaze.backlight_on != display.is_on:
                display.set_backlight(gaze.backlight_on)

            if gaze.state == "asleep":
                time.sleep(RENDER_INTERVAL)
                continue

            closed = blink.update(now)
            if gaze.reblink:
                blink.force(now)

            openness = 0.0 if closed else gaze.openness
            display.show(draw_eyes_image(gaze.look_x, gaze.look_y, openness))
            time.sleep(RENDER_INTERVAL)
    except KeyboardInterrupt:
        print("\n[looking-eyes] shutting down")
    finally:
        tracker.stop()
        display.clear()
        display.set_backlight(False)
        display.cleanup()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify (sim mode, no camera needed — gaze will wander)**

Run:

```bash
cd looking-eyes
timeout 8 /mnt/d/projects/security-eyes/.venv/bin/python main.py
```

Expected: an OpenCV window shows the eyes darting around (wander state, since `tracker.persons` is empty), a `[sim]` backlight stays ON during the 8 s, and Ctrl-C-equivalent (timeout SIGTERM) exits. Then run with a live camera + server and confirm full behavior: tracking, last-spot, wander, sleep (backlight OFF after 30 s), and the sleepy wake on reappearance — checked visually against the checklist in the design doc.

- [ ] **Step 3: Verify with `DISPLAY_KIND="st7789"` on the Pi**

Set `DISPLAY_KIND = "st7789"` in `config.py`, ensure `MEDIAPIPE_SERVER_URL` points at the reachable server, `pip install -r requirements.txt`, then run `python main.py` and watch the LCD: eyes track the biggest person, blink, micro-blink on saccades, sleep after 30 s without people, sleepy-blink-open on wake, mouth tilts opposite gaze. Ctrl-C clears the screen and turns the backlight off.

- [ ] **Step 4: Commit**

```bash
cd looking-eyes && git add main.py && git commit -m "feat: integrate tracker, gaze, and LCD render loop"
```

---

### Task 8: `README.md` and final review

**Files:**
- Create: `looking-eyes/README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
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
  [mediapipe-server](https://github.com/…/mediapipe-server) reachable at
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
```

(Replace the mediapipe-server link with the actual repo URL when known.)

- [ ] **Step 2: Final review against the design spec**

Skim `docs/superpowers/specs/2026-08-29-looking-eyes-design.md` and check every numbered success criterion:
1. Tracking biggest person — covered (gaze.py `pick_target_person`).
2. Last spot ~5 s then dart — covered (`LAST_SPOT_DWELL`), wander saccades.
3. 30 s → backlight off — covered (`NO_PERSON_SLEEP`, `main.py` skip).
4. Mirror view + tracked highlight — covered (`web_view.py`).
5. Mouth/blinks still behave — covered (`eyes_renderer.py`).
6. Ctrl-C cleanup — covered (`main.py` finally block).

- [ ] **Step 3: Commit**

```bash
cd looking-eyes && git add README.md && git commit -m "docs: add README"
```

---

## Self-Review Notes (already checked)

- **Spec coverage:** every spec section maps to a task (see Task 8 Step 2 table); the only spec item without its own file is the manual test checklist — it is the verification steps themselves.
- **Placeholders:** none; every code step contains complete code. The README keeps the mediapipe-server reference as plain text (the server is external; no repo URL assumed).
- **Type consistency:** `PersonTracker.persons` (list[dict]) → `parse_persons(raw)` (list[Person]) → `pick_target_person` → `box_to_look`; `GazeController.state/look_x/look_y/openness/reblink/backlight_on/tracked/persons` used identically in `main.py` and `web_view.py`. `BlinkState.force` defined in Task 2, used in Task 7.