"""PIL rendering of the line-art face + mouth, plus the blink state machine.

Open eyes are single vertical line strokes that translate with the gaze
(up to EYE_TRAVEL_X / EYE_TRAVEL_Y). Eyebrows keep a mild inward frown and
tilt with the look direction. Blink/closed eyes form >_< angle brackets.
The mouth is a neutral line that tilts opposite the gaze and grows with
look deflection. Wake-up plays a rapid blink burst.
"""
import math
import random

from PIL import Image, ImageDraw

from config import (
    WIDTH, HEIGHT,
    LEFT_EYE_CENTER, RIGHT_EYE_CENTER,
    EYE_LINE_LENGTH, EYE_LINE_WIDTH, EYE_TRAVEL_X, EYE_TRAVEL_Y,
    BROW_Y, BROW_HALF_LENGTH, BROW_WIDTH, BROW_FROWN, BROW_LIFT, BROW_TILT,
    BROW_SWAY, BROW_SIDE_LIFT,
    BLINK_REACH_X, BLINK_REACH_Y,
    MOUTH_Y, MOUTH_HALF_LENGTH, MOUTH_LINE_WIDTH,
    MOUTH_TILT_DEGREES, MOUTH_GROW_MAX,
    BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX,
    BLINK_DURATION_MIN, BLINK_DURATION_MAX,
    DOUBLE_BLINK_GAP_MIN, DOUBLE_BLINK_GAP_MAX,
    DOUBLE_BLINK_CHANCE, TRIPLE_BLINK_CHANCE,
    WAKE_BLINK_COUNT, WAKE_BLINK_GAP,
)


def _eye_offsets(look_x: float, look_y: float) -> tuple[float, float]:
    """Pixel offset of the eye strokes for a look direction."""
    return look_x * EYE_TRAVEL_X, look_y * EYE_TRAVEL_Y


def _draw_brows(draw: ImageDraw.ImageDraw, look_x: float, look_y: float) -> None:
    """Eyebrows: frown with the eyes, rise/lower with vertical gaze, and the
    brow on the looked-toward side raises (looking right raises the right
    brow). Inner ends sway with horizontal gaze."""
    base_y = BROW_Y + look_y * BROW_LIFT          # look up -> brows rise
    slope = BROW_FROWN + look_x * BROW_TILT       # inner end hangs lower
    sway = look_x * BROW_SWAY                     # inner ends sway with look_x
    for cx, _cy in [LEFT_EYE_CENTER, RIGHT_EYE_CENTER]:
        inner_dir = 1.0 if cx < WIDTH / 2 else -1.0   # toward the nose
        brow_y = base_y + inner_dir * look_x * BROW_SIDE_LIFT  # side you look toward raises
        inner_x = cx + inner_dir * BROW_HALF_LENGTH + sway
        outer_x = cx - inner_dir * BROW_HALF_LENGTH - sway
        inner_y = brow_y + slope
        outer_y = brow_y - slope
        draw.line((outer_x, outer_y, inner_x, inner_y), fill="white", width=BROW_WIDTH)


def _draw_eyes(draw: ImageDraw.ImageDraw, look_x: float, look_y: float) -> None:
    """Open eyes: vertical line strokes that translate with the gaze."""
    ox, oy = _eye_offsets(look_x, look_y)
    half = EYE_LINE_LENGTH / 2.0
    for cx, cy in [LEFT_EYE_CENTER, RIGHT_EYE_CENTER]:
        x, y = cx + ox, cy + oy
        draw.line((x, y - half, x, y + half), fill="white", width=EYE_LINE_WIDTH)


def _draw_closed(draw: ImageDraw.ImageDraw, look_x: float, look_y: float) -> None:
    """Closed eyes: long flat >_< — left eye '>' (apex right), right '<'."""
    ox, oy = _eye_offsets(look_x, look_y)
    for cx, cy in [LEFT_EYE_CENTER, RIGHT_EYE_CENTER]:
        x, y = cx + ox, cy + oy
        apex_x = x + BLINK_REACH_X if cx < WIDTH / 2 else x - BLINK_REACH_X
        draw.line((x, y - BLINK_REACH_Y, apex_x, y), fill="white", width=EYE_LINE_WIDTH)
        draw.line((x, y + BLINK_REACH_Y, apex_x, y), fill="white", width=EYE_LINE_WIDTH)


def draw_eyes_image(
    look_x: float,
    look_y: float,
    closed: bool = False,
) -> Image.Image:
    """Render one 320x240 frame of the line-art face.

    closed=True draws the >_< blink (normal blinks and the wake burst);
    otherwise open line-stroke eyes + eyebrows + mouth.
    """
    img = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(img)

    if closed:
        _draw_closed(draw, look_x, look_y)
    else:
        _draw_brows(draw, look_x, look_y)
        _draw_eyes(draw, look_x, look_y)

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

    def burst(self, now: float,
              count: int = WAKE_BLINK_COUNT,
              gap: float = WAKE_BLINK_GAP) -> None:
        """Play `count` rapid blinks (wake-up sequence) starting immediately.

        The remaining blinks follow via the cluster gap mechanism in update().
        """
        self.closed = True
        self.close_until = now + random.uniform(BLINK_DURATION_MIN, BLINK_DURATION_MAX)
        self.cluster_remaining = max(0, count - 1)
        self.cluster_gap = gap
