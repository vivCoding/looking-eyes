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
