"""Pure gaze logic: person selection, look mapping, and the wake/sleep
state machine. No hardware, no rendering — fully testable by hand."""
import math
import random
from dataclasses import dataclass

from config import (
    HEAD_FRACTION, TRACK_CENTER, INVERT_LOOK_X, LOOK_SMOOTHING,
    LAST_SPOT_DWELL, NO_PERSON_SLEEP, WAKE_STATE_DURATION,
    WAKE_TRANSITION, SLEEP_SETTLE,
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
        t = 1.0 - (1.0 - self.phase) ** 3  # ease-out: quick start, smooth settle
        self.pos = (
            self.start[0] + (self.target[0] - self.start[0]) * t,
            self.start[1] + (self.target[1] - self.start[1]) * t,
        )
        return self.pos, self.reblink


class GazeController:
    """State machine driving gaze + sleep/wake.

    States: "tracking" | "last_spot" | "wander" | "asleep" | "waking" | "closing".
    Attributes read by main.py / web_view.py:
      state, look_x, look_y, reblink, backlight_on, tracked, persons, wake_pending
    """

    def __init__(self, now: float) -> None:
        self.state = "wander"
        self.look_x = 0.0
        self.look_y = 0.0
        self.wake_pending = False
        self.reblink = False
        self.backlight_on = True
        self.tracked: Person | None = None
        self.persons: list[Person] = []

        self.last_detection_time = now   # so startup doesn't sleep immediately
        self.last_spot_look = (0.0, 0.0)
        self.wake_until = 0.0
        self.smoother = LookSmoother(LOOK_SMOOTHING)
        self.wander = SaccadeWander()
        self._transition: tuple | None = None   # (start_x, start_y, tx, ty, t0, dur)

    def _start_transition(
        self,
        target: tuple[float, float],
        now: float,
        duration: float,
        start: tuple[float, float] | None = None,
    ) -> None:
        """Begin an ease-out transition of the look from `start` (default: the
        current look) to `target` over `duration` seconds."""
        sx, sy = (self.look_x, self.look_y) if start is None else start
        tx, ty = target
        self._transition = (sx, sy, tx, ty, now, duration)

    def _advance_transition(self, now: float) -> tuple[float, float] | None:
        """Return the eased look while a transition is active; on completion
        clear it and return the final target; else None."""
        if self._transition is None:
            return None
        sx, sy, tx, ty, t0, dur = self._transition
        if now >= t0 + dur:
            self._transition = None
            return tx, ty
        p = (now - t0) / dur
        t = 1.0 - (1.0 - p) ** 3   # ease-out, same as saccade wander
        return sx + (tx - sx) * t, sy + (ty - sy) * t

    def update(self, now: float, dt: float, raw_persons) -> None:
        persons = parse_persons(raw_persons)
        person = pick_target_person(persons)
        self.tracked = person
        self.persons = persons
        self.reblink = False
        self.wake_pending = False

        if person is not None:
            self.last_detection_time = now
            tx, ty = box_to_look(person)
            self.look_x, self.look_y = self.smoother.update(tx, ty)
            self.last_spot_look = (self.look_x, self.look_y)

            if self.state in ("asleep",):
                self.state = "waking"
                self.wake_until = now + WAKE_STATE_DURATION
                self.wake_pending = True   # main.py plays the blink burst once
            elif self.state == "waking":
                pass  # wake timing handled below; keep tracking target
            elif self.state == "closing":
                self.state = "tracking"
                self._transition = None    # abort the settle-to-center transition
            else:
                self.state = "tracking"
        else:
            # No person: pick a non-tracking state.
            if self.state == "asleep":
                pass  # stay asleep; clock keeps counting
            elif self.state == "waking":
                self.state = "last_spot"   # abort the wake
            elif self.state == "closing":
                pass  # keep settling to center
            elif self.state in ("tracking", "last_spot"):
                if now - self.last_detection_time > LAST_SPOT_DWELL:
                    self.state = "wander"
                else:
                    self.state = "last_spot"
            else:  # wander
                self.state = "wander"

        # Sleep clock (never pauses, per design).
        if now - self.last_detection_time > NO_PERSON_SLEEP and self.state != "asleep":
            if self.state != "closing":
                self.state = "closing"
                self._start_transition((0.0, 0.0), now, SLEEP_SETTLE)  # settle to center

        # Fill in per-state look/reblink.
        if self.state == "waking":
            self.look_x, self.look_y = 0.0, 0.0   # pupils centered while waking
        elif self.state == "tracking":
            trans = self._advance_transition(now)
            if trans is not None:
                self.look_x, self.look_y = trans  # saccade toward the person
        elif self.state == "last_spot":
            self.look_x, self.look_y = self.last_spot_look
        elif self.state == "wander":
            (wx, wy), reblink = self.wander.update(now, dt)
            self.look_x, self.look_y = _clamp(wx, -1.0, 1.0), _clamp(wy, -1.0, 1.0)
            self.reblink = reblink
        elif self.state == "closing":
            trans = self._advance_transition(now)
            if trans is not None:
                self.look_x, self.look_y = trans  # settling toward center
            else:
                self.state = "asleep"             # centered -> backlight off
        elif self.state == "asleep":
            pass  # render loop pauses; values irrelevant

        # Wake: backlight on, blink burst (driven by main.py via wake_pending),
        # pupils centered; once the wake window elapses, saccade toward the
        # tracked person instead of snapping.
        if self.state == "waking" and now >= self.wake_until:
            self.state = "tracking"
            if self.tracked is not None:
                tx, ty = box_to_look(self.tracked)
                self.smoother.reset()
                self._start_transition((tx, ty), now, WAKE_TRANSITION,
                                       start=(0.0, 0.0))

        self.backlight_on = self.state != "asleep"