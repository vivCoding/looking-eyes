"""looking-eyes entry point: wires WebRTC tracker + gaze + LCD render loop."""
import logging
import time

from config import RENDER_INTERVAL
from person_tracker import PersonTracker
from gaze import GazeController
from eyes_renderer import BlinkState, draw_eyes_image
from hardware import Display
from web_view import start_web_view

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> None:
    display = Display()
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