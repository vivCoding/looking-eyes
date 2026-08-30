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
  return;
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
