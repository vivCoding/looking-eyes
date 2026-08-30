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

from config import (
    MEDIAPIPE_SERVER_URL, CAMERA_ID, RECONNECT_DELAY,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
)

log = logging.getLogger("looking-eyes.tracker")


class CameraVideoTrack(VideoStreamTrack):
    """Captures camera frames (mirrored) and yields them as a video track."""

    def __init__(self, camera_id: int):
        super().__init__()
        self._cap = cv2.VideoCapture(camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        # Cap resolution/fps: the Pi encodes in software, so a high-res default
        # camera makes each frame expensive and delays the detection feed.
        if CAMERA_WIDTH:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        if CAMERA_HEIGHT:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        if CAMERA_FPS:
            self._cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
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
