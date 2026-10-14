#!/usr/bin/env python3
# Copyright © 2026-2026 atombite.ai
# SPDX-License-Identifier: Apache-2.0

"""Small Tk GUI for previewing every camera-bridge image stream."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from threading import Event, Lock, Thread
import time
import tkinter as tk
from tkinter import ttk
from typing import Sequence

import cv2
import numpy as np

from piper_control.camera_bridge.client import CameraSubscriber
from piper_control.camera_bridge.protocol import ReceivedFrame


TOPIC_LAYOUT = (
    ("camera/left_wrist/color", "Left wrist · Color", 0, 0),
    ("camera/right_wrist/color", "Right wrist · Color", 0, 1),
    ("camera/right_wrist/depth", "Right wrist · Depth", 0, 2),
    ("camera/head/color", "Head · Color", 1, 0),
    ("camera/head/depth", "Head · Depth", 1, 1),
    ("camera/head/infrared_1", "Head · Infrared 1", 1, 2),
    ("camera/head/infrared_2", "Head · Infrared 2", 2, 0),
)
IMAGE_TOPICS = tuple(item[0] for item in TOPIC_LAYOUT)
STATUS_TOPICS = (
    "camera/status/left_wrist",
    "camera/status/right_wrist",
    "camera/status/head",
)
PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 240
DEPTH_DISPLAY_MAX_M = 4.0


def normalize_endpoint(text: str) -> str:
    """Accept ``host:port`` or ``tcp://host:port`` and return a ZMQ URL."""
    value = text.strip()
    if value.startswith("tcp://"):
        address = value[6:]
    else:
        address = value
        value = f"tcp://{value}"
    if ":" not in address:
        raise ValueError("address must be IP:PORT")
    host, port_text = address.rsplit(":", 1)
    if not host:
        raise ValueError("IP or host name is required")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    return value


def frame_to_rgb(frame: ReceivedFrame) -> np.ndarray:
    """Decode one protocol frame to an H×W×3 uint8 RGB preview image."""
    if frame.encoding == "jpeg":
        bgr = cv2.imdecode(
            np.frombuffer(frame.payload, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if bgr is None or bgr.shape != (frame.height, frame.width, 3):
            raise ValueError("JPEG decode or dimensions failed")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if frame.stride_bytes is None:
        raise ValueError("raw frame has no stride")
    if len(frame.payload) != frame.stride_bytes * frame.height:
        raise ValueError("raw payload size does not match geometry")

    if frame.pixel_format == "z16":
        if frame.depth_scale is None or frame.stride_bytes % 2:
            raise ValueError("invalid Z16 metadata")
        row_values = frame.stride_bytes // 2
        depth = np.frombuffer(frame.payload, dtype="<u2").reshape(
            frame.height, row_values
        )[:, : frame.width]
        scaled = np.clip(
            depth.astype(np.float32) * frame.depth_scale / DEPTH_DISPLAY_MAX_M,
            0.0,
            1.0,
        )
        intensity = np.rint(scaled * 255.0).astype(np.uint8)
        bgr = cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO)
        bgr[depth == 0] = 0
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if frame.pixel_format == "y8":
        infrared = np.frombuffer(frame.payload, dtype=np.uint8).reshape(
            frame.height, frame.stride_bytes
        )[:, : frame.width]
        return cv2.cvtColor(infrared, cv2.COLOR_GRAY2RGB)

    raise ValueError(f"unsupported frame format: {frame.pixel_format}")


class FpsTracker:
    """Rolling receive-rate calculation for each image topic."""

    def __init__(self, window_s: float = 2.0) -> None:
        self._window_ns = int(window_s * 1_000_000_000)
        self._samples: defaultdict[
            str, deque[tuple[int, int, object | None]]
        ] = defaultdict(deque)

    def record(
        self,
        topic: str,
        timestamp_ns: int,
        *,
        sequence: int | None = None,
        session_id: object | None = None,
    ) -> None:
        samples = self._samples[topic]
        if samples and session_id is not None and samples[-1][2] != session_id:
            samples.clear()
        counter = sequence if sequence is not None else (
            samples[-1][1] + 1 if samples else 0
        )
        samples.append((timestamp_ns, counter, session_id))
        self._prune(samples, timestamp_ns)

    def fps(self, topic: str, now_ns: int) -> float:
        samples = self._samples[topic]
        self._prune(samples, now_ns)
        if len(samples) < 2:
            return 0.0
        span_ns = samples[-1][0] - samples[0][0]
        frame_delta = samples[-1][1] - samples[0][1]
        return (
            frame_delta * 1_000_000_000 / span_ns
            if span_ns > 0 and frame_delta >= 0
            else 0.0
        )

    def _prune(
        self,
        samples: deque[tuple[int, int, object | None]],
        now_ns: int,
    ) -> None:
        cutoff = now_ns - self._window_ns
        while samples and samples[0][0] < cutoff:
            samples.popleft()

    def clear(self) -> None:
        self._samples.clear()


class PreviewInbox:
    """Latest-only, generation-aware handoff from reader thread to Tk."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._generation = 0
        self._frames: dict[str, ReceivedFrame] = {}
        self._messages: list[tuple[str, str]] = []
        self._fps = FpsTracker()

    def begin(self) -> int:
        with self._lock:
            self._generation += 1
            self._frames.clear()
            self._messages.clear()
            self._fps.clear()
            return self._generation

    def put_frames(
        self,
        generation: int,
        frames: dict[str, ReceivedFrame],
        received_ns: int,
    ) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._frames.update(frames)
            for topic, frame in frames.items():
                self._fps.record(
                    topic,
                    received_ns,
                    sequence=frame.sequence,
                    session_id=frame.session_id,
                )

    def put_message(self, generation: int, kind: str, message: str) -> None:
        with self._lock:
            if generation == self._generation:
                self._messages.append((kind, message))

    def drain(
        self, generation: int
    ) -> tuple[dict[str, ReceivedFrame], list[tuple[str, str]]]:
        with self._lock:
            if generation != self._generation:
                return {}, []
            frames = self._frames
            messages = self._messages
            self._frames = {}
            self._messages = []
            return frames, messages

    def fps(self, topic: str, now_ns: int) -> float:
        with self._lock:
            return self._fps.fps(topic, now_ns)


def summarize_camera_statuses(
    statuses: dict[str, dict[str, object]],
) -> tuple[str, str] | None:
    """Return the most safety-relevant status text for the GUI."""
    for desired_state, kind in (("fault", "error"), ("reconnecting", "warning")):
        for topic, status in sorted(statuses.items()):
            if status.get("state") != desired_state:
                continue
            camera = topic.rsplit("/", 1)[-1]
            details = [status.get("error_category"), status.get("error")]
            suffix = ": ".join(str(value) for value in details if value)
            text = f"{camera} {desired_state}"
            if suffix:
                text = f"{text}: {suffix}"
            return kind, text
    if all(statuses.get(topic, {}).get("state") == "online" for topic in STATUS_TOPICS):
        return "online", "All cameras online"
    return None


class CameraPreviewApp:
    def __init__(self, root: tk.Tk, initial_endpoint: str) -> None:
        self.root = root
        self.root.title("Piper Camera Preview")
        self.root.minsize(720, 560)
        self.endpoint = tk.StringVar(value=initial_endpoint)
        self.status = tk.StringVar(value="Enter IP:port and click Preview")
        self._stop_event = Event()
        self._reader_thread: Thread | None = None
        self._inbox = PreviewInbox()
        self._generation = self._inbox.begin()
        self._image_labels: dict[str, ttk.Label] = {}
        self._title_labels: dict[str, ttk.Label] = {}
        self._photos: dict[str, tk.PhotoImage] = {}
        self._dimensions: dict[str, tuple[int, int]] = {}
        self._titles = {topic: title for topic, title, _row, _column in TOPIC_LAYOUT}

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(20, self._refresh)

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Camera bridge:").pack(side=tk.LEFT)
        self.endpoint_entry = ttk.Entry(
            controls, textvariable=self.endpoint, width=32
        )
        self.endpoint_entry.pack(side=tk.LEFT, padx=(6, 8))
        self.preview_button = ttk.Button(
            controls, text="Preview", command=self.toggle_preview
        )
        self.preview_button.pack(side=tk.LEFT)
        ttk.Label(controls, textvariable=self.status).pack(
            side=tk.LEFT, padx=12
        )

        grid = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        grid.pack(fill=tk.BOTH, expand=True)
        for column in range(3):
            grid.columnconfigure(column, weight=1)
        for row in range(3):
            grid.rowconfigure(row, weight=1)

        for topic, title, row, column in TOPIC_LAYOUT:
            panel = ttk.LabelFrame(grid, padding=3)
            panel.grid(row=row, column=column, padx=3, pady=3, sticky="nsew")
            title_label = ttk.Label(panel, text=f"{title} · waiting", anchor=tk.CENTER)
            title_label.pack(fill=tk.X)
            image_label = ttk.Label(
                panel,
                text="No frame",
                anchor=tk.CENTER,
                width=42,
            )
            image_label.pack(fill=tk.BOTH, expand=True)
            self._title_labels[topic] = title_label
            self._image_labels[topic] = image_label

    def toggle_preview(self) -> None:
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self.stop_preview()
        else:
            self.start_preview()

    def start_preview(self) -> None:
        try:
            endpoint = normalize_endpoint(self.endpoint.get())
        except ValueError as exc:
            self.status.set(str(exc))
            return
        if not self.stop_preview(update_status=False):
            self.status.set("Previous preview is still stopping")
            return
        self._generation = self._inbox.begin()
        self._stop_event = Event()
        self.preview_button.configure(text="Stop")
        self.endpoint_entry.configure(state=tk.DISABLED)
        self.status.set(f"Connecting to {endpoint} …")
        self._reader_thread = Thread(
            target=self._read_loop,
            args=(endpoint, self._stop_event, self._generation),
            name="camera-preview-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def stop_preview(self, *, update_status: bool = True) -> bool:
        self._generation = self._inbox.begin()
        self._stop_event.set()
        thread = self._reader_thread
        if thread is not None:
            thread.join(timeout=0.3)
            if thread.is_alive():
                return False
        self._reader_thread = None
        self.preview_button.configure(text="Preview")
        self.endpoint_entry.configure(state=tk.NORMAL)
        if update_status:
            self.status.set("Preview stopped")
        return True

    def _read_loop(
        self, endpoint: str, stop_event: Event, generation: int
    ) -> None:
        subscriber = CameraSubscriber(
            endpoint, topics=(*IMAGE_TOPICS, *STATUS_TOPICS)
        )
        reported_connected = False
        previous_statuses: dict[str, dict[str, object]] = {}
        try:
            subscriber.open()
            while not stop_event.is_set():
                frames = subscriber.recv_latest(100)
                received_ns = time.monotonic_ns()
                if frames:
                    self._inbox.put_frames(generation, frames, received_ns)
                statuses = {
                    topic: dict(status)
                    for topic, status in subscriber.statuses.items()
                }
                if statuses != previous_statuses:
                    previous_statuses = statuses
                    summary = summarize_camera_statuses(statuses)
                    if summary is not None:
                        self._inbox.put_message(generation, *summary)
                if frames and not reported_connected:
                    self._inbox.put_message(generation, "connected", endpoint)
                    reported_connected = True
        except Exception as exc:
            if not stop_event.is_set():
                self._inbox.put_message(
                    generation, "error", f"{type(exc).__name__}: {exc}"
                )
        finally:
            subscriber.close()

    def _refresh(self) -> None:
        frames, messages = self._inbox.drain(self._generation)
        for kind, message in messages:
            if kind == "error":
                self.status.set(f"Preview error: {message}")
                self.stop_preview(update_status=False)
            elif kind == "warning":
                self.status.set(f"Camera warning: {message}")
            elif kind == "online":
                self.status.set(message)
            else:
                self.status.set(f"Receiving from {message}")
        now_ns = time.monotonic_ns()
        for topic, frame in frames.items():
            try:
                rgb = frame_to_rgb(frame)
                preview = cv2.resize(
                    rgb,
                    (PREVIEW_WIDTH, PREVIEW_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
                photo = tk.PhotoImage(
                    data=(
                        f"P6\n{PREVIEW_WIDTH} {PREVIEW_HEIGHT}\n255\n".encode()
                        + preview.tobytes()
                    ),
                    format="PPM",
                )
            except Exception as exc:
                self.status.set(f"Decode error on {topic}: {exc}")
                continue
            self._photos[topic] = photo
            self._dimensions[topic] = (frame.width, frame.height)
            self._image_labels[topic].configure(image=photo, text="")
        for topic, (width, height) in self._dimensions.items():
            self._title_labels[topic].configure(
                text=(
                    f"{self._titles[topic]} · "
                    f"{self._inbox.fps(topic, now_ns):.1f} FPS · "
                    f"{width}×{height}"
                )
            )
        self.root.after(20, self._refresh)

    def close(self) -> None:
        self.stop_preview(update_status=False)
        self.root.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview camera-bridge streams")
    parser.add_argument(
        "--endpoint",
        default="127.0.0.1:15555",
        help="camera-bridge host:port shown in the GUI",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = tk.Tk()
    CameraPreviewApp(root, args.endpoint)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
