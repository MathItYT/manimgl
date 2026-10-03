from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import BinaryIO, Iterable
import os
import struct
import sys

import numpy as np
import wgpu


class FrameSink(ABC):
    """Destination for completed RGBA frames."""

    @abstractmethod
    def write(self, frame: memoryview, *, width: int, height: int, timestamp: float) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class StreamSink(FrameSink):
    """Adapt any binary file-like object into a frame sink."""

    def __init__(self, stream: BinaryIO):
        self.stream = stream

    def write(self, frame, *, width, height, timestamp):
        self.stream.write(frame)

    def flush(self):
        self.stream.flush()

    def close(self):
        # The owner of the stream is responsible for closing it.
        pass


class FileSink(FrameSink):
    """Write raw RGBA frames to a file."""

    def __init__(self, path: str | os.PathLike):
        self.file = open(path, "wb")

    def write(self, frame, *, width, height, timestamp):
        self.file.write(frame)

    def flush(self):
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()


class StdoutSink(FrameSink):
    """Write raw RGBA frames to a binary stream, stdout by default."""

    def __init__(self, stream: BinaryIO | None = None):
        self.stream = stream or sys.stdout.buffer

    def write(self, frame, *, width, height, timestamp):
        self.stream.write(frame)

    def flush(self):
        self.stream.flush()


@dataclass(frozen=True)
class SharedTextureHeader:
    width: int
    height: int
    stride: int
    sequence: int
    timestamp: float


class SharedMemoryTextureSink(FrameSink):
    """Publish RGBA frames into a caller-owned shared buffer.

    The buffer layout is: header + tightly packed RGBA pixels.
    """

    HEADER_FORMAT = "<IIIIQd"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    def __init__(self, width: int, height: int, *, buffer=None, name: str | None = None):
        self.width = int(width)
        self.height = int(height)
        self.stride = self.width * 4
        self.sequence = 0
        size = self.HEADER_SIZE + self.stride * self.height

        if buffer is not None:
            self._shm = None
            self.buffer = memoryview(buffer)
            if self.buffer.nbytes < size:
                raise ValueError(f"Shared buffer requires at least {size} bytes")
        else:
            from multiprocessing import shared_memory
            self._shm = shared_memory.SharedMemory(
                create=name is None,
                size=size if name is None else 0,
                name=name,
            )
            self.buffer = self._shm.buf

    @property
    def pixels(self):
        return self.buffer[self.HEADER_SIZE:self.HEADER_SIZE + self.stride * self.height]

    def write(self, frame, *, width, height, timestamp):
        if (width, height) != (self.width, self.height):
            raise ValueError("Frame dimensions changed while the shared sink is active")
        self.sequence += 1
        struct.pack_into(
            self.HEADER_FORMAT, self.buffer, 0,
            width, height, self.stride, 0, self.sequence, timestamp,
        )
        self.pixels[:] = frame

    def close(self):
        if self._shm is not None:
            self._shm.close()


class FrameStream:
    """Fan-out GPU readback with dynamic sinks."""

    def __init__(self, camera, sinks: Iterable[FrameSink] | None = None, behind: int = 1):
        self.camera = camera
        self.sinks = list(sinks or ())
        self.behind = max(0, int(behind))
        self.device = camera.gpu.device
        self.queue = camera.gpu.queue
        self.width, self.height = camera.get_pixel_shape()
        self.row = 4 * self.width
        self.padded_row = self.row + (-self.row % 256)
        self.buffers = [
            self.device.create_buffer(
                size=self.padded_row * self.height,
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
            )
            for _ in range(self.behind + 1)
        ]
        self.waiting = []
        self.asked = 0

    def add_sink(self, sink: FrameSink):
        if sink not in self.sinks:
            self.sinks.append(sink)
        return sink

    def remove_sink(self, sink: FrameSink):
        if sink in self.sinks:
            self.sinks.remove(sink)
        return sink

    def clear_sinks(self):
        self.sinks.clear()

    def set_sinks(self, *sinks: FrameSink):
        self.sinks[:] = sinks

    def send(self, timestamp: float = 0.0) -> None:
        if not self.sinks:
            return
        buffer = self.buffers[self.asked % len(self.buffers)]
        self.asked += 1
        encoder = self.device.create_command_encoder()
        encoder.copy_texture_to_buffer(
            {"texture": self.camera.color_texture, "mip_level": 0, "origin": (0, 0, 0)},
            {
                "buffer": buffer, "offset": 0,
                "bytes_per_row": self.padded_row,
                "rows_per_image": self.height,
            },
            (self.width, self.height, 1),
        )
        self.queue.submit([encoder.finish()])
        self.waiting.append((buffer, buffer.map_async("READ", 0, buffer.size), timestamp))
        if len(self.waiting) > self.behind:
            self.write_oldest()

    def write_oldest(self) -> None:
        buffer, promise, timestamp = self.waiting.pop(0)
        promise.sync_wait()
        frame = buffer.read_mapped(copy=False)
        if self.padded_row > self.row:
            rows = np.frombuffer(frame, np.uint8).reshape((self.height, self.padded_row))
            frame = memoryview(rows[:, :self.row]).cast("B")
        for sink in tuple(self.sinks):
            sink.write(frame, width=self.width, height=self.height, timestamp=timestamp)
        buffer.unmap()

    def drain(self) -> None:
        while self.waiting:
            self.write_oldest()
        for sink in tuple(self.sinks):
            sink.flush()

    def close(self) -> None:
        self.drain()
        for sink in tuple(self.sinks):
            sink.close()
        self.sinks.clear()
