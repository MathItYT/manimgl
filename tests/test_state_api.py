import io

from manimlib.renderer.frame_stream import SharedMemoryTextureSink, StdoutSink
from manimlib.scene.scene_graph import SceneGraphSerializer, describe_callable


def test_stdout_sink_writes_rgba_bytes():
    stream = io.BytesIO()
    sink = StdoutSink(stream)
    payload = memoryview(b"\x01\x02\x03\x04")
    sink.write(payload, width=1, height=1, timestamp=0.5)
    assert stream.getvalue() == payload.tobytes()


def test_shared_memory_texture_sink_layout():
    buffer = bytearray(SharedMemoryTextureSink.HEADER_SIZE + 8)
    sink = SharedMemoryTextureSink(1, 2, buffer=buffer)
    sink.write(memoryview(b"12345678"), width=1, height=2, timestamp=1.25)
    assert sink.pixels.tobytes() == b"12345678"
    assert sink.sequence == 1


def test_callable_description():
    def rate(t):
        return t

    info = describe_callable(rate)
    assert info["serializable"] is True
    assert info["qualname"].endswith("rate")


def test_scene_graph_serializes_explicit_object_metadata():
    class FakeObject:
        submobjects = []
        _serialization_name = "circle"
        _serialization_constructor = "Circle"
        _serialization_parameters = {"radius": 2}

    class FakeEvent:
        name = "grow"
        kind = "animation"
        t_start = 0.0
        t_end = 1.0
        animations = ()
        metadata = None

    class FakeScene:
        time = 1.0
        mobjects = [FakeObject()]
        timeline = [FakeEvent()]
        def get_top_level_mobjects(self):
            return self.mobjects

    data = SceneGraphSerializer(FakeScene()).to_dict()
    assert data["objects"][0]["name"] == "circle"
    assert data["objects"][0]["parameters"]["radius"] == 2
    assert data["animations"][0]["t_start"] == 0.0
    assert data["animations"][0]["t_end"] == 1.0
