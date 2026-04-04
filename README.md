# ManimGL Presentation Engine
ManimGL Presentation Engine is a fork of original 3Blue1Brown's Manim aimed to be a live presentation engine with realtime performance.

## Installation

```bash
pip install git+https://github.com/MathItYT/manimgl.git
```

If you want a transcription utility powered by ElevenLabs, also install with the `transcription` extra:

```bash
pip install "manimgl[transcription] @ git+https://github.com/MathItYT/manimgl"`
```

If you want to control scenes with an LLM through OpenAI Chat Completions (with strict structured outputs), install with the `llm` extra:

```bash
pip install "manimgl[llm] @ git+https://github.com/MathItYT/manimgl"
```

If you want realtime hand tracking and gesture-driven interactions over `VideoMobject`, install with the `vision` extra:

```bash
pip install "manimgl[vision] @ git+https://github.com/MathItYT/manimgl"
```

If you want to stream the current ManimGL output into a virtual camera, install with the `virtual_camera` extra:

```bash
pip install "manimgl[virtual_camera] @ git+https://github.com/MathItYT/manimgl"
```

If you want to render YouTube live chat messages inside your scene, install with the `youtube_chat` extra:

```bash
pip install "manimgl[youtube_chat] @ git+https://github.com/MathItYT/manimgl"
```

## How to run?
Create an `manimlib.InteractiveScene` subclass in a `main.py` (or any other filename else) module.

For example:

```python
import manimlib
import os

imported_transcriber: bool = False

try:
    from manimlib.extras.transcription import ElevenLabsRealtimeTranscriber, bind_transcriber_to_text
    imported_transcriber = True
except ImportError:
    pass


class Example(manimlib.InteractiveScene):
    def construct(
        self,
        callback=None,
        mic_rate=48000,
        mic_channels=2,
        mic_chunk=4096,
    ) -> None:
        self.start_mic_recording(
            rate=mic_rate,
            channels=mic_channels,
            chunk=mic_chunk,
            callback=callback,
        )  # Iniciar grabación de micrófono
        video_mob = manimlib.VideoMobject.from_video(
            0, height=3, flip_horizontal=True
        )  # Cámara por defecto y flip horizontal para efecto espejo
        circ = manimlib.Circle(radius=1).move_to(video_mob)
        circ.set_fill(color=manimlib.WHITE, opacity=1.0)
        circ.set_stroke(width=0)
        circ.to_corner(manimlib.UR)
        circ.fix_in_frame()
        video_mob.move_to(circ)
        video_mob.play()
        video_mob.fix_in_frame()
        self.capture = manimlib.MaskMobject(
            self, video_mob, circ
        )  # Recortar el video con la máscara circular
        title = manimlib.TexText(
            "¿Qué son los números reales?", font_size=96
        )
        subtitle = manimlib.TexText(
            "Explicación completa", font_size=60
        )
        manimlib.Group(title, subtitle).arrange(manimlib.DOWN)
        self.save_state()
        self.add(self.capture, video_mob, circ, title, subtitle)
        self.play(
            manimlib.GrowFromCenter(circ),
        )
        self.save_state()
        self.play(
            manimlib.FadeOut(manimlib.Group(title, subtitle)),
            hold=False,
        )

    # Por defecto play no hace hold, así que lo sobreescribimos para que sí lo haga
    # como si fuera una presentación.
    def play(
        self,
        *proto_animations,
        run_time=None,
        rate_func=None,
        lag_ratio=None,
        hold=True,
    ) -> None:
        super().play(
            *proto_animations,
            run_time=run_time,
            rate_func=rate_func,
            lag_ratio=lag_ratio,
        )
        if hold:
            self.hold_on_wait = True
            self.hold_loop()


class TranscriptionExample(Example):
    def construct(self) -> None:
        if not imported_transcriber:
            raise RuntimeError(
                "ElevenLabsRealtimeTranscriber is required for TranscriptionExample. Install with: pip install \"manimgl @ git+https://github.com/MathItYT/manimgl[transcription]\""
            )
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("Set ELEVENLABS_API_KEY environment variable before running TranscriptionExample")

        transcriber = ElevenLabsRealtimeTranscriber(
            api_key=api_key,
            sample_rate=16000,
            audio_format="pcm_16000",
            commit_strategy="vad",
            language_code="es",
            max_audio_queue_chunks=24,
            chunks_per_enqueue=2,
        )
        text = manimlib.Text("Habla para transcribir...", font_size=24).to_edge(manimlib.DOWN, buff=0.5)
        text.fix_in_frame()
        bind_transcriber_to_text(
            self,
            text,
            transcriber,
            update_fps=5,
            partial_update_fps=2,
            render_partial=False,
            build_text_off_main_thread=True,
            font_size=24,
        )
        self.add(text)
        super().construct(
            callback=transcriber.on_mic_chunk,
            mic_rate=16000,
            mic_channels=1,
            mic_chunk=1024,
        )
```

Then run in your terminal:

```bash
manimgl main.py Example
```

Also you can run the `TranscriptionExample` scene to see the ElevenLabs transcription utility in action (remember to set your API key in the environment variable `ELEVENLABS_API_KEY`):

```bash
manimgl main.py TranscriptionExample -o
```

Note that `-o` flag is required to render the video as a file, since ManimGL Presentation Engine is designed for live presentations and won't render to file by default, but it can render to file seamlessly with the `-o` flag.

## Transcription Extra

The transcription extra adds realtime speech-to-text utilities powered by ElevenLabs.

### Install

```bash
pip install "manimgl[transcription] @ git+https://github.com/MathItYT/manimgl"
```

### Environment variable

Set your ElevenLabs API key before running scenes:

```bash
# PowerShell
$env:ELEVENLABS_API_KEY="your_api_key"
```

### What it provides

- `ElevenLabsRealtimeTranscriber`: receives audio chunks and streams transcription.
- `bind_transcriber_to_text`: links transcriber output to a Manim text mobject.

Import path:

```python
from manimlib.extras.transcription import ElevenLabsRealtimeTranscriber, bind_transcriber_to_text
```

### Minimal example

```python
import os
import manimlib

from manimlib.extras.transcription import ElevenLabsRealtimeTranscriber, bind_transcriber_to_text


class TranscriptionMinimal(manimlib.InteractiveScene):
    def construct(self) -> None:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("Set ELEVENLABS_API_KEY before running this scene")

        transcriber = ElevenLabsRealtimeTranscriber(
            api_key=api_key,
            sample_rate=16000,
            audio_format="pcm_16000",
            commit_strategy="vad",
            language_code="es",
        )

        subtitle = manimlib.Text("Habla para transcribir...", font_size=24)
        subtitle.to_edge(manimlib.DOWN, buff=0.5)
        subtitle.fix_in_frame()

        bind_transcriber_to_text(
            self,
            subtitle,
            transcriber,
            update_fps=5,
            partial_update_fps=2,
            render_partial=False,
            build_text_off_main_thread=True,
            font_size=24,
        )

        self.add(subtitle)
        self.start_mic_recording(
            rate=16000,
            channels=1,
            chunk=1024,
            callback=transcriber.on_mic_chunk,
        )

        self.wait(10)
```

Run the scene:

```bash
manimgl main.py TranscriptionMinimal -o
```

## Vision Extra (Hand Tracking)

The `vision` extra adds realtime hand tracking utilities running in a background thread.

### Install

```bash
pip install "manimgl[vision] @ git+https://github.com/MathItYT/manimgl"
```

### What it provides

- `HandMotionTracker`: processes webcam frames in a separate thread and emits motion state.
- `HandMesh`: `VMobject` that renders a 2D hand mesh from detected landmarks.
- `bind_hand_tracker_to_video`: taps frames from an existing `VideoMobject` into the tracker.
- `bind_hand_position_to_mobject`: moves a mobject based on normalized hand position.
- `bind_hand_mesh_to_tracker`: updates `HandMesh` automatically from tracker state.

Import path:

```python
from manimlib.extras.vision import (
    HandMesh,
    HandMotionTracker,
    bind_hand_mesh_to_tracker,
    bind_hand_position_to_mobject,
    bind_hand_tracker_to_video,
)
```

### Minimal example

```python
import manimlib

from manimlib.extras.vision import (
    HandMesh,
    HandMotionTracker,
    bind_hand_mesh_to_tracker,
    bind_hand_position_to_mobject,
    bind_hand_tracker_to_video,
)


class HandTrackingMinimal(manimlib.InteractiveScene):
    def construct(self) -> None:
        video = manimlib.VideoMobject.from_video(0, height=5, flip_horizontal=True)
        video.play()

        tracker = HandMotionTracker()
        bind_hand_tracker_to_video(video, tracker, enqueue_every_n_frames=2)

        circle = manimlib.Circle(radius=0.2)
        circle.set_fill(color=manimlib.YELLOW, opacity=0.85)
        circle.set_stroke(color=manimlib.WHITE, width=3)

        mesh = HandMesh(reference_mobject=video, stroke_color=manimlib.BLUE, stroke_width=3)

        bind_hand_position_to_mobject(self, circle, tracker, reference_mobject=video, update_fps=30)
        bind_hand_mesh_to_tracker(self, mesh, tracker, update_fps=30)

        self.add(video, mesh, circle)
        self.wait(10)
```

Run the scene:

```bash
manimgl main.py HandTrackingMinimal -o
```

## Virtual Camera Extra

The `virtual_camera` extra sends rendered frames to a system virtual camera.

### Install

```bash
pip install "manimgl[virtual_camera] @ git+https://github.com/MathItYT/manimgl"
```

### What it provides

- `VirtualCameraSink`: pushes the current rendered frame into a virtual camera.
- `bind_scene_to_virtual_camera`: attaches the sink to a `Scene`.
- `unbind_scene_from_virtual_camera`: detaches and closes the sink.

Import path:

```python
from manimlib.extras.virtual_camera import VirtualCameraSink, bind_scene_to_virtual_camera
```

### Minimal example

```python
import manimlib

from manimlib.extras.virtual_camera import bind_scene_to_virtual_camera


class VirtualCameraMinimal(manimlib.InteractiveScene):
    def setup(self) -> None:
        self.virtual_camera_sink = bind_scene_to_virtual_camera(self, fps=30)

    def construct(self) -> None:
        title = manimlib.Text("ManimGL Virtual Camera", font_size=72)
        self.add(title)
        self.wait(5)
```

Run the scene:

```bash
manimgl main.py VirtualCameraMinimal
```

Then in your video app (OBS, Zoom, Teams, Meet, etc.) choose the virtual camera exposed by `pyvirtualcam` as the camera source.

## YouTube Chat Extra

The `youtube_chat` extra streams YouTube live chat messages and exposes helpers to render them in-scene.

### Install

```bash
pip install "manimgl[youtube_chat] @ git+https://github.com/MathItYT/manimgl"
```

### Environment variable

Set the target live video id (or full YouTube live URL) before running scenes:

```bash
# PowerShell
$env:YOUTUBE_LIVE_VIDEO_ID="your_live_video_id_or_url"
```

### What it provides

- `YouTubeLiveChatClient`: background client that receives live chat messages.
- `bind_youtube_chat_callback`: dispatches incoming messages to a callback.
- `bind_youtube_chat_to_feed`: renders a rolling chat feed with circular masked avatars and bold author names.
- `bind_youtube_chat_to_text`: simple text-only binding for lightweight use cases.

Import path:

```python
from manimlib.extras.youtube_chat import YouTubeLiveChatClient, bind_youtube_chat_to_feed
```

### Minimal example

```python
import os
import manimlib

from manimlib.extras.youtube_chat import YouTubeLiveChatClient, bind_youtube_chat_to_feed


class YouTubeChatMinimal(manimlib.InteractiveScene):
    def construct(self) -> None:
        video_id = os.getenv("YOUTUBE_LIVE_VIDEO_ID")
        if not video_id:
            raise RuntimeError("Set YOUTUBE_LIVE_VIDEO_ID before running this scene")

        chat_client = YouTubeLiveChatClient(video_id)

        chat_anchor = manimlib.Dot(radius=0.001)
        chat_anchor.to_edge(manimlib.DOWN, buff=1.0)
        chat_anchor.fix_in_frame()

        bind_youtube_chat_to_feed(
            self,
            chat_anchor,
            chat_client,
            max_messages=8,
            update_fps=8,
            avatar_height=0.55,
            text_font_size=20,
        )

        self.add(chat_anchor)
        self.wait(60)
```

Run the scene:

```bash
manimgl main.py YouTubeChatMinimal
```

## LLM Scene Control (Safe Structured Outputs)

The `manimlib.extras.llm` module allows an LLM to operate a scene with strict JSON actions only.

- Uses OpenAI Chat Completions with `response_format.json_schema.strict = true`
- Supports custom `base_url` and custom `api_key`
- Avoids `exec` by executing a declarative operation plan
- Builds JSON schema from a symbol whitelist and per-module blacklists

Example:

```python
import os
import manimlib

from manimlib.extras.llm import LLMSceneController


class LLMExample(manimlib.InteractiveScene):
    def construct(self) -> None:
        controller = LLMSceneController(
            scene=self,
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://your-openai-compatible-endpoint/v1",
            model="gpt-4.1-mini",
            whitelist_modules=["manimlib"],
        )

        controller.run_prompt(
            "Create a circle and a square, animate both, add a ValueTracker updater, then wait 1 second."
        )
```
