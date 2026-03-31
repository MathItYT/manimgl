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
