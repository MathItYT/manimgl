import manimlib
import os
from pyglet.window import key
import string
import threading

imported_transcriber: bool = False
imported_llm_scene_controller: bool = False

try:
    from manimlib.extras.transcription import ElevenLabsRealtimeTranscriber, bind_transcriber_callback, bind_transcriber_to_text
    imported_transcriber = True
except ImportError:
    pass


try:
    from manimlib.extras.llm import LLMSceneController
    imported_llm_scene_controller = True
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
    def construct(self, callback=None) -> None:
        if not imported_transcriber:
            raise RuntimeError(
                "ElevenLabsRealtimeTranscriber is required for TranscriptionExample. Install with: pip install \"manimgl[transcription] @ git+https://github.com/MathItYT/manimgl\""
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


class LLMExample(Example):
    def construct(self) -> None:
        if not imported_llm_scene_controller:
            raise RuntimeError(
                "LLMSceneController is required for LLMExample. Install with: pip install \"manimgl[llm] @ git+https://github.com/MathItYT/manimgl\""
            )
        self.prompt_mode: bool = False
        self.llm_controller = LLMSceneController(
            self,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            model="openai/gpt-oss-120b",
        )
        self.prompt = manimlib.Text("").add(manimlib.Dot()).to_edge(manimlib.DOWN, buff=0.5)
        self.add(self.prompt)
        super().construct()
    
    def on_key_press(self, symbol, modifiers):
        if symbol == key.P and modifiers & key.MOD_CTRL:
            self.prompt_mode = not self.prompt_mode
            if self.prompt_mode:
                self.prompt.become(manimlib.Text("Escribe un prompt para el LLM y presiona Enter", font_size=24).to_edge(manimlib.DOWN, buff=0.5))
                self.prompt.text = None
            else:
                self.prompt.become(manimlib.Text("").add(manimlib.Dot()).to_edge(manimlib.DOWN, buff=0.5))
                self.prompt.text = None
        elif symbol == key.ENTER and self.prompt_mode:
            prompt: manimlib.Text = self.prompt
            prompt_text = prompt.text
            prompt.become(manimlib.Text("").add(manimlib.Dot()).to_edge(manimlib.DOWN, buff=0.5))
            self.prompt.text = None
            self.prompt_mode = False
            threading.Thread(target=lambda: self.llm_controller.run_prompt(prompt_text, reasoning_effort="high")).start()
        elif not self.prompt_mode:
            super().on_key_press(symbol, modifiers)
        else:
            if symbol == key.BACKSPACE:
                prompt: manimlib.Text = self.prompt
                prompt_text = prompt.text
                if prompt_text:
                    prompt.become(manimlib.Text(prompt_text[:-1], font_size=24).to_edge(manimlib.DOWN, buff=0.5))
                    prompt.text = prompt_text[:-1]
            else:
                char = chr(symbol)
                if char not in string.printable:
                    return
                is_cap = modifiers & key.MOD_SHIFT
                if char.isalpha():
                    char = char.upper() if is_cap else char.lower()
                prompt: manimlib.Text = self.prompt
                prompt_text = prompt.text or ""
                prompt.become(manimlib.Text(prompt_text + char, font_size=24).to_edge(manimlib.DOWN, buff=0.5))
                prompt.text = prompt_text + char


class TranscriptionLLMExample(Example):
    def construct(self) -> None:
        if not imported_transcriber:
            raise RuntimeError(
                "ElevenLabsRealtimeTranscriber is required for TranscriptionLLMExample. Install with: pip install \"manimgl[transcription] @ git+https://github.com/MathItYT/manimgl\""
            )
        if not imported_llm_scene_controller:
            raise RuntimeError(
                "LLMSceneController is required for TranscriptionLLMExample. Install with: pip install \"manimgl[llm] @ git+https://github.com/MathItYT/manimgl\""
            )

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("Set ELEVENLABS_API_KEY environment variable before running TranscriptionLLMExample")

        transcriber = ElevenLabsRealtimeTranscriber(
            api_key=api_key,
            sample_rate=16000,
            audio_format="pcm_16000",
            commit_strategy="vad",
            language_code="es",
            max_audio_queue_chunks=24,
            chunks_per_enqueue=2,
        )

        transcript_text = manimlib.Text("Habla para transcribir...", font_size=24).to_edge(manimlib.DOWN, buff=0.5)
        transcript_text.fix_in_frame()
        bind_transcriber_to_text(
            self,
            transcript_text,
            transcriber,
            update_fps=5,
            partial_update_fps=2,
            render_partial=False,
            build_text_off_main_thread=True,
            font_size=24,
        )

        llm_status = manimlib.Text("LLM listo", font_size=20).to_edge(manimlib.DOWN, buff=1.2)
        llm_status.fix_in_frame()
        self.add(transcript_text, llm_status)

        self.llm_controller = LLMSceneController(
            self,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            model="openai/gpt-oss-120b",
        )
        self.llm_status = llm_status
        self.prompt_on_next_commit = False

        llm_lock = threading.Lock()
        llm_busy = False
        llm_ready_pending = False
        llm_error_pending = False

        def _flush_llm_status(_dt: float) -> None:
            nonlocal llm_ready_pending, llm_error_pending
            should_set_ready = False
            should_set_error = False
            with llm_lock:
                if llm_error_pending:
                    llm_error_pending = False
                    should_set_error = True
                if llm_ready_pending:
                    llm_ready_pending = False
                    should_set_ready = True
            if should_set_error:
                llm_status.become(manimlib.Text("Error en la interaccion previa", font_size=20).to_edge(manimlib.DOWN, buff=1.2))
                llm_status.fix_in_frame()
                return
            if should_set_ready:
                llm_status.become(manimlib.Text("LLM listo", font_size=20).to_edge(manimlib.DOWN, buff=1.2))
                llm_status.fix_in_frame()

        self.add_updater(_flush_llm_status)

        def _run_llm_from_transcript(prompt: str) -> None:
            nonlocal llm_busy, llm_ready_pending, llm_error_pending
            had_error = False
            try:
                self.llm_controller.run_prompt(prompt, reasoning_effort="high")
            except Exception as exc:
                had_error = True
                print(f"LLM error: {exc}")
            finally:
                with llm_lock:
                    llm_busy = False
                    if had_error:
                        llm_error_pending = True
                    else:
                        llm_ready_pending = True

        def _on_transcript_event(event_kind: str, text: str) -> None:
            nonlocal llm_busy
            if event_kind != "committed":
                return
            if not self.prompt_on_next_commit:
                return
            self.prompt_on_next_commit = False

            prompt = text.strip()
            if not prompt:
                llm_status.become(manimlib.Text("LLM listo", font_size=20).to_edge(manimlib.DOWN, buff=1.2))
                llm_status.fix_in_frame()
                return

            with llm_lock:
                if llm_busy:
                    llm_status.become(manimlib.Text("LLM ocupado", font_size=20).to_edge(manimlib.DOWN, buff=1.2))
                    llm_status.fix_in_frame()
                    return
                llm_busy = True

            llm_status.become(manimlib.Text(f"LLM ejecutando: {prompt[:40]}", font_size=20).to_edge(manimlib.DOWN, buff=1.2))
            llm_status.fix_in_frame()
            threading.Thread(
                target=_run_llm_from_transcript,
                args=(prompt,),
                daemon=True,
            ).start()

        def _arm_prompt_mode() -> None:
            nonlocal llm_ready_pending, llm_error_pending
            self.prompt_on_next_commit = True
            with llm_lock:
                llm_ready_pending = False
                llm_error_pending = False
            self.llm_status.become(
                manimlib.Text("Ctrl+P activo: esperando transcripcion committed", font_size=20).to_edge(manimlib.DOWN, buff=1.2)
            )
            self.llm_status.fix_in_frame()

        self._arm_prompt_mode = _arm_prompt_mode

        bind_transcriber_callback(
            self,
            transcriber,
            _on_transcript_event,
            event_kinds=("committed",),
            update_fps=10,
        )

        super().construct(
            callback=transcriber.on_mic_chunk,
            mic_rate=16000,
            mic_channels=1,
            mic_chunk=1024,
        )

    def on_key_press(self, symbol, modifiers):
        if symbol == key.P and modifiers & key.MOD_CTRL:
            arm_prompt_mode = getattr(self, "_arm_prompt_mode", None)
            if callable(arm_prompt_mode):
                arm_prompt_mode()
            return
        super().on_key_press(symbol, modifiers)
