from typing import Optional

import manimlib
import os
from pyglet.window import key
import string
import threading
from openai import OpenAI
import json
import random
import heapq

from manimlib.extras.llm.scene_agent import DynNum, VMobjectParams

imported_transcriber: bool = False
imported_llm_scene_controller: bool = False
imported_hand_tracking: bool = False
imported_virtual_camera: bool = False
imported_youtube_chat: bool = False

try:
    from manimlib.extras.transcription import (
        ElevenLabsRealtimeTranscriber,
        bind_transcriber_callback,
        bind_transcriber_to_text,
    )

    imported_transcriber = True
except ImportError:
    pass


try:
    from manimlib.extras.llm import (
        LLMSceneController,
    )

    imported_llm_scene_controller = True
except ImportError:
    pass


try:
    from manimlib.extras.vision import (
        HandMesh,
        HandMotionState,
        HandMotionTracker,
        bind_hand_gesture_callback,
        bind_hand_mesh_to_tracker,
        bind_hand_position_to_mobject,
        bind_hand_tracker_to_video,
        unbind_hand_tracker_from_video,
    )

    imported_hand_tracking = True
except ImportError:
    pass


try:
    from manimlib.extras.virtual_camera import (
        bind_scene_to_virtual_camera,
    )

    imported_virtual_camera = True
except ImportError:
    pass


try:
    from manimlib.extras.youtube_chat import (
        YouTubeLiveChatClient,
        bind_youtube_chat_to_feed,
    )

    imported_youtube_chat = True
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
        self.title = title
        self.subtitle = subtitle

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
                'ElevenLabsRealtimeTranscriber is required for TranscriptionExample. Install with: pip install "manimgl[transcription] @ git+https://github.com/MathItYT/manimgl"'
            )
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set ELEVENLABS_API_KEY environment variable before running TranscriptionExample"
            )

        transcriber = ElevenLabsRealtimeTranscriber(
            api_key=api_key,
            sample_rate=16000,
            audio_format="pcm_16000",
            commit_strategy="vad",
            language_code="es",
            max_audio_queue_chunks=24,
            chunks_per_enqueue=2,
        )
        text = manimlib.Text(
            "Habla para transcribir...", font_size=24
        ).to_edge(manimlib.DOWN, buff=0.5)
        text.fix_in_frame()
        bind_transcriber_to_text(
            self,
            text,
            transcriber,
            update_fps=5,
            partial_update_fps=2,
            render_partial=True,
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

class AStarAnimation(manimlib.Animation):
    """
    Animación nativa optimizada. En lugar de crear 100 ApplyMethods individuales,
    esta clase repinta la cuadrícula progresivamente basándose en el 'alpha' del frame.
    """
    def __init__(self, grid, visited_order, path, **kwargs):
        self.grid = grid
        self.visited_order = visited_order
        self.path = path
        self.total_explored = len(visited_order)
        self.total_path = len(path)
        super().__init__(grid, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        # Asignamos el primer 70% del tiempo de la animación a la expansión
        exp_alpha = min(alpha / 0.7, 1.0)
        target_exp = int(exp_alpha * self.total_explored)
        
        for i in range(target_exp):
            r, c = self.visited_order[i]
            if (r, c) != self.grid.start_coord and (r, c) != self.grid.end_coord:
                self.grid.get_cell(r, c).set_fill(manimlib.TEAL, opacity=0.3)

        # Asignamos el último 30% del tiempo a trazar el camino ganador
        if alpha >= 0.7:
            path_alpha = min((alpha - 0.7) / 0.3, 1.0)
            target_path = int(path_alpha * self.total_path)
            for i in range(target_path):
                r, c = self.path[i]
                if (r, c) != self.grid.start_coord and (r, c) != self.grid.end_coord:
                    self.grid.get_cell(r, c).set_fill(manimlib.BLUE, opacity=0.8)


class AStarGrid(manimlib.VGroup):
    def __init__(
        self,
        rows: int = 10, 
        cols: int = 15, 
        obstacle_density: float = 0.25, 
        cell_size: float = 0.5, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rows = int(rows)
        self.cols = int(cols)
        self.obstacle_density = float(obstacle_density)
        self.cell_size = float(cell_size)
        
        self.cells = []
        self.grid_state = [] 
        
        self._setup_grid()

    def _setup_grid(self):
        self.start_coord = (random.randint(0, self.rows - 1), random.randint(0, self.cols - 1))
        self.end_coord = (random.randint(0, self.rows - 1), random.randint(0, self.cols - 1))
        while self.end_coord == self.start_coord:
            self.end_coord = (random.randint(0, self.rows - 1), random.randint(0, self.cols - 1))

        for r in range(self.rows):
            cell_row = []
            state_row = []
            for c in range(self.cols):
                sq = manimlib.Square(side_length=self.cell_size)
                
                x = (c - self.cols / 2 + 0.5) * self.cell_size
                y = (self.rows / 2 - r - 0.5) * self.cell_size
                sq.move_to(manimlib.RIGHT * x + manimlib.UP * y)

                is_start = (r, c) == self.start_coord
                is_end = (r, c) == self.end_coord
                
                if is_start:
                    sq.set_fill(manimlib.GREEN, opacity=0.8)
                    state = 2
                elif is_end:
                    sq.set_fill(manimlib.RED, opacity=0.8)
                    state = 3
                else:
                    if random.random() < self.obstacle_density:
                        sq.set_fill(manimlib.GREY, opacity=0.8)
                        state = 1
                    else:
                        sq.set_fill(manimlib.BLACK, opacity=0.0)
                        state = 0
                
                sq.set_stroke(manimlib.WHITE, width=1.0)
                
                self.add(sq)
                cell_row.append(sq)
                state_row.append(state)
            
            self.cells.append(cell_row)
            self.grid_state.append(state_row)

    def get_cell(self, r: int, c: int) -> manimlib.Square:
        return self.cells[r][c]

    def _heuristic(self, a: tuple, b: tuple) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def animate_astar_pathfinding(self):
        """Calcula el algoritmo A* internamente y devuelve la animación custom."""
        start = self.start_coord
        end = self.end_coord

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        
        g_score = { (r, c): float('inf') for r in range(self.rows) for c in range(self.cols) }
        g_score[start] = 0
        
        f_score = { (r, c): float('inf') for r in range(self.rows) for c in range(self.cols) }
        f_score[start] = self._heuristic(start, end)
        
        open_set_hash = {start}

        visited_order = []
        path = []

        # Cálculo puramente matemático (instantáneo)
        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)

            if current == end:
                curr = current
                path = [curr]
                while curr in came_from:
                    curr = came_from[curr]
                    path.append(curr)
                path.reverse()
                break 

            if current != start:
                visited_order.append(current)

            for d_r, d_c in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + d_r, current[1] + d_c)
                
                if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                    if self.grid_state[neighbor[0]][neighbor[1]] == 1:
                        continue

                    tentative_g_score = g_score[current] + 1
                    
                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                        if neighbor not in open_set_hash:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
                            open_set_hash.add(neighbor)

        # Delegar la tarea visual a las clases especializadas
        anims_to_play = []

        if visited_order:
            # Esta animación manejará todo el color sin trabar el loop
            anims_to_play.append(AStarAnimation(self, visited_order, path, run_time=3.0))

        if path:
            # Agregamos la línea amarilla limpia con ShowCreation
            path_points = [self.get_cell(r, c).get_center() for r, c in path]
            path_line = manimlib.VMobject()
            path_line.set_points_as_corners(path_points)
            path_line.set_stroke(manimlib.YELLOW, width=6.0)
            path_line.hide = True
            self.add(path_line)
            animation = manimlib.ShowCreation(path_line, run_time=1.5)
            old_begin = animation.begin
            def new_begin():
                old_begin()
                path_line.hide = False
            animation.begin = new_begin
            anims_to_play.append(animation)
        else:
            print("A*: ¡No se encontró camino!")

        return anims_to_play


class LLMExample(Example):
    def construct(self) -> None:
        if not imported_llm_scene_controller:
            raise RuntimeError(
                'LLMSceneController is required for LLMExample. Install with: pip install "manimgl[llm] @ git+https://github.com/MathItYT/manimgl"'
            )
        self.prompt_mode: str = "no_prompt"
        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        self.llm_controller = LLMSceneController(
            self,
            # api_key=os.getenv("GROQ_API_KEY"),
            # base_url="https://api.groq.com/openai/v1",
            # model="openai/gpt-oss-120b",
            client=self.client,
            model="openai/gpt-oss-120b",
        )
        def _build_astar_grid(scene, kwargs: dict):
            return AStarGrid(**kwargs)

        # 2. Registrar el Mobject
        self.llm_controller.register_mobject_type(
            name="AStarGrid",
            base_model=VMobjectParams, 
            fields={
                "rows": (Optional[DynNum], ...),
                "cols": (Optional[DynNum], ...),
                "obstacle_density": (Optional[DynNum], ...),
                "cell_size": (Optional[DynNum], ...)
            },
            builder=_build_astar_grid
        )

        # 3. Registrar el método de ejecución
        # Como el método no recibe parámetros extras, pasamos un diccionario vacío en fields
        self.llm_controller.register_method_type(
            name="animate_astar_pathfinding",
            fields={}
        )

        self.prompt = (
            manimlib.Text("")
            .add(manimlib.Dot())
            .to_edge(manimlib.DOWN, buff=0.5)
        )
        self.add(self.prompt)
    
    def prompt_pipeline(self, prompt: str) -> bool:
        new_prompt = prompt + "\n\nRemember that when creating an object you must display it on the scene using `add` or `play`. Also don't use SVGMobject as there're no SVG files in this environment."
        return self.llm_controller.run_prompt(
            new_prompt,
            response_mode="actions",
            additional_system_prompt="""1. When plotting 2D graphs or parametric curves, sample up to 100 points for smoothness, not more, to avoid performance issues.
2. When plotting on top of axes, make sure to use CoordinateSystemGraph, CoordinateSystemParametricCurve, or similar methods that are aware of the axes, instead of plotting raw FunctionGraph/ParametricCurves or whatever you are using, to ensure proper scaling and alignment.
3. Make sure objects will fit inside the frame and be visible. Center has coordinates (0, 0) and width and height of the frame are 14.22222 and 8 respectively, so don't make objects too large or too far from the center.
4. When plotting 3D objects use a minimum of 100, up to 200 samples in each u and v to bring a high resolution without crashing the environment.
5. Always prefer number plane over axes.
6. When adding a number plane all parameters should be null even if prompt specifies a color or x/y range to avoid performance issues.""",
            reasoning_effort="medium",
        )

    def on_key_press(self, symbol, modifiers):
        if (
            symbol == key.P
            and modifiers & key.MOD_CTRL
            and self.prompt_mode != "busy"
        ):
            self.prompt_mode = (
                "prompt"
                if self.prompt_mode != "prompt"
                else "no_prompt"
            )
            if self.prompt_mode == "prompt":
                self.prompt.become(
                    manimlib.Text(
                        "Escribe un prompt para el LLM y presiona Enter",
                        font_size=24,
                    ).to_edge(manimlib.DOWN, buff=0.5)
                )
                self.prompt.text = None
            else:
                self.prompt.become(
                    manimlib.Text("")
                    .add(manimlib.Dot())
                    .to_edge(manimlib.DOWN, buff=0.5)
                )
                self.prompt.text = None
        elif symbol == key.ENTER and self.prompt_mode == "prompt":
            prompt: manimlib.Text = self.prompt
            prompt_text = prompt.text
            prompt.become(
                manimlib.Text(
                    "Obteniendo respuesta del LLM...", font_size=24
                ).to_edge(manimlib.DOWN, buff=0.5)
            )
            self.prompt.text = None
            self.prompt_mode = "busy"

            def target():
                self.save_state()
                success = self.prompt_pipeline(prompt_text)
                self.prompt.become(
                    manimlib.Text(
                        "",
                        font_size=24,
                    )
                    .add(manimlib.Dot())
                    .to_edge(manimlib.DOWN, buff=0.5)
                )
                self.prompt_mode = "no_prompt"
                if not success:
                    self.undo()

            threading.Thread(
                target=target,
                daemon=True,
            ).start()
        elif self.prompt_mode == "no_prompt":
            super().on_key_press(symbol, modifiers)
        elif self.prompt_mode != "busy":
            if symbol == key.BACKSPACE:
                prompt: manimlib.Text = self.prompt
                prompt_text = prompt.text
                if prompt_text:
                    prompt.become(
                        manimlib.Text(
                            prompt_text[:-1], font_size=24
                        ).to_edge(manimlib.DOWN, buff=0.5)
                    )
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
                prompt.become(
                    manimlib.Text(
                        prompt_text + char, font_size=24
                    ).to_edge(manimlib.DOWN, buff=0.5)
                )
                prompt.text = prompt_text + char
    
    def get_state(self):
        return manimlib.SceneState(
            self,
            dont_modify=[self.prompt],
        )


class ExplanationLLMExample(LLMExample):
    def setup(self):
        super().setup()
        self.conversation = None

    def prompt_pipeline(self, prompt: str) -> bool:
        instructions = self.generate_instructions(prompt)
        if instructions is None:
            print("Failed to generate instructions from LLM")
            return False
        for instruction in instructions:
            success = super().prompt_pipeline(instruction)
            if not success:
                print(f"Failed to execute instruction: {instruction}")
                return False
        return True

    def generate_instructions(self, prompt: str) -> list[str] | None:
        attempts = 3
        json_schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["steps"],
            "additionalProperties": False,
        }
        if self.conversation is None:
            self.init_conversation()
        self.conversation.append({
            "role": "user",
            "content": prompt
        })

        for attempt in range(attempts):
            try:
                response = self.client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=self.conversation,
                    response_format={"type": "json_schema", "json_schema": {"name": "Instructions", "strict": True, "schema": json_schema}},
                    reasoning_effort="medium"
                )
                response_text = response.choices[0].message.content or ""
                data = json.loads(response_text)
                self.conversation.append({
                    "role": "assistant",
                    "content": response_text
                })
                return data["steps"]
            except Exception as exc:
                print(f"Error generating instructions (attempt {attempt+1}/{attempts}): {exc}")
        return None
    
    def init_conversation(self) -> None:
        self.conversation = [
            {
                "role": "system",
                "content": f"""
You are an expert Manim Scene Architect. Your sole purpose is to translate mathematical prompts into a precise, step-by-step blueprint for a Manim coding agent. 

You must strictly adhere to the following directives:

### 1. Core Principles
* **Show, Don't Tell:** Never explain the math in text. Your output must consist entirely of instructions to create the visual scene.
* **Zero Commentary:** Do not include greetings, explanations, or conversational filler. Output only the requested JSON.
* **Atomic Actions:** Each instruction must contain exactly ONE action. If you want to create a red circle and write text, that must be two completely separate instructions.

### 2. Animation & Object Rules
* **Direct Animation:** If an object is going to be animated into the scene, use `animated_add` or `animated_remove` directly with the animation details. 
* **Static Rendering:** Only use `add_no_animation` or `remove_no_animation` if the object must appear or disappear instantly without any transition. Never use a static add/remove immediately followed by an animation of that same object appearing/disappearing.
* **Explicit Variables:** Always assign clear, logical variable names to objects (e.g., `circle_1`, `text_eq_1`) so the agent can reference them accurately in subsequent steps.

### 3. Granularity Guideline (Strict)
Never write complex, multi-step instructions. Break everything down into the smallest possible logical units. 
* *Bad Example:* "Add three clusters of four points each."
* *Good Example:* 1. "Create four points assigned to variable `points1`."
    2. "Arrange `points1`."
    3. "Surround `points1` with a circle named `circle1`."
    4. "Group `points1` and `circle1` into a variable named `cluster1`."
    5. [Repeat for other clusters]
    6. "Arrange `cluster1`, `cluster2`, `cluster3`."
    7. "Group all clusters into `all_clusters`."
    8. "Animate creation of `all_clusters`."

### 4. Output Specification
* You must return a valid JSON object strictly adhering to the provided schema.
* Do not invent parameters. Do not omit required parameters.
* The `details` field must be exhaustive: specify object types, colors, positions, sizes, and animation styles so the agent has zero ambiguity when coding.

### 5. Follow-up Protocol
* Treat all subsequent user questions or clarifications as requests to modify the Manim scene. 
* Never answer directly. Translate the answer into scene modification instructions (e.g., adding text to the screen, highlighting a part of the equation). 
* If a question cannot be visually answered, output instructions to render the question as a Text object on the screen.

### 6. Mobject organization
* Always put title at the top of the scene and subtitle right below it, both centered horizontally.
* Main visual content should be centered in the remaining space between subtitle and bottom of the screen, but can be shifted or rearranged as needed to fit the content and make it visually appealing.
* Make sure to not overlap the title and subtitle with the main visual content, and to keep enough space between them for clarity.
""".strip()
            }
        ]


class VirtualCameraExample(Example):
    def setup(self) -> None:
        if not imported_virtual_camera:
            raise RuntimeError(
                'bind_scene_to_virtual_camera is required for VirtualCameraExample. Install with: pip install "manimgl[virtual_camera] @ git+https://github.com/MathItYT/manimgl"'
            )
        self.virtual_camera_sink = bind_scene_to_virtual_camera(
            self,
            fps=30,
            frame_stride=2,
            block_until_next_frame=False,
        )
        super().setup()

    def construct(self) -> None:
        title = manimlib.Text("ManimGL Virtual Camera", font_size=72)
        subtitle = manimlib.Text(
            "Virtual camera output", font_size=28
        )
        manimlib.Group(title, subtitle).arrange(manimlib.DOWN)
        self.add(title, subtitle)
        self.play(manimlib.FadeIn(title), manimlib.FadeIn(subtitle))
        self.wait(5)


class YouTubeChatExample(manimlib.InteractiveScene):
    """Render YouTube live chat messages into a fixed text overlay.

    Set YOUTUBE_LIVE_VIDEO_ID to an active livestream video id (or full URL).
    """

    def construct(self) -> None:
        if not imported_youtube_chat:
            raise RuntimeError(
                'YouTube chat extra is required for YouTubeChatExample. Install with: pip install "manimgl[youtube_chat] @ git+https://github.com/MathItYT/manimgl"'
            )

        video_id = os.getenv("YOUTUBE_LIVE_VIDEO_ID")
        if not video_id:
            raise RuntimeError(
                "Set YOUTUBE_LIVE_VIDEO_ID with a live video id or URL before running YouTubeChatExample"
            )

        self.chat_client = YouTubeLiveChatClient(video_id)

        title = manimlib.Text("Sample text", font_size=40).to_edge(
            manimlib.UP, buff=0.4
        )
        title.fix_in_frame()

        chat_box = manimlib.Rectangle(width=11.6, height=6.0)
        chat_box.set_stroke(color=manimlib.BLUE, width=2)
        chat_box.set_fill(color=manimlib.BLACK, opacity=0.3)
        chat_box.to_edge(manimlib.DOWN, buff=0.4)
        chat_box.fix_in_frame()

        bind_youtube_chat_to_feed(
            self,
            chat_box,
            self.chat_client,
            max_messages=8,
            update_fps=8,
            avatar_height=0.25,
            text_font_size=20,
            markdown_mobject_config={
                "text_font": [
                    "SF Pro Display",
                    "Twitter Color Emoji",
                ],
            },
            line_buff=0.12,
        )

        self.add(chat_box, title)
        self.wait(60)

    def on_close(self) -> None:
        if hasattr(self, "chat_client"):
            self.chat_client.stop(timeout=0.0)
        super().on_close()


class TranscriptionLLMExample(Example):
    def construct(self) -> None:
        if not imported_transcriber:
            raise RuntimeError(
                'ElevenLabsRealtimeTranscriber is required for TranscriptionLLMExample. Install with: pip install "manimgl[transcription] @ git+https://github.com/MathItYT/manimgl"'
            )
        if not imported_llm_scene_controller:
            raise RuntimeError(
                'LLMSceneController is required for TranscriptionLLMExample. Install with: pip install "manimgl[llm] @ git+https://github.com/MathItYT/manimgl"'
            )

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set ELEVENLABS_API_KEY environment variable before running TranscriptionLLMExample"
            )

        transcriber = ElevenLabsRealtimeTranscriber(
            api_key=api_key,
            sample_rate=16000,
            audio_format="pcm_16000",
            commit_strategy="vad",
            language_code="es",
            max_audio_queue_chunks=24,
            chunks_per_enqueue=2,
        )

        transcript_text = manimlib.Text(
            "Habla para transcribir...", font_size=24
        ).to_edge(manimlib.DOWN, buff=0.5)
        transcript_text.fix_in_frame()
        bind_transcriber_to_text(
            self,
            transcript_text,
            transcriber,
            update_fps=5,
            partial_update_fps=2,
            render_partial=True,
            build_text_off_main_thread=True,
            font_size=24,
        )

        llm_status = manimlib.Text("LLM listo", font_size=20).to_edge(
            manimlib.DOWN, buff=1.2
        )
        llm_status.fix_in_frame()
        self.add(transcript_text, llm_status)

        self.llm_controller = LLMSceneController(
            self,
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1",
            model="moonshotai/kimi-k2.5",
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
                llm_status.become(
                    manimlib.Text(
                        "Error en la interaccion previa", font_size=20
                    ).to_edge(manimlib.DOWN, buff=1.2)
                )
                llm_status.fix_in_frame()
                return
            if should_set_ready:
                llm_status.become(
                    manimlib.Text("LLM listo", font_size=20).to_edge(
                        manimlib.DOWN, buff=1.2
                    )
                )
                llm_status.fix_in_frame()

        self.add_updater(_flush_llm_status)

        def _run_llm_from_transcript(prompt: str) -> None:
            nonlocal llm_busy, llm_ready_pending, llm_error_pending
            had_error = False
            try:
                self.llm_controller.run_prompt(
                    prompt, reasoning_effort="high"
                )
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
                llm_status.become(
                    manimlib.Text("LLM listo", font_size=20).to_edge(
                        manimlib.DOWN, buff=1.2
                    )
                )
                llm_status.fix_in_frame()
                return

            with llm_lock:
                if llm_busy:
                    llm_status.become(
                        manimlib.Text(
                            "LLM ocupado", font_size=20
                        ).to_edge(manimlib.DOWN, buff=1.2)
                    )
                    llm_status.fix_in_frame()
                    return
                llm_busy = True

            llm_status.become(
                manimlib.Text(
                    f"LLM ejecutando: {prompt[:40]}", font_size=20
                ).to_edge(manimlib.DOWN, buff=1.2)
            )
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
                manimlib.Text(
                    "Ctrl+P activo: esperando transcripcion committed",
                    font_size=20,
                ).to_edge(manimlib.DOWN, buff=1.2)
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


class HandTrackingExample(manimlib.InteractiveScene):
    def construct(self) -> None:
        if not imported_hand_tracking:
            raise RuntimeError(
                'Vision extra is required for HandTrackingExample. Install with: pip install "manimgl[vision] @ git+https://github.com/MathItYT/manimgl"'
            )

        video_mob = manimlib.VideoMobject.from_video(
            0,
            height=5,
            flip_horizontal=True,
        )
        self.video_mob = video_mob
        video_mob.play()

        self.tracker = HandMotionTracker(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            pinch_threshold=0.055,
            movement_threshold=0.012,
            smoothing=0.45,
            model_cache_dir="D:\\manimgl_cache",
        )
        bind_hand_tracker_to_video(
            video_mob, self.tracker, enqueue_every_n_frames=2
        )

        circle = manimlib.Circle(radius=0.22)
        circle.set_fill(color=manimlib.YELLOW, opacity=0.85)
        circle.set_stroke(color=manimlib.WHITE, width=3)
        circle.move_to(video_mob)

        hand_mesh = HandMesh(
            reference_mobject=video_mob,
            stroke_color=manimlib.BLUE,
            stroke_width=3,
            z_value=0.05,
        )
        bind_hand_mesh_to_tracker(
            self, hand_mesh, self.tracker, update_fps=30
        )

        status = manimlib.Text(
            "Mueve tu mano para controlar el circulo", font_size=24
        )
        status.to_edge(manimlib.DOWN, buff=0.3)
        status.fix_in_frame()
        self._last_hand_status_message = (
            "Mueve tu mano para controlar el circulo"
        )

        bind_hand_position_to_mobject(
            self,
            circle,
            self.tracker,
            reference_mobject=video_mob,
            update_fps=30,
            z_value=0.0,
            only_when_detected=True,
        )

        def _on_gesture(state: HandMotionState) -> None:
            if not state.detected:
                message = "No se detecta mano"
            elif state.pinch:
                message = "Gesto: pinch"
            else:
                message = f"Gesto: {state.gesture}"

            # Avoid rebuilding text every callback; this removes expensive per-frame work.
            if message != self._last_hand_status_message:
                status.become(
                    manimlib.Text(message, font_size=24).to_edge(
                        manimlib.DOWN, buff=0.3
                    )
                )
                status.fix_in_frame()
                self._last_hand_status_message = message

            if state.pinch:
                circle.set_fill(color=manimlib.GREEN, opacity=0.9)
            else:
                circle.set_fill(color=manimlib.YELLOW, opacity=0.85)

        bind_hand_gesture_callback(
            self, self.tracker, _on_gesture, update_fps=20
        )

        self.add(video_mob, hand_mesh, circle, status)

    def on_close(self) -> None:
        if hasattr(self, "video_mob"):
            self.video_mob.stop()
            unbind_hand_tracker_from_video(self.video_mob)

        if hasattr(self, "tracker"):
            self.tracker.stop(timeout=0.0)

        super().on_close()
