# ManimGL Presentation Engine
ManimGL Presentation Engine is a fork of original 3Blue1Brown's Manim aimed to be a live presentation engine with realtime performance.

## Installation
`pip install git+https://github.com/MathItYT/manimgl.git`

## How to run?
Create an `manimlib.InteractiveScene` subclass in a `main.py` (or any other filename else) module.

For example:

```python
import manimlib


class Example(manimlib.InteractiveScene):
    def construct(self) -> None:
        self.start_mic_recording(rate=48000, channels=2)  # Iniciar grabación de micrófono
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
```

Then run in your terminal:

```bash
manimgl main.py Example
```
