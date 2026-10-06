from manimlib import *
import re
from collections import defaultdict
from enum import Enum
from pydub import AudioSegment
from manimlib.utils.vfx_presets import *


class VFXScene(Scene):
    def construct(self):
        self.file_writer.crf = 1
        glowing_circle = Circle(radius=1.8, stroke_color=TEAL_A, stroke_width=6)
        glowing_circle.to_edge(LEFT)
        glowing_circle.add_vfx(Glow(threshold=0.05, intensity=1.6, radius=40.0))

        crisp_text = Text("CRISP TEXT", font_size=40, color=WHITE)
        crisp_text.to_edge(RIGHT)

        glitchy_square = Square(side_length=2.0, stroke_color=PURE_RED, stroke_width=4)
        glitchy_square.add_vfx(Glitch(intensity=0.7, speed=8.0))

        self.play(
            ShowCreation(glowing_circle),
            Write(crisp_text),
            ShowCreation(glitchy_square),
            run_time=3
        )
        self.wait(1)


class TestTrajectoryMotionBlur(Scene):
    def construct(self):
        self.file_writer.crf = 1

        # Objeto de prueba con MotionBlur dinámico
        ball = Circle(radius=0.4, fill_color=YELLOW, fill_opacity=1.0, stroke_color=WHITE, stroke_width=3)
        ball.move_to(LEFT * 5 + DOWN * 2)

        # auto_direction es True por defecto si no se pasa 'direction'
        ball.add_vfx(MotionBlur(strength=0.5, max_blur=0.18))
        # ball.add_vfx(Glow(radius=20.0, intensity=1.4))

        label = Text("Dynamic Trajectory Motion Blur", font_size=28).to_edge(UP)
        self.add(label, ball)

        # 1. Aceleración horizontal rápida (estela puramente horizontal)
        self.play(
            ball.animate.shift(RIGHT * 10),
            run_time=1.2,
            rate_func=rush_into
        )

        # 2. Desplazamiento diagonal rápido hacia arriba a la izquierda
        self.play(
            ball.animate.shift(LEFT * 8 + UP * 4),
            run_time=1.0,
            rate_func=smooth
        )

        # 3. Trayectoria circular continua (el vector de blur rota 360° en tiempo real)
        self.play(
            Rotate(ball, angle=TAU, about_point=ORIGIN),
            run_time=2.0,
            rate_func=linear
        )

        # 4. Estado estacionario (la estela debe colapsar a 0 sin blur residual)
        self.wait(1.0)



def select_word(mobject: StringMobject, word: str, index: int = 0) -> re.Pattern:
    return mobject.select_part(re.compile(rf"(?<!\S){re.escape(word)}(?!\S)"), index=index)


def ease_out_cubic(x: float) -> float:
    """Input x must be between 0.0 and 1.0"""
    return 1 - pow(1 - x, 3)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def smoothstep(x: float) -> float:
    """Input x must be between 0.0 and 1.0"""
    t = clamp(x, 0.0, 1.0)
    # Evaluate polynomial
    return t * t * (3.0 - 2.0 * t)


class AnimationType(Enum):
    FADE_IN_WORDS = "fade-in-words"
    SCALE_AND_WAIT = "scale-and-wait"
    NO_ANIMATION = "no-animation"


class SubtitleScene(InteractiveScene):
    def subtitle(
        self,
        *extra_animations: Animation,
        words: list[str],
        audio: str | None = None,
        run_time: float | None = None,
        pause: float = 0.0,
        font_size: float = 72.0,
        word_to_color: dict[str, str] | None = None,
        edge: np.ndarray | None = None,
        animation_type: AnimationType = AnimationType.FADE_IN_WORDS
    ) -> None:
        t = TypstText(" ".join(words), font="New Computer Modern", font_size=font_size)
        if edge is not None:
            t.to_edge(edge)
        if word_to_color:
            for word, color in word_to_color.items():
                t.set_color_by_typst(word, color)

        # 1. Obtenemos las partes de cada palabra una sola vez
        word_mobs = [select_word(t, word) for word in words]
        # 3. Animamos las mismas instancias
        if audio:
            self.add_sound(audio)
            run_time = len(AudioSegment.from_file(audio)) / 1000
        if run_time is None:
            raise ValueError("Subtitle run_time can't be None")
        if animation_type == AnimationType.FADE_IN_WORDS:
            for wm in word_mobs:
                wm.add_vfx(Glow(radius=12.5, intensity=0.75))
                wm.add_vfx(MotionBlur(strength=0.5, max_blur=0.1, mobject=wm, auto_direction=True))
            self.play(
                LaggedStart(
                    *(FadeIn(wm, shift=UP, rate_func=ease_out_cubic) for wm in word_mobs),
                    group=Group(*word_mobs),
                    lag_ratio=0.5,
                    run_time=run_time
                ),
                *extra_animations
            )
        elif animation_type == AnimationType.SCALE_AND_WAIT:
            t.add_vfx(Glow(radius=12.5, intensity=0.75))
            t.add_vfx(ZoomBlur(mobject=t, strength=0.5))
            scale_run_time = min(run_time, 0.25)
            wait_run_time = run_time - max([scale_run_time] + [anim.run_time for anim in extra_animations])
            self.play(GrowFromCenter(t, run_time=scale_run_time, rate_func=ease_out_cubic), *extra_animations)
            if wait_run_time > 0.0:
                self.wait(max(1 / self.camera.fps, wait_run_time))
        elif animation_type == AnimationType.NO_ANIMATION:
            t.add_vfx(Glow(radius=12.5, intensity=0.75))
            if extra_animations:
                wait_run_time = run_time - max([anim.run_time for anim in extra_animations])
                self.play(*extra_animations)
                if wait_run_time > 0.0:
                    self.wait(max(1 / self.camera.fps, wait_run_time))
            else:
                self.wait(run_time)
        if pause > 0.0:
            self.wait(pause)
        self.remove(t)

    def subtitle_this(
        self,
        *extra_animations: Animation,
        words: list[str],
        audio: str,
        pause: float = 0.0,
        word_to_color: dict[str, str] | None = None
    ) -> None:
        self.subtitle(
            *extra_animations,
            words=words,
            audio=audio,
            edge=DOWN,
            pause=pause,
            word_to_color=word_to_color,
            animation_type=AnimationType.SCALE_AND_WAIT
        )


class Intro(SubtitleScene):
    def construct(self):
        self.file_writer.crf = 1
        self.camera.background_rgba = color_to_rgba("#101010", 1)
        to_speak = [
            (["Hoy", "aprenderás"], "hoy_aprenderas.mp3", 0.0, None),
            (["sobre", "números", "reales", "($RR$)."], "sobre_numeros_reales.mp3", 0.25, {"números reales ($RR$)": YELLOW}),
            (["De", "seguro", "ya", "sabes"], "de_seguro_ya_sabes.mp3", 0.0, None),
            (["sobre", "los", "números", "racionales", "($QQ$)"], "sobre_los_numeros_racionales.mp3", 0.0, {"números racionales ($QQ$)": YELLOW}),
            (["y", "es", "un", "requisito"], "y_es_un_requisito.mp3", 0.0, None),
            (["para", "ver", "este", "video."], "para_ver_este_video.mp3", 0.25, None),
            (["También", "es", "requisito", "saber"], "tambien_es_requisito_saber.mp3", 0.0, None),
            (["resolver", "desigualdades", "lineales."], "resolver_desigualdades_lineales.mp3", 0.25, {"desigualdades lineales": YELLOW}),
            (["Si", "manejas", "bien", "ambas", "cosas"], "si_manejas_bien_ambas_cosas.mp3", 0.0, None),
            (["podemos", "comenzar", "a", "aprender."], "podemos_comenzar_a_aprender.mp3", 0.25, None),
            (["Y", "recuerda", "dar", "*like*,"], "y_recuerda_dar_like.mp3", 0.0, {"like": PURE_RED}),
            (["*suscribirte*", "y", "*comentar*"], "suscribirte_y_comentar.mp3", 0.0, {"suscribirte": PURE_RED, "comentar": PURE_RED}),
            (["para", "apoyar", "al", "canal."], "para_apoyar_al_canal.mp3", 0.0, None)
        ]
        for words, audio, pause, word_to_color in to_speak:
            self.subtitle(words=words, audio=audio, pause=pause, word_to_color=word_to_color)


class RepasoRacionales(SubtitleScene):
    def construct(self) -> None:
        self.file_writer.crf = 1
        self.camera.background_rgba = color_to_rgba("#101010", 1.0)
        self.subtitle_this(words=["Bien,"], audio="bien.mp3")
        self.subtitle_this(words=["entonces como debes de saber,"], audio="entonces_como_debes_de_saber.mp3")
        definition = Typst("QQ = {p/q : p,q in ZZ, q != 0}", font_size=96)
        definition.add_vfx(Glow(radius=12.5, intensity=0.75))
        rect = SurroundingRectangle(definition, color=WHITE, buff=0.5)
        rect.add_vfx(Glow(radius=12.5, intensity=0.75))
        self.subtitle_this(Write(definition, run_time=1.0), ShowCreation(rect, run_time=1.0), words=["el conjunto de todos"], audio="el_conjunto_de_todos.mp3")
        self.subtitle_this(words=["los números racionales ($QQ$)"], audio="los_numeros_racionales.mp3")
        self.subtitle_this(Indicate(definition["p/q"], scale_factor=1), words=["son todas las fracciones"], audio="son_todas_las_fracciones.mp3")
        self.subtitle_this(Indicate(definition["p,q in ZZ"], scale_factor=1), words=["de números enteros"], audio="de_numeros_enteros.mp3")
        self.subtitle_this(words=["donde el denominador"], audio="donde_el_denominador.mp3")
        self.subtitle_this(Indicate(definition["q != 0"], scale_factor=1), words=["es diferente de cero."], audio="es_diferente_de_cero.mp3")
        self.play(FadeOut(definition, run_time=0.5), FadeOut(rect, run_time=0.5))


class Wait(Animation):
    def __init__(self, run_time: float) -> None:
        super().__init__(Mobject(), run_time=run_time, remover=True)


class FieldAxioms(SubtitleScene):
    def construct(self) -> None:
        self.file_writer.crf = 1
        self.camera.background_rgba = color_to_rgba("#101010", 1.0)
        self.subtitle_this(words=["Además,"], audio="ademas.mp3")
        self.subtitle_this(words=["es importante reconocer"], audio="es_importante_reconocer.mp3")
        self.subtitle_this(words=["que los números racionales ($QQ$)"], audio="que_los_numeros_racionales.mp3")
        self.subtitle_this(words=["son un cuerpo ordenado."], audio="son_un_cuerpo_ordenado.mp3", word_to_color={"cuerpo ordenado": YELLOW}, pause=0.25)
        self.subtitle_this(words=["Que sea un cuerpo"], audio="que_sea_un_cuerpo.mp3", word_to_color={"cuerpo": YELLOW})
        self.subtitle_this(words=["quiere decir que cumple"], audio="quiere_decir_que_cumple.mp3")
        field_axioms = TypstText("*Axiomas de Cuerpo*", font="New Computer Modern", font_size=96).to_edge(UP)
        field_axioms.add_vfx(Glow(radius=12.5, intensity=0.75))
        self.subtitle_this(Succession(Write(field_axioms, run_time=1), Indicate(field_axioms, scale_factor=1, run_time=0.5)), words=["con los llamados axiomas de cuerpo,"], audio="con_los_llamados_axiomas_de_cuerpo.mp3", word_to_color={"axiomas de cuerpo": YELLOW})
        conmutatividad_suma = TypstText("- *Conmutatividad de la suma:* $forall a,b in QQ, a + b = b + a$", font="New Computer Modern", font_size=48)
        conmutatividad_suma.add_vfx(Glow(radius=12.5, intensity=0.75))
        conmutatividad_suma.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=conmutatividad_suma, auto_direction=True))
        conmutatividad_producto = TypstText("- *Conmutatividad del producto:* $forall a,b in QQ, a dot b = b dot a$", font="New Computer Modern", font_size=48)
        conmutatividad_producto.add_vfx(Glow(radius=12.5, intensity=0.75))
        conmutatividad_producto.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=conmutatividad_producto, auto_direction=True))
        Group(conmutatividad_suma, conmutatividad_producto).arrange(DOWN, aligned_edge=LEFT)
        self.subtitle_this(FadeIn(conmutatividad_suma, run_time=1.0, shift=UP, rate_func=ease_out_cubic), FadeIn(conmutatividad_producto, run_time=1.0, shift=UP, rate_func=ease_out_cubic), words=["que son la conmutatividad"], audio="que_son_la_conmutatividad.mp3", word_to_color={"conmutatividad": YELLOW})
        self.subtitle_this(Succession(Wait(0.5), AnimationGroup(FadeOut(conmutatividad_suma, run_time=0.5), FadeOut(conmutatividad_producto, run_time=0.5))), words=["en la suma y producto,"], audio="en_la_suma_y_producto.mp3", word_to_color={"suma": YELLOW, "producto": YELLOW})
        asociatividad_suma = TypstText("- *Asociatividad de la suma:* $forall a,b,c in QQ, (a + b) + c = a + (b + c)$", font="New Computer Modern", font_size=36)
        asociatividad_suma.add_vfx(Glow(radius=12.5, intensity=0.75))
        asociatividad_suma.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=asociatividad_suma, auto_direction=True))
        asociatividad_producto = TypstText("- *Asociatividad del producto:* $forall a,b,c in QQ, (a dot b) dot c = a dot (b dot c)$", font="New Computer Modern", font_size=36)
        asociatividad_producto.add_vfx(Glow(radius=12.5, intensity=0.75))
        asociatividad_producto.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=asociatividad_producto, auto_direction=True))
        Group(asociatividad_suma, asociatividad_producto).arrange(DOWN, aligned_edge=LEFT)
        self.subtitle_this(FadeIn(asociatividad_suma, run_time=1.0, shift=UP, rate_func=ease_out_cubic), FadeIn(asociatividad_producto, run_time=1.0, shift=UP, rate_func=ease_out_cubic), words=["la asociatividad"], audio="la_asociatividad.mp3", word_to_color={"asociatividad": YELLOW})
        self.subtitle_this(Succession(Wait(0.5), AnimationGroup(FadeOut(asociatividad_suma, run_time=0.5), FadeOut(asociatividad_producto, run_time=0.5))), words=["en ambas operaciones también,"], audio="en_ambas_operaciones_tambien.mp3", word_to_color={"ambas operaciones": YELLOW})
        elemento_neutro_suma = TypstText("- *Elemento neutro de la suma:* $exists 0 in QQ, forall a in QQ, a + 0 = a$", font="New Computer Modern", font_size=48)
        elemento_neutro_suma.add_vfx(Glow(radius=12.5, intensity=0.75))
        elemento_neutro_suma.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=elemento_neutro_suma, auto_direction=True))
        elemento_neutro_producto = TypstText("- *Elemento neutro del producto:* $exists 1 in QQ, forall a in QQ, a dot 1 = a$", font="New Computer Modern", font_size=48)
        elemento_neutro_producto.add_vfx(Glow(radius=12.5, intensity=0.75))
        elemento_neutro_producto.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=elemento_neutro_producto, auto_direction=True))
        Group(elemento_neutro_suma, elemento_neutro_producto).arrange(DOWN, aligned_edge=LEFT)
        self.subtitle_this(FadeIn(elemento_neutro_suma, run_time=1.0, shift=UP, rate_func=ease_out_cubic), FadeIn(elemento_neutro_producto, run_time=1.0, shift=UP, rate_func=ease_out_cubic), words=["la existencia de un elemento neutro"], audio="la_existencia_de_un_elemento_neutro.mp3", word_to_color={"elemento neutro": YELLOW})
        self.subtitle_this(Succession(Wait(0.5), AnimationGroup(FadeOut(elemento_neutro_suma, run_time=0.5), FadeOut(elemento_neutro_producto, run_time=0.5))), words=["en cada una,"], audio="en_cada_una.mp3", word_to_color={"cada una": YELLOW})
        inverso_suma = TypstText("- *Inverso aditivo:* $forall a in QQ, exists -a in QQ, a + (-a) = 0$", font="New Computer Modern", font_size=36)
        inverso_suma.add_vfx(Glow(radius=12.5, intensity=0.75))
        inverso_suma.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=inverso_suma, auto_direction=True))
        inverso_producto = TypstText("- *Inverso multiplicativo:* $forall a in QQ, a != 0, exists a^(-1) in QQ, a dot a^(-1) = 1$", font="New Computer Modern", font_size=36)
        inverso_producto.add_vfx(Glow(radius=12.5, intensity=0.75))
        inverso_producto.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=inverso_producto, auto_direction=True))
        Group(inverso_suma, inverso_producto).arrange(DOWN, aligned_edge=LEFT)
        self.subtitle_this(FadeIn(inverso_suma, run_time=1.0, shift=UP, rate_func=ease_out_cubic), FadeIn(inverso_producto, run_time=1.0, shift=UP, rate_func=ease_out_cubic), words=["la existencia de inverso aditivo"], audio="la_existencia_de_inverso_aditivo.mp3", word_to_color={"inverso aditivo": YELLOW})
        self.subtitle_this(Succession(Wait(0.5), AnimationGroup(FadeOut(inverso_suma, run_time=0.5), FadeOut(inverso_producto, run_time=0.5))), words=["e inverso multiplicativo,"], audio="e_inverso_multiplicativo.mp3", word_to_color={"inverso multiplicativo": YELLOW})
        distributividad = TypstText("- *Distributividad:* $forall a,b,c in QQ, a dot (b + c) = a dot b + a dot c$", font="New Computer Modern", font_size=48)
        distributividad.add_vfx(Glow(radius=12.5, intensity=0.75))
        distributividad.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=distributividad, auto_direction=True))
        self.subtitle_this(FadeIn(distributividad, run_time=1.0, shift=UP, rate_func=ease_out_cubic), words=["y la distributividad"], audio="y_la_distributividad.mp3", word_to_color={"distributividad": YELLOW})
        self.subtitle_this(Succession(Wait(0.5), FadeOut(distributividad, run_time=0.5)), words=["de la multiplicación respecto de la suma."], audio="de_la_multiplicacion_respecto_de_la_suma.mp3", word_to_color={"multiplicación": YELLOW, "suma": YELLOW})
        self.play(FadeOut(field_axioms, run_time=0.25))


class OrderAxioms(SubtitleScene):
    def construct(self) -> None:
        self.file_writer.crf = 1
        self.camera.background_rgba = color_to_rgba("#101010", 1.0)
        self.subtitle_this(words=["Y que sea un cuerpo ordenado es decir"], audio="y_que_sea_un_cuerpo_ordenado_es_decir.mp3", word_to_color={"cuerpo ordenado": YELLOW})
        self.subtitle_this(words=["que es un cuerpo que cumple además"], audio="que_es_un_cuerpo_que_cumple_ademas.mp3")
        order_axioms = TypstText("*Axiomas de Orden*", font="New Computer Modern", font_size=96).to_edge(UP)
        order_axioms.add_vfx(Glow(radius=12.5, intensity=0.75))
        self.subtitle_this(Succession(Write(order_axioms, run_time=1), Indicate(order_axioms, scale_factor=1, run_time=0.5)), words=["con los axiomas de orden,"], audio="con_los_axiomas_de_orden.mp3", word_to_color={"axiomas de orden": YELLOW})
        tricotomia = TypstText("- *Tricotomía:* $forall x,y in QQ, x < y or x = y or x > y$", font="New Computer Modern", font_size=48)
        tricotomia.add_vfx(Glow(radius=12.5, intensity=0.75))
        tricotomia.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=tricotomia, auto_direction=True))
        self.subtitle_this(FadeIn(tricotomia, run_time=1.0, shift=UP, rate_func=ease_out_cubic), words=["que son la tricotomía,"], audio="que_son_la_tricotomia.mp3", word_to_color={"tricotomía": YELLOW})
        self.subtitle_this(words=["o sea que dados"], audio="o_sea_que_dados.mp3")
        self.subtitle_this(Indicate(tricotomia["x,y in QQ"], scale_factor=1, run_time=1), words=["$x,y in QQ$,"], audio="dos_numeros_racionales_x_e_y.mp3", word_to_color={"x,y in QQ": YELLOW})
        self.subtitle_this(Succession(Wait(0.25), Indicate(tricotomia["x < y"], scale_factor=1, run_time=1.0), Wait(0.25), Indicate(tricotomia["x = y"], scale_factor=1, run_time=1.25), Wait(0.25), Indicate(tricotomia["x > y"], scale_factor=1, run_time=1.0)), words=["o bien $x < y$, o bien $x = y$, o bien $x > y$."], audio="o_bien_x_es_menor_que_y_o_bien_son_iguales_x_e_y_o_bien_x_es_mayor_que_y.mp3", word_to_color={"x < y": YELLOW, "x = y": YELLOW, "x > y": YELLOW})
        self.play(FadeOut(tricotomia, run_time=0.25))
        transitividad = TypstText("- *Transitividad:* $forall x,y,z in QQ, x < y and y < z => x < z$", font="New Computer Modern", font_size=36)
        transitividad.add_vfx(Glow(radius=12.5, intensity=0.75))
        transitividad.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=transitividad, auto_direction=True))
        self.subtitle_this(FadeIn(transitividad, run_time=1.0, shift=UP, rate_func=ease_out_cubic), words=["También está la transitividad,"], audio="tambien_esta_la_transitividad.mp3", word_to_color={"transitividad": YELLOW})
        self.subtitle_this(words=["que en simples palabras es que"], audio="que_en_simples_palabras_es_que.mp3")
        self.subtitle_this(Succession(Wait(0.25), Indicate(transitividad["x < y"], scale_factor=1, run_time=1.0), Wait(0.25), Indicate(transitividad["y < z"], scale_factor=1, run_time=1.25), Wait(0.25), Indicate(transitividad["x < z"], scale_factor=1, run_time=1.25)), words=["si $x < y$ y además $y < z$, entonces $x < z$."], audio="si_x_es_menor_que_y_y_ademas_la_y_es_menor_que_z_entonces_x_va_a_ser_menor_que_z.mp3", word_to_color={"x < y": YELLOW, "y < z": YELLOW, "x < z": YELLOW})
        self.play(FadeOut(transitividad, run_time=0.25))
        monotonia_suma = TypstText("- *Monotonía de la suma:* $forall x,y,z in QQ, x < y => x + z < y + z$", font="New Computer Modern", font_size=36)
        monotonia_suma.add_vfx(Glow(radius=12.5, intensity=0.75))
        monotonia_suma.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=monotonia_suma, auto_direction=True))
        self.subtitle_this(FadeIn(monotonia_suma, run_time=1.0, shift=UP, rate_func=ease_out_cubic), words=["También está la monotonía de la suma,"], audio="tambien_esta_la_monotonia_de_la_suma.mp3", word_to_color={"monotonía de la suma": YELLOW})
        self.subtitle_this(words=["o sea, podemos sumar un número racional"], audio="o_sea_podemos_sumar_un_numero_racional.mp3", word_to_color={"sumar un número racional": YELLOW})
        self.subtitle_this(words=["cualquiera a ambos lados de"], audio="cualquiera_a_ambos_lados_de.mp3", word_to_color={"cualquiera a ambos lados de": YELLOW})
        self.subtitle_this(Succession(Wait(0.25), FadeOut(monotonia_suma, run_time=0.25)), words=["una desigualdad,"], audio="una_desigualdad.mp3", word_to_color={"una desigualdad": YELLOW})
        monotonia_producto = TypstText("- *Monotonía del producto:* $forall x,y,z in QQ, x < y and 0 < z => x dot z < y dot z$", font="New Computer Modern", font_size=36)
        monotonia_producto.add_vfx(Glow(radius=12.5, intensity=0.75))
        monotonia_producto.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=monotonia_producto, auto_direction=True))
        self.subtitle_this(FadeIn(monotonia_producto, run_time=1.0, shift=UP, rate_func=ease_out_cubic), words=["y la monotonía del producto,"], audio="y_la_monotonia_del_producto.mp3", word_to_color={"monotonia del producto": YELLOW})
        self.subtitle_this(words=["que dice que podemos"], audio="que_dice_que_podemos.mp3")
        self.subtitle_this(words=["multiplicar por un número racional positivo"], audio="multiplicar_por_un_numero_racional_positivo.mp3", word_to_color={"multiplicar por un número racional positivo": YELLOW})
        self.subtitle_this(words=["cualquiera a ambos lados"], audio="cualquiera_a_ambos_lados.mp3", word_to_color={"cualquiera a ambos lados": YELLOW})
        self.subtitle_this(Succession(Wait(0.25), FadeOut(monotonia_producto, run_time=0.25)), words=["de una desigualdad."], audio="de_una_desigualdad.mp3", word_to_color={"de una desigualdad": YELLOW})
        self.play(FadeOut(order_axioms, run_time=0.25))


@lru_cache()
def new_char_to_cached_mob(char: str, **text_config):
    return Typst(char, **text_config)


manimlib.mobject.numbers.char_to_cahced_mob = new_char_to_cached_mob


def get_label(number_line: manimlib.NumberLine, x: float, label: str) -> Typst:
    label_typst = Typst(label, font_size=48).next_to(number_line.n2p(x), manimlib.DOWN, buff=0.3)
    label_typst.shift((label_typst.get_height() - Typst("0").get_height()) * manimlib.UP)
    return label_typst


class DataPresentation(InteractiveScene):
    drag_to_pan = False

    def construct(self) -> None:
        self.file_writer.crf = 1
        self.camera.background_rgba = color_to_rgba("#101010", 1.0)
        if self.window:
            glfw.set_window_monitor(self.window.glfw_window, glfw.get_primary_monitor(), 0, 0, 1920, 1080, glfw.DONT_CARE)
        datacut = TypstText("*DataCut*", font="New Computer Modern", font_size=192)
        datacut.add_vfx(Glow(radius=12.5, intensity=0.75))
        datacut.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=datacut, auto_direction=True))
        self.play(FadeIn(datacut, run_time=1.0, shift=2 * UP, rate_func=ease_out_cubic))
        self.play(datacut["Data"].animate(rate_func=smoothstep, run_time=0.5).set_color(YELLOW))
        self.wait()
        self.play(FadeOut(datacut, run_time=0.5, shift=2 * DOWN, rate_func=smoothstep))
        words = [
            "Editor",
            "de",
            "videos",
            "y",
            "presentador",
            "de",
            "livestreams",
            "_data-driven_"
        ]
        renderiza = TypstText(" ".join(words), font="New Computer Modern", font_size=192)
        renderiza.add_vfx(Glow(radius=12.5, intensity=0.75))
        words_group = VGroup()
        seen = defaultdict(int)
        for word in words:
            word_mob = select_word(renderiza, word, index=seen[word])
            seen[word] += 1
            word_mob.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=word_mob, auto_direction=True))
            words_group.add(word_mob)
        renderiza.shift(-words_group[0].get_center())
        last_word_center = words_group[-1].get_center()
        self.play(LaggedStart(*(FadeIn(word, run_time=1.0, shift=2 * UP, rate_func=ease_out_cubic) for word in words_group), lag_ratio=0.5, run_time=3.0), self.camera.frame.animate(run_time=4.0, rate_func=smoothstep).move_to(last_word_center))
        self.wait()
        self.play(self.camera.frame.animate(rate_func=smoothstep).move_to(ORIGIN), FadeOut(renderiza, run_time=1.0, rate_func=smoothstep))
        self.wait()
        # Linear regression
        x_range = [0, 16, 1]
        y_range = [0, 8, 1]
        self.ax = Axes(x_range=x_range, y_range=y_range, width=12, height=6)
        self.ax.add_coordinate_labels()
        self.ax.add_vfx(Glow(radius=12.5, intensity=0.75))
        desired_slope = 0.35
        desired_intercept = 0.5
        noise_level = 0.5
        x = np.clip(np.linspace(x_range[0], x_range[1], 50) + np.random.normal(-noise_level, noise_level, 50), x_range[0], x_range[1])
        y = np.clip(desired_slope * x + desired_intercept + np.random.normal(-noise_level, noise_level, 50), y_range[0], y_range[1])
        points = VGroup(*[Dot(self.ax.c2p(x[i], y[i]), radius=0.04, color=YELLOW) for i in range(len(x))])
        points.add_vfx(Glow(radius=20.0, intensity=1.0))
        self.play(Write(self.ax, run_time=2.0), LaggedStart(*(FadeIn(point, rate_func=ease_out_cubic) for point in points), lag_ratio=0.1, run_time=2.0, group=points))
                # --- Internal Mechanism of Polyfit (OLS) ---
        # 1. Compute target parameters
        self.target_slope, self.target_intercept = np.polyfit(x, y, 1)
        self.slope_track = LinearNumberSlider(value=0.0, min_value=-1.0, step=0.01, max_value=1.0).set_color(YELLOW)
        self.intercept_track = LinearNumberSlider(value=0.0, min_value=-1.0, step=0.01, max_value=1.0).set_color(BLUE)
        dec_m = DecimalNumber(0.0).set_color(YELLOW).fix_in_frame()
        dec_m.add_updater(lambda d: d.set_value(self.slope_track.get_value()).add_vfx(Glow(radius=12.5, intensity=0.75)).fix_in_frame())
        dec_m.add_vfx(Glow(radius=12.5, intensity=0.75))
        dec_b = DecimalNumber(0.0).set_color(BLUE).fix_in_frame()
        dec_b.add_updater(lambda d: d.set_value(self.intercept_track.get_value()).add_vfx(Glow(radius=12.5, intensity=0.75)).fix_in_frame())
        dec_b.add_vfx(Glow(radius=12.5, intensity=0.75))
        g_m = Group(self.slope_track, dec_m).arrange(RIGHT).to_corner(DL)
        g_b = Group(self.intercept_track, dec_b).arrange(RIGHT).to_corner(DR)

        formula = Typst("y = m x + b", font_size=72).fix_in_frame()
        formula.add_vfx(Glow(radius=12.5, intensity=0.75))
        formula["m"].set_color(YELLOW)
        formula["b"].set_color(BLUE)
        formula.to_edge(UP)
        # 3. Create the dynamic line
        self.reg_line = self.ax.get_graph(
            lambda x_val: self.slope_track.get_value() * x_val + self.intercept_track.get_value(),
            stroke_color=PURE_RED,
            x_range=[x_range[0], x_range[1]]
        ).add_vfx(Glow(radius=15.0, intensity=1.2))

        # 5. Display initial bad guess and the error residuals
        self.play(ShowCreation(self.reg_line), run_time=1.0)
        self.wait(1.0)

        # 6. Animate optimization (Simulating the algebraic/gradient descent convergence)
        self.slope_track.add_vfx(Glow(radius=12.5, intensity=0.75))
        self.intercept_track.add_vfx(Glow(radius=12.5, intensity=0.75))
        self.play(FadeIn(formula, shift=UP, rate_func=ease_out_cubic), FadeIn(g_m, shift=UP, rate_func=ease_out_cubic), FadeIn(g_b, shift=UP, rate_func=ease_out_cubic), run_time=1.0)
        self.reg_line.add_updater(lambda m: self.update_regression_line())

    def on_key_press(self, symbol, modifiers):
        if chr(symbol) == "o":
            self.play(self.slope_track.animate.set_value(self.target_slope), self.intercept_track.animate.set_value(self.target_intercept), run_time=1.0, rate_func=smoothstep)
        super().on_key_press(symbol, modifiers)

    def update_regression_line(self):
        new_slope = self.slope_track.get_value()
        new_intercept = self.intercept_track.get_value()
        new_line = self.ax.get_graph(
            lambda x_val: new_slope * x_val + new_intercept,
            stroke_color=PURE_RED,
            x_range=[0, 16]
        ).add_vfx(Glow(radius=15.0, intensity=1.2))
        self.reg_line.become(new_line)


class TestMultipleMasking(Scene):
    def construct(self):
        self.file_writer.crf = 1

        # Mobject objetivo con resplandor
        grid = NumberPlane(
            x_range=[-8, 8, 1],
            y_range=[-5, 5, 1],
            background_line_style={"stroke_color": TEAL, "stroke_width": 2}
        )
        grid.add_vfx(Glow(radius=25.0, intensity=1.6))

        # Máscara 1: Contenedor circular exterior (intersección)
        outer_circle = Circle(radius=2.6).move_to(ORIGIN)

        # Máscara 2: Perforación interior cuadrada (sustracción)
        inner_hole = Square(side_length=1.2).move_to(LEFT * 1.5)

        # Aplicamos ambas máscaras encadenadas
        grid.add_mask(outer_circle, mode="intersect")
        grid.add_mask(inner_hole, mode="subtract")

        self.add(grid)

        # Animamos el agujero interior desplazándolo de izquierda a derecha dentro del círculo
        self.play(
            inner_hole.animate.shift(RIGHT * 3),
            run_time=2.5,
            rate_func=smooth
        )
        self.wait(1)


class TestLiquidGlass(Scene):
    def construct(self):
        # ---- Fondo: es lo que el vidrio refracta y desenfoca ----
        plane = NumberPlane(background_line_style={"stroke_color": TEAL, "stroke_width": 2})
        title = Text("Liquid Glass", font_size=140).set_color_by_gradient(BLUE_B, PINK)
        dots = VGroup(*[
            Circle(radius=0.55).set_fill(c, opacity=0.9).set_stroke(width=0)
            for c in (RED, YELLOW, GREEN, TEAL)
        ]).arrange(RIGHT, buff=1.4).to_edge(DOWN, buff=0.8)

        # ---- Vidrio ----
        # Necesita algo visible (trazo o relleno tenue), si no el renderer
        # no genera drawings y el efecto no se dispara.
        glass = RoundedRectangle(width=3.2, height=2.2, corner_radius=0.6)
        glass.set_fill(WHITE, opacity=0)
        glass.set_stroke(WHITE, width=1.5, opacity=0.35)
        glass.set_liquid_glass(
            power=3.0,
            blur_radius=0.0001,
            noise=0.0,
        )
        glass.add_vfx(Glow(radius=1.0, intensity=0.4))
        glass.move_to(4.5 * LEFT)

        # El vidrio se añade al final: solo refracta lo dibujado antes que él
        self.add(plane, title, dots, glass)
        self.wait(0.5)

        # 1) Se desplaza sobre el texto
        self.play(glass.animate.move_to(4.5 * RIGHT), run_time=4)

        # 2) Cambia de tamaño (el vidrio sigue al bounding box del mobject)
        self.play(glass.animate.scale(1.5).move_to(ORIGIN), run_time=2)


class TestDropShadow(Scene):
    def construct(self):
        # Fondo con contraste
        bg = Rectangle(width=16, height=9, fill_color="#1e1e2e", fill_opacity=1.0)
        self.add(bg)

        # Texto con sombra proyectada hacia abajo y derecha
        tex = Typst(r"nabla times EE = -(partial bold(B)) / (partial t)", font_size=72)
        tex.set_color(WHITE)
        tex.set_drop_shadow(offset=(8.0, -8.0), radius=16.0, opacity=0.85, color=BLACK)

        # Polígono vectorial con sombra coloreada (estilo 'ambient occlusion')
        sq = Square(side_length=2.5, fill_color=TEAL, fill_opacity=1.0, stroke_width=0)
        sq.shift(LEFT * 3)
        sq.set_drop_shadow(offset=(10.0, -10.0), radius=24.0, opacity=0.9, color="#051015")

        self.play(FadeIn(sq), Write(tex))
        self.wait()


def ease_in_cubic(t: float) -> float:
    """
    Cubic ease-in function.
    :param t: Current progress from 0.0 to 1.0
    :return: Eased progress from 0.0 to 1.0
    """
    # Clamp t to ensure it stays between 0 and 1
    t = max(0.0, min(1.0, t))
    return t * t * t


class QuantumHUDScene(Scene):
    def construct(self):
        # 1. FONDO CON CONTRASTE (imprescindible para ver refracción)
        bg_plane = Rectangle(width=16, height=9, fill_color="#0d111a", fill_opacity=1.0)
        bg_plane.set_vignette(radius=0.45, softness=0.5, intensity=0.7)

        # Rejilla luminosa de coordenadas que se distorsiona con el cristal
        grid = NumberPlane(
            x_range=(-8, 8, 1),
            y_range=(-5, 5, 1),
            background_line_style={"stroke_color": "#223854", "stroke_width": 1.5, "stroke_opacity": 0.8},
            axis_config={"stroke_color": "#3d6494", "stroke_width": 2.5},
        )
        # Círculos concéntricos de telemetría de fondo para evidenciar la curvatura de la lente
        radar_rings = VGroup(*[
            Circle(radius=r, stroke_color="#1c304a", stroke_width=1.2, stroke_opacity=0.6)
            for r in [1.5, 2.5, 3.5, 4.5]
        ])
        self.add(bg_plane, grid, radar_rings)

        # 2. PANEL DE VIDRIO LÍQUIDO (OverShifted auténtico)
        hud_glass = RoundedRectangle(
            width=7.5,
            height=4.2,
            corner_radius=0.4,
            stroke_width=0,  # Sin trazo artificial: el bisel lo genera el shader
        ).scale(1 / 0.95)
        liquid_glass = LiquidGlass(
            blur_radius=1.5,           # Desenfoque leve para no destruir las líneas
            blur_iterations=1,
            noise=0.01,
            glow_weight=0.35,          # Brillo angular original
            show_mobject=False,
        )
        hud_glass.add_vfx(liquid_glass)
        self.play(hud_glass.animate.scale(0.95), VFXAnimation(Mobject(), liquid_glass, lambda t: {"opacity": t}, run_time=1.0))

        # 3. IMPACTO DE ALTA VELOCIDAD
        probe = Dot(point=LEFT * 9 + UP * 3, radius=0.18, color=YELLOW_A)
        probe.add_vfx(MotionBlur(strength=2.2, auto_direction=True, max_blur=0.35))

        self.add(probe)
        self.play(
            probe.animate.move_to(hud_glass.get_center()),
            run_time=0.55,
            rate_func=rush_into,
        )

        # 4. ECUACIÓN CON GLOW Y SOMBRA
        equation = Tex(r"\mathbf{S} = \frac{1}{\mu_0} (\mathbf{E} \times \mathbf{B})", font_size=56)
        equation.move_to(hud_glass.get_center() + UP * 0.4)
        equation.set_drop_shadow(offset=(4.0, -5.0), radius=14.0, opacity=0.9, color=BLACK)
        equation.add_vfx(Glow(radius=20.0, intensity=1.5, color=TEAL_D))

        self.remove(probe)
        self.play(FadeIn(equation, scale=1.05), run_time=0.6)

        # 5. ANOMALÍA TRANSITORIA
        glitch_fx = Glitch(intensity=0.75, speed=16.0, slice_height=16.0)
        ca_fx = ChromaticAberration(offset=0.03)

        equation.add_vfx(glitch_fx)
        equation.add_vfx(ca_fx)
        self.wait(0.5)

        equation.remove_vfx(glitch_fx)
        equation.remove_vfx(ca_fx)
        self.wait(0.3)

        # 6. ESCANEO CON MÁSCARA
        status_label = Text("FLUX CONVERGENCE: CRITICAL", font="Monospace", font_size=19, color=RED_B)
        status_label.next_to(equation, DOWN, buff=0.55)
        status_label.set_drop_shadow(offset=(2.0, -3.0), radius=8.0, opacity=0.85)

        scan_curtain = Rectangle(
            width=0.01,
            height=0.7,
            fill_color=WHITE,
            fill_opacity=1.0,
            stroke_width=0,
        )
        scan_curtain.move_to(status_label.get_left(), aligned_edge=LEFT)
        status_label.set_mask(scan_curtain)

        self.add(status_label)
        self.play(
            scan_curtain.animate.stretch_to_fit_width(status_label.get_width() + 0.3, about_edge=LEFT),
            run_time=0.9,
            rate_func=linear,
        )
        self.wait(0.5)
        status_label.remove_mask()
        self.remove(scan_curtain)

        # 7. SALTO ZOOM BLUR
        focal_cluster = Group(hud_glass, equation, status_label)
        focal_cluster.add_vfx(ZoomBlur(strength=2.2, auto_scale=True, mode="outward"))

        self.play(FadeOut(focal_cluster, scale=5.0), VFXAnimation(Mobject(), liquid_glass, lambda t: {"opacity": 1-t}), run_time=0.6, rate_func=rush_into)


class Intro(Scene):
    def construct(self):
        self.file_writer.crf = 1
        if self.window:
            glfw.set_window_monitor(self.window.glfw_window, glfw.get_primary_monitor(), 0, 0, 1920, 1080, glfw.DONT_CARE)
        self.camera.background_rgba = color_to_rgba("#101010", 1.0)
        words = [
            "Matemáticas",
            "con",
            "un",
            "random"
        ]
        txt = TypstText(" ".join(words), font="New Computer Modern", font_size=192)
        txt.add_vfx(Glow(radius=25, intensity=1))
        word_mobs = VGroup()
        for word in words:
            word_mob = select_word(txt, word)
            word_mob.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=word_mob, auto_direction=True))
            word_mobs.add(word_mob)
        txt.shift(-word_mobs[0].get_center())
        last_word_center = word_mobs[-1].get_center()
        self.play(LaggedStart(*(FadeIn(word, shift=2 * UP, rate_func=ease_out_cubic) for word in word_mobs), group=word_mobs, run_time=1.5, lag_ratio=0.5), self.camera.frame.animate(run_time=1.5, rate_func=smoothstep).move_to(last_word_center))
        self.wait(0.5)
        self.play(self.camera.frame.animate(rate_func=smoothstep).scale(2).move_to(txt))
        txt.add_vfx(Glitch(intensity=0.375, speed=8.0, slice_height=8.0))
        self.wait()
        self.play(FadeOut(txt, run_time=1.0, rate_func=ease_out_cubic))


class WebcamScene(Scene):
    def construct(self):
        webcam = VideoMobject(
            "/dev/video0",
            height=5,
            live=True,
            device_format="v4l2",
            device_options={
                "video_size": "1280x720",
                "framerate": "30",
            },
        )
        webcam.play_from()

        self.add(webcam)


import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import mediapipe as mp

from manimlib import *


# ============================================================
# MediaPipe
# ============================================================

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ============================================================
# Configuración
# ============================================================

HAND_MODEL_PATH = Path("hand_landmarker.task")

MAX_HANDS = 2

MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


# ============================================================
# Landmarks de MediaPipe
# ============================================================

WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


# ============================================================
# Conexiones de la mano
#
# MediaPipe Hand Landmarker entrega 21 landmarks.
# Mantenemos explícitamente la topología para poder dibujar
# nuestro propio esqueleto en Manim.
# ============================================================

HAND_CONNECTIONS = (
    # Pulgar
    (WRIST, THUMB_CMC),
    (THUMB_CMC, THUMB_MCP),
    (THUMB_MCP, THUMB_IP),
    (THUMB_IP, THUMB_TIP),

    # Índice
    (WRIST, INDEX_MCP),
    (INDEX_MCP, INDEX_PIP),
    (INDEX_PIP, INDEX_DIP),
    (INDEX_DIP, INDEX_TIP),

    # Medio
    (INDEX_MCP, MIDDLE_MCP),
    (MIDDLE_MCP, MIDDLE_PIP),
    (MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_DIP, MIDDLE_TIP),

    # Anular
    (MIDDLE_MCP, RING_MCP),
    (RING_MCP, RING_PIP),
    (RING_PIP, RING_DIP),
    (RING_DIP, RING_TIP),

    # Meñique
    (RING_MCP, PINKY_MCP),
    (PINKY_MCP, PINKY_PIP),
    (PINKY_PIP, PINKY_DIP),
    (PINKY_DIP, PINKY_TIP),

    # Palma
    (WRIST, PINKY_MCP),
)


# ============================================================
# Resultado independiente del callback de MediaPipe
# ============================================================

@dataclass
class HandDetection:
    landmarks: np.ndarray
    world_landmarks: Optional[np.ndarray]
    handedness: str
    handedness_score: float
    timestamp_ms: int


# ============================================================
# MediaPipe Hand Landmarker
#
# IMPORTANTE:
#
# El callback de MediaPipe NO toca ningún Mobject de Manim.
# Solamente actualiza datos protegidos por lock.
#
# Esto evita que el callback de MediaPipe y el renderer de
# Manim se pisen entre sí.
# ============================================================

class MediaPipeHandDetector:

    def __init__(
        self,
        model_path: str | Path,
        num_hands: int = MAX_HANDS,
        min_hand_detection_confidence: float = MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence: float = MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ):
        self.model_path = str(model_path)

        self.num_hands = num_hands

        self.min_hand_detection_confidence = (
            min_hand_detection_confidence
        )
        self.min_hand_presence_confidence = (
            min_hand_presence_confidence
        )
        self.min_tracking_confidence = (
            min_tracking_confidence
        )

        self._lock = threading.Lock()

        self._latest_results: list[HandDetection] = []
        self._latest_timestamp_ms = -1

        self._last_submitted_timestamp_ms = -1

        self._closed = False
        self._error: Optional[BaseException] = None

        self._landmarker = self._create_landmarker()

    # --------------------------------------------------------

    def _create_landmarker(self):

        base_options = mp_python.BaseOptions(
            model_asset_path=self.model_path,
        )

        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,

            running_mode=(
                mp_vision.RunningMode.LIVE_STREAM
            ),

            num_hands=self.num_hands,

            min_hand_detection_confidence=(
                self.min_hand_detection_confidence
            ),

            min_hand_presence_confidence=(
                self.min_hand_presence_confidence
            ),

            min_tracking_confidence=(
                self.min_tracking_confidence
            ),

            result_callback=self._on_result,
        )

        return mp_vision.HandLandmarker.create_from_options(
            options
        )

    # --------------------------------------------------------

    def _on_result(
        self,
        result,
        output_image,
        timestamp_ms: int,
    ):
        """
        Callback ejecutado por MediaPipe.

        NO modificar Manim aquí.
        """

        try:
            detections = []

            for hand_index, landmarks in enumerate(
                result.hand_landmarks
            ):

                points = np.array(
                    [
                        [landmark.x, landmark.y]
                        for landmark in landmarks
                    ],
                    dtype=np.float32,
                )

                world_points = None

                if (
                    result.hand_world_landmarks
                    and hand_index
                    < len(result.hand_world_landmarks)
                ):
                    world_points = np.array(
                        [
                            [
                                landmark.x,
                                landmark.y,
                                landmark.z,
                            ]
                            for landmark in (
                                result.hand_world_landmarks[
                                    hand_index
                                ]
                            )
                        ],
                        dtype=np.float32,
                    )

                handedness = "Unknown"
                handedness_score = 0.0

                if (
                    result.handedness
                    and hand_index < len(result.handedness)
                ):
                    categories = result.handedness[hand_index]

                    if categories:
                        category = categories[0]

                        handedness = category.category_name
                        handedness_score = float(
                            category.score or 0.0
                        )

                detections.append(
                    HandDetection(
                        landmarks=points,
                        world_landmarks=world_points,
                        handedness=handedness,
                        handedness_score=handedness_score,
                        timestamp_ms=timestamp_ms,
                    )
                )

            with self._lock:
                self._latest_results = detections
                self._latest_timestamp_ms = timestamp_ms

        except BaseException as exc:
            with self._lock:
                self._error = exc

    # --------------------------------------------------------

    def submit(
        self,
        rgb_frame: np.ndarray,
        timestamp_ms: int,
    ) -> bool:
        if self._closed:
            return False

        # MediaPipe exige timestamps monótonamente crecientes.
        if timestamp_ms <= self._last_submitted_timestamp_ms:
            return False

        self._last_submitted_timestamp_ms = timestamp_ms

        try:
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_frame),
            )

            self._landmarker.detect_async(
                image,
                timestamp_ms,
            )

            return True

        except BaseException as exc:
            with self._lock:
                self._error = exc
            return False

    # --------------------------------------------------------

    def get_latest(self) -> list[HandDetection]:

        with self._lock:
            return list(self._latest_results)

    # --------------------------------------------------------

    @property
    def latest_timestamp_ms(self) -> int:

        with self._lock:
            return self._latest_timestamp_ms

    # --------------------------------------------------------

    @property
    def error(self):

        with self._lock:
            return self._error

    # --------------------------------------------------------

    def close(self):

        if self._closed:
            return

        self._closed = True

        self._landmarker.close()


# ============================================================
# Utilidades geométricas
# ============================================================

def distance(a, b):

    return float(
        np.linalg.norm(
            np.asarray(a) - np.asarray(b)
        )
    )


def angle(a, b, c):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
    )

    if denominator < 1e-8:
        return 180.0

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(
        np.degrees(np.arccos(cosine))
    )


# ============================================================
# Reconocimiento de dedos
# ============================================================

def finger_extended(
    points,
    mcp,
    pip,
    dip,
    tip,
):
    """
    Determina si un dedo está extendido utilizando los
    ángulos articulares.

    Esto es deliberadamente geométrico: no depende de un
    clasificador adicional.
    """

    pip_angle = angle(
        points[mcp],
        points[pip],
        points[dip],
    )

    dip_angle = angle(
        points[pip],
        points[dip],
        points[tip],
    )

    return (
        pip_angle > 155.0
        and dip_angle > 150.0
    )


def thumb_extended(points):

    return (
        angle(
            points[THUMB_CMC],
            points[THUMB_MCP],
            points[THUMB_IP],
        ) > 150.0
        and distance(
            points[THUMB_TIP],
            points[INDEX_MCP],
        )
        >
        distance(
            points[THUMB_IP],
            points[INDEX_MCP],
        )
    )


# ============================================================
# Gestos
# ============================================================

def classify_gesture(points):

    thumb = thumb_extended(points)

    index = finger_extended(
        points,
        INDEX_MCP,
        INDEX_PIP,
        INDEX_DIP,
        INDEX_TIP,
    )

    middle = finger_extended(
        points,
        MIDDLE_MCP,
        MIDDLE_PIP,
        MIDDLE_DIP,
        MIDDLE_TIP,
    )

    ring = finger_extended(
        points,
        RING_MCP,
        RING_PIP,
        RING_DIP,
        RING_TIP,
    )

    pinky = finger_extended(
        points,
        PINKY_MCP,
        PINKY_PIP,
        PINKY_DIP,
        PINKY_TIP,
    )

    extended = (
        thumb,
        index,
        middle,
        ring,
        pinky,
    )

    # --------------------------------------------------------
    # Palma abierta
    # --------------------------------------------------------

    if all(extended):
        return "OPEN_PALM"

    # --------------------------------------------------------
    # Puño
    # --------------------------------------------------------

    if not any(extended):
        return "FIST"

    # --------------------------------------------------------
    # Señalar
    # --------------------------------------------------------

    if (
        index
        and not middle
        and not ring
        and not pinky
    ):
        return "POINT"

    # --------------------------------------------------------
    # Paz
    # --------------------------------------------------------

    if (
        index
        and middle
        and not ring
        and not pinky
    ):
        return "PEACE"

    return "NONE"


# ============================================================
# Estabilización
# ============================================================

class GestureStabilizer:

    def __init__(
        self,
        required_frames: int = 4,
    ):
        self.required_frames = required_frames

        self.current = "NONE"
        self.candidate = None
        self.count = 0

    def update(self, gesture):

        if gesture == self.current:
            self.candidate = None
            self.count = 0
            return self.current

        if gesture != self.candidate:

            self.candidate = gesture
            self.count = 1

        else:

            self.count += 1

        if self.count >= self.required_frames:

            self.current = gesture
            self.candidate = None
            self.count = 0

        return self.current


# ============================================================
# Esqueleto de mano
# ============================================================

class HandSkeleton(VGroup):

    def __init__(
        self,
        webcam,
        flip=False,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.webcam = webcam
        self._flip = flip

        self.landmark_dots = VGroup(
            *[
                Dot(
                    radius=0.045,
                    color=YELLOW,
                )
                for _ in range(21)
            ]
        )

        self.bones = VGroup(
            *[
                Line(
                    ORIGIN,
                    ORIGIN,
                    stroke_width=4,
                    color=TEAL,
                )
                for _ in HAND_CONNECTIONS
            ]
        )

        self.add(
            self.bones,
            self.landmark_dots,
        )

    # --------------------------------------------------------

    def update_from_landmarks(
        self,
        points,
    ):

        if points is None:
            self.set_opacity(0)
            return

        self.set_opacity(1)

        pixel_width = self.webcam.source.width
        pixel_height = self.webcam.source.height

        scene_width = FRAME_WIDTH
        scene_height = FRAME_HEIGHT

        center = self.webcam.get_center()

        positions = []

        for x, y in points:

            # MediaPipe:
            #
            # x = 0 izquierda
            # x = 1 derecha
            # y = 0 arriba
            # y = 1 abajo

            px = (
                (x - 0.5)
                * scene_width
                + center[0]
            )

            py = (
                (0.5 - y)
                * scene_height
                + center[1]
            )

            positions.append(
                np.array(
                    [px, py, center[2] + 0.01]
                )
            )

        for dot, position in zip(
            self.landmark_dots,
            positions,
        ):
            dot.move_to(position)

        for line, (a, b) in zip(
            self.bones,
            HAND_CONNECTIONS,
        ):
            line.put_start_and_end_on(
                positions[a],
                positions[b],
            )
        if self._flip:
            self.flip(UP, about_point=self.webcam.get_center())


# ============================================================
# Etiqueta del gesto
# ============================================================

class GestureLabel(Text):

    def __init__(self, **kwargs):

        super().__init__(
            "NONE",
            font_size=32,
            **kwargs,
        )

        self.to_edge(UP)

    def set_gesture(self, gesture):

        self.set_text(
            gesture.replace("_", " ")
        )

        return self

    def set_text(self, text):

        self.become(
            Text(
                text,
                font_size=32,
            ).to_edge(UP)
        )

        return self


# ============================================================
# Escena
# ============================================================

class HandGestureScene(Scene):

    def construct(self):
        self.file_writer.crf = 1
        if self.window:
            glfw.set_window_monitor(self.window.glfw_window, glfw.get_primary_monitor(), 0, 0, 1920, 1080, glfw.DONT_CARE)
        self.camera.background_rgba = color_to_rgba("#101010", 1.0)

        # ====================================================
        # Webcam
        #
        # Utiliza el VideoMobject live que ya implementamos.
        # ====================================================

        webcam = VideoMobject(
            "/dev/video0",
            live=True,
            device_format="v4l2",
        ).flip(UP)

        webcam.set_width(FRAME_WIDTH)
        webcam.play_from()
        webcam.move_to(ORIGIN)

        self.add(webcam)

        # ====================================================
        # MediaPipe
        # ====================================================

        hand_detector = MediaPipeHandDetector(
            HAND_MODEL_PATH,
            num_hands=MAX_HANDS,
        )

        # ====================================================
        # Esqueleto
        # ====================================================

        skeleton = HandSkeleton(
            webcam,
            flip=True
        )
        skeleton.add_vfx(Glow(radius=18.75, intensity=1.125))

        self.add(skeleton)

        # ====================================================
        # UI
        # ====================================================

        label = GestureLabel()

        self.add(label)

        # ====================================================
        # Objeto controlado
        # ====================================================

        target = Circle(
            radius=0.6,
            stroke_color=WHITE,
        )
        target.add_vfx(Glow(radius=18.75, intensity=1.125))

        target.move_to(
            2.5 * RIGHT
        )

        self.add(target)

        # ====================================================
        # Estado
        # ====================================================

        stabilizer = GestureStabilizer(
            required_frames=4,
        )

        last_camera_frame = -1

        last_detection_timestamp = -1

        current_gesture = "NONE"

        # ====================================================
        # Webcam -> MediaPipe
        #
        # Esto ocurre en el updater de Manim pero la llamada
        # detect_async() NO bloquea esperando el resultado.
        #
        # MediaPipe procesa en su propio pipeline.
        # ====================================================

        def submit_frame(_):

            nonlocal last_camera_frame

            try:

                frame_index = (
                    webcam.source.latest_index
                )

                if frame_index == last_camera_frame:
                    return

                last_camera_frame = frame_index

                frame = webcam.get_pixels()

                if frame is None:
                    return

                # --------------------------------------------
                # Aseguramos RGB uint8
                # --------------------------------------------

                if frame.dtype != np.uint8:

                    if np.issubdtype(
                        frame.dtype,
                        np.floating,
                    ):
                        frame = (
                            np.clip(
                                frame,
                                0,
                                1,
                            )
                            * 255
                        ).astype(np.uint8)

                    else:

                        frame = frame.astype(
                            np.uint8
                        )

                # --------------------------------------------
                # RGBA -> RGB
                # --------------------------------------------

                if frame.ndim == 3:

                    if frame.shape[2] == 4:

                        frame = frame[:, :, :3]

                    elif frame.shape[2] != 3:

                        return

                else:

                    return

                # --------------------------------------------
                # Timestamp monotónico
                # --------------------------------------------

                timestamp_ms = int(
                    time.monotonic() * 1000
                )

                hand_detector.submit(
                    frame,
                    timestamp_ms,
                )

            except Exception:
                # Nunca dejamos que un problema de la webcam
                # mate el renderer de Manim.
                pass

        # ====================================================
        # Resultado MediaPipe -> Manim
        # ====================================================

        def update_scene(_):

            nonlocal current_gesture
            nonlocal last_detection_timestamp

            try:

                error = hand_detector.error

                if error is not None:
                    raise error

                detections = (
                    hand_detector.get_latest()
                )

                if not detections:
                    skeleton.set_opacity(0)
                    label.set_gesture("NONE")
                    current_gesture = "NONE"
                    return

                # --------------------------------------------
                # Por ahora controlamos la primera mano.
                # La arquitectura ya permite múltiples manos.
                # --------------------------------------------

                hand = detections[0]

                if (
                    hand.timestamp_ms
                    == last_detection_timestamp
                ):
                    return

                last_detection_timestamp = (
                    hand.timestamp_ms
                )

                points = hand.landmarks

                # --------------------------------------------
                # Esqueleto
                # --------------------------------------------

                skeleton.update_from_landmarks(
                    points
                )

                # --------------------------------------------
                # Gesto
                # --------------------------------------------

                raw_gesture = classify_gesture(
                    points
                )

                gesture = stabilizer.update(
                    raw_gesture
                )

                if gesture != current_gesture:

                    current_gesture = gesture

                    label.set_gesture(
                        gesture
                    )

                # --------------------------------------------
                # Acciones
                # --------------------------------------------

                if gesture == "OPEN_PALM":
                    target.set_width(1.5)
                    target.set_height(1.5)

                elif gesture == "FIST":
                    target.set_width(0.7)
                    target.set_height(0.7)

                elif gesture == "POINT":

                    # Índice -> fingertip = landmark 8

                    x, y = points[INDEX_TIP]

                    target_x = (
                        (x - 0.5)
                        * webcam.get_width()
                        + webcam.get_center()[0]
                    )

                    target_y = (
                        (0.5 - y)
                        * webcam.get_height()
                        + webcam.get_center()[1]
                    )

                    target.move_to(
                        np.array(
                            [
                                target_x,
                                target_y,
                                target.get_center()[2],
                            ]
                        )
                    )
                    target.flip(UP, about_point=webcam.get_center())

                elif gesture == "PEACE":

                    target.rotate(
                        0.03
                    )

            except Exception:
                # El updater visual jamás debe tumbar la escena
                # por un frame corrupto o un resultado inválido.
                pass

        # ====================================================
        # Orden de actualización
        # ====================================================

        webcam.add_updater(
            submit_frame
        )

        skeleton.add_updater(
            update_scene
        )

        # ====================================================
        # Ejecutar
        # ====================================================

        # try:

        #     self.wait(60)

        # finally:

        #     webcam.clear_updaters()
        #     skeleton.clear_updaters()

        #     hand_detector.close()

        #     try:
        #         webcam.source.close()
        #     except Exception:
        #         pass
        self.webcam = webcam
        self.hand_detector = hand_detector
        self.skeleton = skeleton

    def interact(self):
        try:
            super().interact()
        finally:
            self.webcam.clear_updaters()
            self.skeleton.clear_updaters()

            self.hand_detector.close()

            try:
                self.webcam.source.close()
            except Exception:
                pass


class Probability(InteractiveScene):
    samples = 4

    def construct(self):
        self.file_writer.crf = 1
        if self.window:
            glfw.set_window_monitor(self.window.glfw_window, glfw.get_primary_monitor(), 0, 0, 1920, 1080, glfw.DONT_CARE)
        self.camera.background_rgba = color_to_rgba("#101010", 1.0)
        dice: Group = ThreeDModel("Dice.obj")[4:6]
        dice.scale(0.5)
        dice.set_shading(0.2, 0.2, 0.2)
        self.add(dice)
        faces_pointing = {
            6: OUT,
            3: DOWN,
            1: IN,
            4: UP,
            5: LEFT,
            2: RIGHT,
        }
        # txt = TypstText("Probabilidad", font="New Computer Modern", font_size=96).to_edge(UP)
        # txt.apply_depth_test(anti_alias_width=1.5)
        # txt.add_vfx(Glow(radius=18.75, intensity=1.125))
        # self.add(txt)
        # dice.add_vfx(MotionBlur(strength=1, max_blur=0.15, auto_direction=True, mobject=dice))
        iterations = 5
        self.camera.frame.set_euler_angles(phi=-15 * DEGREES)
        for _ in range(iterations):
            shift_vector = RIGHT * 3 + OUT * 0.5
            dice.move_to(-shift_vector + DOWN)
            self.play(RollDice(dice, faces_pointing, target_position=ORIGIN))


class IntroProbability(Scene):
    def construct(self):
        self.file_writer.crf = 1
        if self.window:
            glfw.set_window_monitor(self.window.glfw_window, glfw.get_primary_monitor(), 0, 0, 1920, 1080, glfw.DONT_CARE)
        self.camera.background_rgba = color_to_rgba("#101010", 1.0)
        video1 = VideoMobject("videos/Probability.mp4", height=FRAME_HEIGHT)
        video2 = VideoMobject("patient.mp4", height=FRAME_HEIGHT)
        video3 = VideoMobject("call_center.mp4", height=FRAME_HEIGHT)
        video4 = VideoMobject("rain.mp4", height=FRAME_HEIGHT)
        self.add(video1)
        self.add_sound("intro.wav")
        video1.play_from()
        self.wait(2)
        video1.clear_updaters()
        self.remove(video1)
        self.add(video2)
        video2.play_from()
        self.wait(2.5)
        self.remove(video2)
        video2.clear_updaters()
        self.add(video3)
        video3.play_from()
        self.wait(3.5)
        self.remove(video3)
        video3.clear_updaters()
        self.add(video4)
        video4.play_from()
        self.wait(3)
        self.remove(video4)
        video4.clear_updaters()
        experimentos_aleatorios = TypstText("Experimentos aleatorios", font="New Computer Modern", font_size=96)
        experimentos_aleatorios.add_vfx(Glow(radius=18.75, intensity=1.125))
        words = ["Experimentos", "aleatorios"]
        word_mobs = VGroup()
        for word in words:
            word_mob = select_word(experimentos_aleatorios, word)
            word_mob.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=word_mob, auto_direction=True))
            word_mobs.add(word_mob)
        self.play(LaggedStart(*(FadeIn(word, shift=2 * UP, rate_func=ease_out_cubic) for word in word_mobs), group=word_mobs, run_time=2.0, lag_ratio=0.5))
        self.play(Indicate(experimentos_aleatorios, scale_factor=1, run_time=1))
        self.play(FadeOut(experimentos_aleatorios, run_time=1.0, rate_func=ease_out_cubic))


class EspacioMuestral(Scene):
    samples = 4

    def construct(self):
        self.file_writer.crf = 1
        if self.window:
            glfw.set_window_monitor(self.window.glfw_window, glfw.get_primary_monitor(), 0, 0, 1920, 1080, glfw.DONT_CARE)
        self.camera.background_rgba = color_to_rgba("#101010", 1.0)
        self.add_sound("espacio_muestral.wav")
        dice = ThreeDModel("Dice.obj")[4:6]
        dice.scale(0.5)
        dice.set_shading(0.2, 0.2, 0.2)
        faces_pointing = {
            6: OUT,
            3: DOWN,
            1: IN,
            4: UP,
            5: LEFT,
            2: RIGHT,
        }
        self.camera.frame.set_euler_angles(phi=-30 * DEGREES)
        dice_copy = dice.copy()
        squares = VGroup(*(Square(side_length=1, stroke_color=WHITE) for _ in range(6))).fix_in_frame()
        squares.add_vfx(Glow(radius=12.5, intensity=0.75))
        squares.arrange(RIGHT, buff=0)
        squares.to_edge(UP)
        txts = VGroup(*(Typst(str(i), font="New Computer Modern", font_size=48).move_to(squares[i - 1]) for i in range(1, 7))).fix_in_frame()
        txts.add_vfx(Glow(radius=12.5, intensity=0.75))
        for i in range(1, 7):
            dice.become(dice_copy)
            dice.move_to(-RIGHT * 3 + DOWN)
            animations = [
                RollDice(dice, faces_pointing, outcome=i, target_position=ORIGIN, run_time=0.5),
                ShowCreation(squares[i - 1], run_time=0.5)
            ]
            self.play(*animations)
            txts[i - 1].add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=txts[i - 1], auto_direction=True))
            txts[i - 1].move_to(dice.get_center() + UP * 0.5)
            txts[i - 1].set_opacity(0)
            self.play(txts[i - 1].animate.set_opacity(1).move_to(squares[i - 1]), run_time=0.5, rate_func=ease_out_cubic)
        espacio_muestral = TypstText("Espacio muestral", font="New Computer Modern", font_size=96).fix_in_frame()
        espacio_muestral.add_vfx(Glow(radius=18.75, intensity=1.125))
        word_mobs = VGroup()
        for word in ["Espacio", "muestral"]:
            word_mob = select_word(espacio_muestral, word)
            word_mob.add_vfx(MotionBlur(strength=1.5, max_blur=0.3, mobject=word_mob, auto_direction=True))
            word_mobs.add(word_mob)
        as_set = Typst("Omega = {1, 2, 3, 4, 5, 6}", font="New Computer Modern", font_size=72).fix_in_frame()
        as_set.add_vfx(Glow(radius=12.5, intensity=0.75))
        anims = []
        Group(espacio_muestral, as_set).arrange(DOWN)
        for i in range(6):
            target = as_set[str(i + 1)]
            # anims.append(ReplacementTransform(txts[i], target))
            anims.append(txts[i].animate.replace(target, stretch=True))
        anims2 = [FadeIn(as_set["Omega"]), FadeIn(as_set["="]), FadeIn(as_set["{"]), FadeIn(as_set["}"]), FadeIn(as_set[","])]
        self.play(dice.animate(rate_func=ease_in_cubic).shift(8 * DOWN), LaggedStart(*(FadeIn(word, shift=2 * UP, rate_func=ease_out_cubic) for word in word_mobs), group=word_mobs, run_time=2.0, lag_ratio=0.5), FadeOut(squares, run_time=1.0, rate_func=ease_out_cubic), LaggedStart(*anims, lag_ratio=0.25, run_time=2.0), *anims2)
        self.remove(dice)
        self.wait()

    def add(self, *mobjects: Mobject, set_depth_test: bool = True):
        for mob in mobjects:
            # Asked of each member of the family in turn, rather than of what holds them.
            # A group has a say of its own on being fixed in frame, which is nothing to do
            # with what it holds, and animations wrap what they are given in a fresh one,
            # so text fixed in frame would otherwise find itself depth tested for the
            # length of a FadeTransform, and vanish behind whatever it was written over.
            if set_depth_test:
                for sm in mob.get_family():
                    if not sm.is_fixed_in_frame():
                        if isinstance(sm, VMobject):
                            sm.apply_depth_test(anti_alias_width=1.5, recurse=False)
                        else:
                            sm.apply_depth_test(recurse=False)
        super().add(*mobjects)