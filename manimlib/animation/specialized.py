from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from manimlib.animation.animation import Animation
from manimlib.animation.composition import LaggedStart
from manimlib.animation.transform import Restore
from manimlib.constants import BLACK, WHITE
from manimlib.mobject.geometry import Circle
from manimlib.mobject.types.vectorized_mobject import VGroup

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manimlib.typing import ManimColor


class Broadcast(LaggedStart):
    def __init__(
        self,
        focal_point: np.ndarray,
        small_radius: float = 0.0,
        big_radius: float = 5.0,
        n_circles: int = 5,
        start_stroke_width: float = 8.0,
        color: ManimColor = WHITE,
        run_time: float = 3.0,
        lag_ratio: float = 0.2,
        remover: bool = True,
        **kwargs
    ):
        self.focal_point = focal_point
        self.small_radius = small_radius
        self.big_radius = big_radius
        self.n_circles = n_circles
        self.start_stroke_width = start_stroke_width
        self.color = color

        circles = VGroup()
        for x in range(n_circles):
            circle = Circle(
                radius=big_radius,
                stroke_color=BLACK,
                stroke_width=0,
            )
            circle.add_updater(lambda c: c.move_to(focal_point))
            circle.save_state()
            circle.set_width(small_radius * 2)
            circle.set_stroke(color, start_stroke_width)
            circles.add(circle)
        super().__init__(
            *map(Restore, circles),
            run_time=run_time,
            lag_ratio=lag_ratio,
            remover=remover,
            **kwargs
        )


# ----------------------------------------------------------------------
# Cuaterniones (w, x, y, z)
# ----------------------------------------------------------------------

def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * angle
    return np.array([np.cos(half), *(axis * np.sin(half))])


def _slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(min(d, 1.0))
    s = np.sin(theta)
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / s


def _smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


# Signos de las 8 esquinas del cubo, en el orden fijo que usa la simulación.
_SIGNS = tuple(itertools.product((-1.0, 1.0), repeat=3))


@dataclass
class _Throw:
    positions: np.ndarray    # (N, 3)
    quaternions: np.ndarray  # (N, 4)
    face: int
    settle_time: float       # segundos de simulación hasta quedar en reposo
    yaw_error: float = 0.0   # giro (rad) que hubo que corregir al final
    target_error: float = 0.0  # distancia (escena) que hubo que corregir al final


@dataclass
class _Candidate:
    """Resultado barato de una simulación; se "construye" solo si se elige."""
    positions: list
    quats: list
    activity: list
    q: np.ndarray            # orientación al quedar en reposo
    q_flat: np.ndarray       # orientación final (plana y alineada)
    x: np.ndarray            # centro al quedar en reposo
    face: int
    yaw: float               # giro (con signo) que se reparte en el movimiento
    residual: np.ndarray | None
    settle_time: float
    yaw_error: float
    target_error: float


class RollDice(Animation):
    """
    Lanzamiento de un dado con dinámica de cuerpo rígido real.

    Qué ocurre en la animación:

        1. El dado sale de su posición actual con velocidad lineal y
           angular (sale hacia arriba, girando, en una dirección).
        2. Vuela en parábola mientras da vueltas.
        3. Golpea la mesa por una esquina: impulso con restitución y
           fricción de Coulomb (el dado rebota y se le "come" la
           velocidad de forma natural).
        4. Rebota, rueda y se vuelca sobre aristas y caras.
        5. Se detiene (detección de reposo) y se alinea exactamente
           plano sobre la mesa.

    Nada está "guionizado": para que salga `outcome` se simulan
    lanzamientos aleatorios distintos y se reproduce el primero que
    termina de forma natural con esa cara arriba.

    Convención de ejes (manim): Y es arriba, la mesa es el plano
    y = ground_y.

    `faces_pointing` son las normales LOCALES de cada cara:

        faces_pointing = {
            6: OUT, 3: DOWN, 1: IN, 4: UP, 5: LEFT, 2: RIGHT,
        }

    Unidades: las velocidades y la gravedad se expresan en "tamaños de
    dado" (por segundo y por segundo²), así el resultado se ve igual
    sin importar la escala del modelo. Las velocidades angulares van
    en rad/s.

    Posiciones (coordenadas de la escena; la mesa es el plano XZ):

        start_position   Dónde empieza el lanzamiento (centro del dado).
                         Con 3 valores (x, y, z) fija también la altura de
                         suelta; con 2 valores (x, z) mantiene la altura
                         actual del dado. Por defecto: donde está el dado.
                         Ojo: en el primer frame el dado aparece ahí.
        target_position  Dónde termina el dado en la mesa (centro). Acepta
                         (x, z) o (x, y, z); la y se ignora. El dado se
                         lanza apuntando hacia ese punto y el pequeño error
                         que quede se reparte durante el movimiento, igual
                         que el giro, así que no hay "tirones" al final.
        target_tolerance Cuánto puede quedar de error, en tamaños de dado,
                         antes de que se descarte un lanzamiento (el error
                         que quede se corrige igualmente). Con `target_position`
                         se ignora `launch_direction`.

    Rendimiento:

        - La búsqueda del lanzamiento (al construir la animación) usa una
          simulación en escalares puros de Python, sin numpy por paso.
        - Solo se "construye" la trayectoria completa del lanzamiento
          elegido; los descartados se evalúan con unos pocos números.
        - Durante la animación se aplica al mobject solo la transformación
          incremental entre frames (una sola pasada por sus puntos) y no
          se hace nada cuando la pose no cambia (dado ya en reposo).

    Ejemplo:

        self.play(
            RollDice(dice, faces_pointing, outcome=5,
                     start_position=LEFT * 4,
                     target_position=RIGHT * 2 + OUT * 1.5)
        )
    """

    UP = np.array([0.0, 1.0, 0.0])
    _ZERO = np.zeros(3)

    SLEEP_TIME = 0.10
    SLEEP_TILT_COS = float(np.cos(np.radians(2.0)))
    BLEND_TIME = 0.18
    CONTACT_ITERATIONS = 10

    AIR_DAMPING_LINEAR = 0.05
    AIR_DAMPING_ANGULAR = 0.10
    GROUND_DAMPING_LINEAR = 0.6
    GROUND_DAMPING_ANGULAR = 1.2

    def __init__(
        self,
        dice,
        faces_pointing: dict[int, np.ndarray],
        outcome: int | None = None,
        run_time: float = 2.8,
        seed: int | None = None,

        # Física (en tamaños de dado)
        gravity: float = 22.0,
        restitution: float = 0.35,
        friction: float = 0.45,

        # Lanzamiento
        launch_direction: np.ndarray | None = None,
        launch_speed_range: tuple[float, float] = (2.0, 5.0),
        vertical_speed_range: tuple[float, float] = (7.5, 10.5),
        angular_speed_range: tuple[float, float] = (8.0, 18.0),

        # Mesa y límites
        ground_y: float | None = None,
        max_travel: float | None = None,

        # Simulación
        max_attempts: int = 400,
        simulation_dt: float = 1.0 / 180.0,
        settle_fraction: float = 0.92,

        # Reposo final alineado con los ejes de la escena (múltiplos de
        # 90° alrededor de Y). Se prefieren lanzamientos que ya terminan
        # a menos de `yaw_tolerance_deg` de esa alineación, para que el
        # ajuste final sea imperceptible.
        snap_yaw: bool = True,
        yaw_tolerance_deg: float = 25.0,

        # Posiciones inicial y final (ver docstring)
        start_position: np.ndarray | None = None,
        target_position: np.ndarray | None = None,
        target_tolerance: float = 1.0,

        **kwargs,
    ):
        self.dice = dice
        self.target_tolerance = float(target_tolerance)
        self.snap_yaw = bool(snap_yaw)
        self.yaw_tolerance_deg = float(yaw_tolerance_deg)

        self.faces_pointing = {
            int(face): self._normalize(np.asarray(normal, dtype=float))
            for face, normal in faces_pointing.items()
        }
        if len(self.faces_pointing) != 6:
            raise ValueError("`faces_pointing` debe contener exactamente 6 caras.")

        self._faces = list(self.faces_pointing.keys())
        self._normals = np.array([self.faces_pointing[f] for f in self._faces])
        self._normal_rows = [tuple(float(c) for c in n) for n in self._normals]

        self.rng = np.random.default_rng(seed)

        if outcome is None:
            outcome = int(self.rng.choice(self._faces))
        self.outcome = int(outcome)
        if self.outcome not in self.faces_pointing:
            raise ValueError(f"La cara {self.outcome} no existe en faces_pointing.")

        self.gravity_in_sizes = float(gravity)
        self.restitution = float(restitution)
        self.friction = float(friction)
        self.launch_speed_range = launch_speed_range
        self.vertical_speed_range = vertical_speed_range
        self.angular_speed_range = angular_speed_range
        self.max_travel = max_travel
        self.max_attempts = int(max_attempts)
        self.simulation_dt = float(simulation_dt)
        self.settle_fraction = float(settle_fraction)

        if launch_direction is not None:
            d = np.asarray(launch_direction, dtype=float).copy()
            d[1] = 0.0
            if np.linalg.norm(d) < 1e-8:
                raise ValueError("`launch_direction` debe tener componente horizontal.")
            self.launch_direction = self._normalize(d)
        else:
            self.launch_direction = None

        # --------------------------------------------------------------
        # Geometría del dado
        # --------------------------------------------------------------
        self._initial_dice = dice.copy()
        self._initial_center = np.asarray(dice.get_center(), dtype=float)

        bbox = np.asarray(dice.get_bounding_box(), dtype=float)
        extent = np.abs(bbox[2] - bbox[0])
        self._size = float(np.mean(extent))
        if self._size <= 1e-8:
            raise ValueError("No se pudo determinar el tamaño del dado.")
        self._half = self._size / 2.0

        self._ground_y = float(bbox[0][1] if ground_y is None else ground_y)

        # --------------------------------------------------------------
        # Posición inicial y objetivo
        # --------------------------------------------------------------
        if start_position is None:
            self._start = self._initial_center.copy()
        else:
            sp = np.asarray(start_position, dtype=float).ravel()
            if sp.size == 2:
                self._start = np.array([sp[0], self._initial_center[1], sp[1]])
            elif sp.size == 3:
                self._start = sp.copy()
            else:
                raise ValueError("`start_position` debe ser (x, z) o (x, y, z).")

        if target_position is None:
            self._target_xz = None
        else:
            tp = np.asarray(target_position, dtype=float).ravel()
            if tp.size == 2:
                self._target_xz = tp.copy()
            elif tp.size == 3:
                self._target_xz = np.array([tp[0], tp[2]])
            else:
                raise ValueError("`target_position` debe ser (x, z) o (x, y, z).")

        # Con objetivo, el límite de recorrido nunca debe impedir llegar.
        self._travel_limit = self.max_travel
        if self._target_xz is not None and self.max_travel is not None:
            dist = float(np.linalg.norm(self._target_xz - self._start[[0, 2]]))
            self._travel_limit = max(self.max_travel, 1.6 * dist + 2.0 * self._size)

        # Aprendizaje de "cuánto recorre el dado por cada unidad de
        # velocidad horizontal", para apuntar mejor en cada intento.
        self._ratios: list[float] = []
        self._last_launch = None

        # Cubo de masa 1: I = m s² / 6 (isótropo).
        self._inv_inertia = 6.0 / (self._size ** 2)
        self._gravity = self.gravity_in_sizes * self._size
        self._rest_threshold = max(
            0.4 * self._size, 5.0 * self._gravity * self.simulation_dt
        )

        # --------------------------------------------------------------
        # Buscar un lanzamiento físico que termine en `outcome`
        # --------------------------------------------------------------
        self._throw: _Throw | None = None
        self._time_scale = 1.0
        self._find_throw(run_time)

        # Pose actualmente aplicada al mobject (para la reproducción
        # incremental).
        self._cur_pos = self._initial_center.copy()
        self._cur_R = np.eye(3)
        self._fused_ok = True

        kwargs.setdefault("rate_func", linear)  # la física no se "suaviza"
        super().__init__(dice, run_time=run_time, **kwargs)

    # ==================================================================
    # Utilidades
    # ==================================================================

    @staticmethod
    def _normalize(v):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        if n < 1e-12:
            raise ValueError("No se puede normalizar un vector de longitud cero.")
        return v / n

    # ==================================================================
    # Estado inicial del lanzamiento
    # ==================================================================

    def _flight_ratio(self):
        """Segundos equivalentes: recorrido total / velocidad horizontal."""
        if not self._ratios:
            return 1.0
        return float(np.clip(np.median(self._ratios[-15:]), 0.3, 3.0))

    def _initial_velocities(self):
        s = self._size
        self._last_launch = None

        if self._target_xz is not None:
            delta = self._target_xz - self._start[[0, 2]]
            dist = float(np.linalg.norm(delta))
            if dist > 1e-6:
                base = np.array([delta[0], 0.0, delta[1]]) / dist
            else:
                a = self.rng.uniform(0.0, 2.0 * np.pi)
                base = np.array([np.cos(a), 0.0, np.sin(a)])

            spread = min(0.15, 0.3 * s / max(dist, 1e-6))
            a = self.rng.uniform(-spread, spread)
            direction = np.array([
                base[0] * np.cos(a) + base[2] * np.sin(a),
                0.0,
                -base[0] * np.sin(a) + base[2] * np.cos(a),
            ])
            h_speed = dist / self._flight_ratio() * self.rng.uniform(0.85, 1.15)
            self._last_launch = (direction.copy(), h_speed)
        elif self.launch_direction is None:
            a = self.rng.uniform(0.0, 2.0 * np.pi)
            direction = np.array([np.cos(a), 0.0, np.sin(a)])
            h_speed = self.rng.uniform(*self.launch_speed_range) * s
        else:
            a = self.rng.uniform(-0.25, 0.25)
            d = self.launch_direction
            direction = np.array([
                d[0] * np.cos(a) + d[2] * np.sin(a),
                0.0,
                -d[0] * np.sin(a) + d[2] * np.cos(a),
            ])
            h_speed = self.rng.uniform(*self.launch_speed_range) * s

        v = direction * h_speed
        v[1] = self.rng.uniform(*self.vertical_speed_range) * s

        # Giro mayormente hacia adelante (como un dado que "rueda" por
        # el aire), con algo de ruido en los otros ejes.
        axis = np.cross(self.UP, direction) + 0.45 * self.rng.normal(size=3)
        axis = self._normalize(axis)
        sign = 1.0 if self.rng.random() < 0.8 else -1.0
        w = axis * sign * self.rng.uniform(*self.angular_speed_range)

        # Que ninguna esquina arranque hundiéndose en la mesa.
        if self._start[1] - self._half <= self._ground_y + 1e-6:
            wz, wx = w[2], w[0]
            h = self._half
            lowest = min(wz * (sx * h) - wx * (sz * h) for sx, _, sz in _SIGNS)
            v[1] = max(v[1], -1.05 * lowest)

        return v, w

    # ==================================================================
    # Contactos (impulsos secuenciales con fricción de Coulomb)
    # ==================================================================

    def _solve_contacts(self, rs, vx, vy, vz, wx, wy, wz):
        """
        `rs` son los vectores centro→esquina (rx, ry, rz) de las esquinas
        en contacto. Devuelve las nuevas velocidades lineal y angular.
        """
        inv_i = self._inv_inertia
        mu = self.friction
        e = self.restitution
        thr = self._rest_threshold

        contacts = []
        for rx, ry, rz in rs:
            vcn = vy + wz * rx - wx * rz
            bounce = -e * vcn if vcn < -thr else 0.0
            kn = 1.0 + inv_i * (rz * rz + rx * rx)
            kx = 1.0 + inv_i * (rz * rz + ry * ry)
            kz = 1.0 + inv_i * (ry * ry + rx * rx)
            contacts.append([rx, ry, rz, bounce, kn, kx, kz, 0.0, 0.0, 0.0])

        for _ in range(self.CONTACT_ITERATIONS):
            for c in contacts:
                rx, ry, rz, bounce, kn, kx, kz, jn, jx, jz = c
                max_f = mu * jn

                # Fricción en X
                vcx = vx + wy * rz - wz * ry
                new = min(max(jx - vcx / kx, -max_f), max_f)
                dj, jx = new - jx, new
                vx += dj
                k = inv_i * dj
                wy += k * rz
                wz -= k * ry

                # Fricción en Z
                vcz = vz + wx * ry - wy * rx
                new = min(max(jz - vcz / kz, -max_f), max_f)
                dj, jz = new - jz, new
                vz += dj
                k = inv_i * dj
                wx += k * ry
                wy -= k * rx

                # Normal
                vcy = vy + wz * rx - wx * rz
                new = max(jn - (vcy - bounce) / kn, 0.0)
                dj, jn = new - jn, new
                vy += dj
                k = inv_i * dj
                wx -= k * rz
                wz += k * rx

                c[7], c[8], c[9] = jn, jx, jz

        return vx, vy, vz, wx, wy, wz

    # ==================================================================
    # Simulación de un lanzamiento (escalares puros, sin numpy por paso)
    # ==================================================================

    def _simulate_throw(self, t_limit: float) -> _Candidate | None:
        dt = self.simulation_dt
        half_dt = 0.5 * dt
        size = self._size
        h = self._half
        g = self._gravity
        ground = self._ground_y
        margin = 1e-3 * size
        v_sleep = 0.08 * size
        w_sleep = 0.3
        tilt_cos = self.SLEEP_TILT_COS
        sleep_time = self.SLEEP_TIME
        normals = self._normal_rows
        limit = self._travel_limit

        lin_air = math.exp(-self.AIR_DAMPING_LINEAR * dt)
        ang_air = math.exp(-self.AIR_DAMPING_ANGULAR * dt)
        lin_gnd = math.exp(-self.GROUND_DAMPING_LINEAR * dt)
        ang_gnd = math.exp(-self.GROUND_DAMPING_ANGULAR * dt)

        sx0, sy0, sz0 = (float(c) for c in self._start)
        xx, xy, xz = sx0, sy0, sz0
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        v0, w0 = self._initial_velocities()
        vx, vy, vz = (float(c) for c in v0)
        wx, wy, wz = (float(c) for c in w0)

        positions = [(xx, xy, xz)]
        quats = [(qw, qx, qy, qz)]
        activity = [0.0]  # "cuánto se mueve" en cada paso (rad/s + tamaños/s)

        sleep_t0 = None
        asleep = False

        for step in range(1, int(t_limit / dt) + 1):
            t = step * dt

            vy -= g * dt
            vx *= lin_air
            vy *= lin_air
            vz *= lin_air
            wx *= ang_air
            wy *= ang_air
            wz *= ang_air

            xx += vx * dt
            xy += vy * dt
            xz += vz * dt

            # q += 0.5 dt (0, w) ⊗ q  (todas las componentes con el q anterior)
            dqw = -wx * qx - wy * qy - wz * qz
            dqx = wx * qw + wy * qz - wz * qy
            dqy = -wx * qz + wy * qw + wz * qx
            dqz = wx * qy - wy * qx + wz * qw
            qw += half_dt * dqw
            qx += half_dt * dqx
            qy += half_dt * dqy
            qz += half_dt * dqz
            n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            qw /= n
            qx /= n
            qy /= n
            qz /= n

            # Fila Y de la matriz de rotación (altura de las esquinas).
            r10 = 2.0 * (qx * qy + qz * qw)
            r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
            r12 = 2.0 * (qy * qz - qx * qw)

            low = xy - ground - h * (abs(r10) + abs(r11) + abs(r12))
            grounded = low < margin

            if grounded:
                r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
                r01 = 2.0 * (qx * qy - qz * qw)
                r02 = 2.0 * (qx * qz + qy * qw)
                r20 = 2.0 * (qx * qz - qy * qw)
                r21 = 2.0 * (qy * qz + qx * qw)
                r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

                rs = []
                for sx, sy, sz in _SIGNS:
                    ry = h * (sx * r10 + sy * r11 + sz * r12)
                    if xy + ry - ground < margin:
                        rs.append((
                            h * (sx * r00 + sy * r01 + sz * r02),
                            ry,
                            h * (sx * r20 + sy * r21 + sz * r22),
                        ))

                vx, vy, vz, wx, wy, wz = self._solve_contacts(
                    rs, vx, vy, vz, wx, wy, wz
                )
                if low < 0.0:
                    xy -= low
                vx *= lin_gnd
                vy *= lin_gnd
                vz *= lin_gnd
                wx *= ang_gnd
                wy *= ang_gnd
                wz *= ang_gnd

            vspeed = math.sqrt(vx * vx + vy * vy + vz * vz)
            wspeed = math.sqrt(wx * wx + wy * wy + wz * wz)

            positions.append((xx, xy, xz))
            quats.append((qw, qx, qy, qz))
            activity.append(wspeed + vspeed / size)

            if limit is not None and math.hypot(xx - sx0, xz - sz0) > limit:
                return None

            # --- ¿está en reposo? --------------------------------------
            if (
                grounded
                and vspeed < v_sleep
                and wspeed < w_sleep
                and max(r10 * a + r11 * b + r12 * c for a, b, c in normals) > tilt_cos
            ):
                if sleep_t0 is None:
                    sleep_t0 = t
                if t - sleep_t0 >= sleep_time:
                    asleep = True
                    break
            else:
                sleep_t0 = None

        if not asleep:
            return None

        q = np.array([qw, qx, qy, qz])
        x = np.array([xx, xy, xz])
        R = _quat_to_matrix(q)

        # Aprendemos cuánto recorrió el dado respecto a su velocidad
        # horizontal de salida (sirve para apuntar mejor al siguiente).
        if self._last_launch is not None:
            d_launch, h_launch = self._last_launch
            if h_launch > 1e-3 * size:
                proj = float(np.dot((x - self._start)[[0, 2]], d_launch[[0, 2]]))
                self._ratios.append(proj / h_launch)

        # --- Cara superior y orientación plana ----------------------
        k_up = int(np.argmax((self._normals @ R.T)[:, 1]))
        face_normal = R @ self._normals[k_up]

        axis = np.cross(face_normal, self.UP)
        sin_a = float(np.linalg.norm(axis))
        cos_a = float(np.dot(face_normal, self.UP))
        if sin_a > 1e-9:
            q_flat = _qmul(_quat_axis_angle(axis / sin_a, np.arctan2(sin_a, cos_a)), q)
        else:
            q_flat = q.copy()

        # --- Giro final alrededor de Y: alinear con los ejes ---------
        # Con la cara superior ya horizontal, las otras cuatro normales
        # son horizontales. Medimos su ángulo respecto a los ejes X/Z y
        # lo llevamos al múltiplo de 90° más cercano.
        yaw = 0.0
        if self.snap_yaw:
            R_flat = _quat_to_matrix(q_flat)
            side = next(
                R_flat @ n for n in self._normals
                if abs((R_flat @ n)[1]) < 0.5
            )
            angle = np.arctan2(side[2], side[0])
            quarter = 0.5 * np.pi
            yaw = float((angle + quarter / 2) % quarter - quarter / 2)
            q_flat = _qmul(_quat_axis_angle(self.UP, yaw), q_flat)

        residual = None
        target_error = 0.0
        if self._target_xz is not None:
            residual = self._target_xz - x[[0, 2]]
            target_error = float(np.linalg.norm(residual))

        blend_steps = max(1, int(self.BLEND_TIME / dt))

        return _Candidate(
            positions=positions,
            quats=quats,
            activity=activity,
            q=q,
            q_flat=q_flat,
            x=x,
            face=self._faces[k_up],
            yaw=yaw,
            residual=residual,
            settle_time=(len(positions) - 1 + blend_steps) * dt,
            yaw_error=abs(yaw),
            target_error=target_error,
        )

    # ==================================================================
    # Construcción de la trayectoria del lanzamiento elegido
    # ==================================================================

    def _build_throw(self, cand: _Candidate) -> _Throw:
        dt = self.simulation_dt

        pos = np.array(cand.positions)
        quats = np.array(cand.quats)
        activity = np.array(cand.activity)

        # Reparto de las correcciones (giro y posición) a lo largo del
        # movimiento, en proporción a cuánto se mueve el dado: mucho en
        # el aire, casi nada cuando ya está quieto. Así no hay ningún
        # ajuste visible al final.
        cumulative = np.cumsum(activity)
        if cumulative[-1] > 1e-9:
            weights = cumulative / cumulative[-1]
        else:
            weights = np.linspace(0.0, 1.0, len(activity))

        q = cand.q
        if self.snap_yaw:
            # Giro alrededor de Y por el centro: no altera el contacto con
            # la mesa ni la posición. Equivale a multiplicar cada
            # cuaternión por (cos θ/2, 0, sin θ/2, 0).
            half = 0.5 * cand.yaw * weights
            a = np.cos(half)
            b = np.sin(half)
            w0, x0, y0, z0 = quats.T
            quats = np.column_stack((
                a * w0 - b * y0,
                a * x0 + b * z0,
                a * y0 + b * w0,
                a * z0 - b * x0,
            ))
            q = quats[-1].copy()

        # Posición final exacta en el plano XZ (desplazamiento horizontal:
        # no toca la altura ni el contacto con la mesa).
        if cand.residual is not None:
            pos[:, 0] += cand.residual[0] * weights
            pos[:, 2] += cand.residual[1] * weights
        x = pos[-1].copy()

        x_flat = x.copy()
        x_flat[1] = self._ground_y + self._half

        # Lo que queda por corregir aquí es solo una inclinación mínima
        # (menos de ~2°) y un ajuste de altura de décimas de milímetro.
        blend_steps = max(1, int(self.BLEND_TIME / dt))
        ts = np.array([_smoothstep(k / blend_steps) for k in range(1, blend_steps + 1)])
        blend_pos = x[None, :] * (1.0 - ts)[:, None] + x_flat[None, :] * ts[:, None]
        blend_q = np.array([_slerp(q, cand.q_flat, a) for a in ts])

        return _Throw(
            positions=np.vstack((pos, blend_pos)),
            quaternions=np.vstack((quats, blend_q)),
            face=cand.face,
            settle_time=cand.settle_time,
            yaw_error=cand.yaw_error,
            target_error=cand.target_error,
        )

    # ==================================================================
    # Búsqueda del lanzamiento
    # ==================================================================

    def _find_throw(self, run_time: float):
        max_settle = self.settle_fraction * run_time
        t_limit = 1.6 * max_settle
        tol = np.radians(self.yaw_tolerance_deg)
        tol_t = self.target_tolerance * self._size

        # Plan B: primero el menor error de posición, luego el menor
        # giro final, luego el menor tiempo.
        def cost(t):
            return (
                max(0.0, t.target_error - tol_t),
                max(0.0, t.yaw_error - tol),
                max(0.0, t.settle_time - max_settle),
            )

        best: _Candidate | None = None

        for _ in range(self.max_attempts):
            cand = self._simulate_throw(t_limit)

            if cand is None or cand.face != self.outcome:
                continue

            if (
                cand.settle_time <= max_settle
                and cand.yaw_error <= tol
                and cand.target_error <= tol_t
            ):
                best = cand
                break

            if best is None or cost(cand) < cost(best):
                best = cand

        if best is None:
            raise RuntimeError(
                f"No se encontró un lanzamiento que termine con la cara "
                f"{self.outcome} arriba tras {self.max_attempts} intentos. "
                "Prueba con más `max_attempts`, un `run_time` mayor o un "
                "`max_travel` más grande."
            )

        self._throw = self._build_throw(best)
        # Si el mejor lanzamiento tarda más que el run_time, se acelera
        # ligeramente la reproducción (casi nunca ocurre).
        self._time_scale = max(1.0, best.settle_time / max_settle)

    # ==================================================================
    # Reproducción
    # ==================================================================

    def _set_pose_exact(self, position, quaternion):
        """Pose exacta desde la geometría original (sin deriva)."""
        R = _quat_to_matrix(quaternion)
        position = np.asarray(position, dtype=float)
        self.mobject.become(self._initial_dice)
        self.mobject.apply_matrix(R, about_point=self._initial_center)
        self.mobject.shift(position - self._initial_center)
        self._cur_pos = position.copy()
        self._cur_R = R

    def _apply_pose(self, position, R):
        """
        Lleva el mobject de la pose actual a (position, R) aplicando solo
        la transformación relativa, en una única pasada por sus puntos.
        """
        cur_p, cur_R = self._cur_pos, self._cur_R

        # Pose sin cambios (p. ej. dado ya en reposo): no se toca nada.
        if (
            np.abs(position - cur_p).max() < 1e-12
            and np.abs(R - cur_R).max() < 1e-12
        ):
            return

        M = R @ cur_R.T
        t = position - M @ cur_p

        done = False
        if self._fused_ok:
            MT = M.T
            try:
                self.mobject.apply_points_function(
                    lambda pts: pts @ MT + t,
                    about_point=self._ZERO,
                )
                done = True
            except TypeError:
                self._fused_ok = False
        if not done:
            self.mobject.apply_matrix(M, about_point=cur_p)
            self.mobject.shift(position - cur_p)

        self._cur_pos = position
        self._cur_R = R

    def begin(self):
        # Partimos siempre de la geometría original exacta (también si la
        # animación se reproduce más de una vez).
        self.mobject.become(self._initial_dice)
        self._cur_pos = self._initial_center.copy()
        self._cur_R = np.eye(3)
        super().begin()

    def interpolate_mobject(self, alpha: float):
        alpha = float(self.rate_func(float(np.clip(alpha, 0.0, 1.0))))

        throw = self._throw
        n = len(throw.positions)

        frame = alpha * self.run_time * self._time_scale / self.simulation_dt
        i0 = int(frame)

        if i0 >= n - 1:
            self._apply_pose(
                throw.positions[-1],
                _quat_to_matrix(throw.quaternions[-1]),
            )
            return

        a = frame - i0
        p = throw.positions[i0] * (1 - a) + throw.positions[i0 + 1] * a
        q = _slerp(throw.quaternions[i0], throw.quaternions[i0 + 1], a)
        self._apply_pose(p, _quat_to_matrix(q))

    def finish(self):
        # Pose final exacta: elimina cualquier deriva de redondeo.
        self._set_pose_exact(self._throw.positions[-1], self._throw.quaternions[-1])
        return super().finish()