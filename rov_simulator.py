import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pygame

WINDOW_SIZE = (1280, 720)
FPS = 60
DEADBAND = 0.1
MPU_DEADBAND_DEGREES = 1.5

# View controls (keyboard): arrow keys rotate camera around ROV, +/- zoom.
CAMERA_YAW_SPEED = 1.8
CAMERA_PITCH_SPEED = 1.3
CAMERA_ZOOM_SPEED = 3.5

PS3_LAYOUT = {
    "claw_angle_increase": 13,
    "claw_angle_decrease": 14,
    "claw_rotate_increase": 12,
    "claw_rotate_decrease": 15,
    "pitch_positive": 4,
    "pitch_negative": 6,
    "roll_positive": 5,
    "roll_negative": 7,
    "claw_angle_preset_low": None,
    "claw_angle_preset_high": None,
    "syringe_open": None,
    "syringe_close": None,
    "camera_zero": None,
    "camera_max": None,
    "stabilization_toggle": 3,
}

XBOX_ONE_LAYOUT = {
    "claw_angle_increase": 1,
    "claw_angle_decrease": 0,
    "claw_rotate_increase": 3,
    "claw_rotate_decrease": 2,
    "pitch_positive": "dpad_up",
    "pitch_negative": "dpad_down",
    "roll_positive": "dpad_right",
    "roll_negative": "dpad_left",
    "syringe_open": None,
    "syringe_close": None,
    "claw_angle_preset_high": None,
    "claw_angle_preset_low": None,
    "camera_zero": None,
    "camera_max": None,
    "stabilization_toggle": 7,
}


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def vec_add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def vec_scale(v: Sequence[float], s: float) -> List[float]:
    return [v[0] * s, v[1] * s, v[2] * s]


def vec_cross(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def vec_normalize(v: Sequence[float]) -> List[float]:
    magnitude = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if magnitude < 1e-8:
        return [0.0, 0.0, 0.0]
    return [v[0] / magnitude, v[1] / magnitude, v[2] / magnitude]


def quat_multiply(a: Sequence[float], b: Sequence[float]) -> List[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def quat_normalize(q: Sequence[float]) -> List[float]:
    magnitude = math.sqrt(sum(component * component for component in q))
    if magnitude < 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    return [component / magnitude for component in q]


def quat_from_axis_angle(axis: Sequence[float], angle: float) -> List[float]:
    half = angle * 0.5
    s = math.sin(half)
    return [math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s]


def quat_rotate(point: Sequence[float], q: Sequence[float]) -> List[float]:
    qw, qx, qy, qz = q
    px, py, pz = point

    ix = qw * px + qy * pz - qz * py
    iy = qw * py + qz * px - qx * pz
    iz = qw * pz + qx * py - qy * px
    iw = -qx * px - qy * py - qz * pz

    x = ix * qw + iw * -qx + iy * -qz - iz * -qy
    y = iy * qw + iw * -qy + iz * -qx - ix * -qz
    z = iz * qw + iw * -qz + ix * -qy - iy * -qx
    return [x, y, z]


def quat_conjugate(q: Sequence[float]) -> List[float]:
    return [q[0], -q[1], -q[2], -q[3]]


def quat_to_euler(q: Sequence[float]) -> List[float]:
    qw, qx, qy, qz = q

    sinp = 2.0 * (qw * qx - qy * qz)
    pitch = math.asin(clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (qw * qy + qx * qz)
    cosy_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    sinr_cosp = 2.0 * (qw * qz + qx * qy)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qz * qz)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    return [pitch, yaw, roll]


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def apply_deadband(value: float, threshold: float) -> float:
    if abs(value) < threshold:
        return 0.0
    return value


def merge_inputs(joystick_input: float, mpu_input: float) -> float:
    if mpu_input < 0:
        mpu_input = -((-mpu_input) ** 0.5)
    else:
        mpu_input **= 0.5

    return lerp(mpu_input, joystick_input, math.fabs(joystick_input))


def calculate_orientation_from_sim(sim: "ROVSimulator") -> Tuple[float, float]:
    # Derive a simulated MPU-6050 accelerometer vector from simulator orientation.
    # Simulator body axes are +X right, +Y up, +Z forward with yaw around +Y.
    # Use world-up in body frame, then remap to MPU-style axes expected by
    # bottom-side.py formulas so yaw does not appear as pitch.
    body_right, body_up, body_forward = quat_rotate((0.0, 1.0, 0.0), quat_conjugate(sim.orientation))

    # Align simulator body axes (+X right, +Y up, +Z forward) with the MPU axis
    # convention used in bottom-side.py's pitch/roll equations.
    #
    # Map the simulated body frame onto the same MPU axis convention used by
    # bottom-side.py so pitch reflects forward tilt and roll reflects right tilt,
    # while heading (yaw about +Y) stays decoupled from both readings.
    accel_x = -body_forward
    accel_y = -body_right
    accel_z = body_up

    pitch = math.atan2(accel_x, math.sqrt(accel_y**2 + accel_z**2))
    roll = math.atan2(-accel_y, accel_z)
    return pitch, roll


def apply_stabilization(control_input: List[float], sim: "ROVSimulator") -> Tuple[float, float]:
    pitch_rad, roll_rad = calculate_orientation_from_sim(sim)

    if control_input[12]:
        pitch = pitch_rad / math.pi
        roll = roll_rad / math.pi

        pitch = apply_deadband(pitch, MPU_DEADBAND_DEGREES / 180.0)
        roll = apply_deadband(roll, MPU_DEADBAND_DEGREES / 180.0)

        control_input[0] = merge_inputs(control_input[0], roll - pitch)
        control_input[1] = merge_inputs(control_input[1], -roll - pitch)
        control_input[2] = merge_inputs(control_input[2], -roll + pitch)
        control_input[3] = merge_inputs(control_input[3], roll + pitch)
        control_input[4] = merge_inputs(control_input[4], roll + pitch)
        control_input[5] = merge_inputs(control_input[5], -roll + pitch)
        control_input[6] = merge_inputs(control_input[6], -roll - pitch)
        control_input[7] = merge_inputs(control_input[7], roll - pitch)

    for i in range(8):
        control_input[i] = clamp(control_input[i])

    return math.degrees(pitch_rad), math.degrees(roll_rad)


ControlMapping = Dict[str, Union[int, str]]


def get_controller_layout(joystick_name: str) -> ControlMapping:
    if "xbox" in joystick_name.lower():
        print("Using Xbox One button mapping")
        return XBOX_ONE_LAYOUT
    print("Using PS3 button mapping")
    return PS3_LAYOUT


@dataclass
class Thruster:
    motor_label: str
    position: Tuple[float, float, float]
    positive_direction: Tuple[float, float, float]


# Body frame convention:
# +X right, +Y up, +Z forward.
THRUSTERS = [
    Thruster("M1", (-1.35,  0.55,  0.95), ( 0.612, -0.5,  0.612)),
    Thruster("M2", (-1.35,  0.55, -0.95), ( 0.612, -0.5, -0.612)),
    Thruster("M3", (-1.35, -0.55,  0.95), ( 0.612,  0.5,  0.612)),
    Thruster("M4", (-1.35, -0.55, -0.95), ( 0.612,  0.5, -0.612)),
    Thruster("M5", ( 1.35,  0.55,  0.95), (-0.612, -0.5,  0.612)),
    Thruster("M6", ( 1.35,  0.55, -0.95), (-0.612, -0.5, -0.612)),
    Thruster("M7", ( 1.35, -0.55,  0.95), (-0.612,  0.5,  0.612)),
    Thruster("M8", ( 1.35, -0.55, -0.95), (-0.612,  0.5, -0.612)),
]


class InputModel:
    def __init__(self):
        self.controller_layout: ControlMapping = PS3_LAYOUT
        self.joystick: Optional[pygame.joystick.Joystick] = None
        self.stabilization_debounce = True

        self.claw_angle = 50
        self.claw_rotate = 90
        self.syringe_angle = 90
        self.camera_angle = 90
        self.enable_stabilization = False

        self.control_input = [0.0] * 13
        self.control_input[8] = self.claw_angle
        self.control_input[9] = self.claw_rotate
        self.control_input[10] = self.syringe_angle
        self.control_input[11] = self.camera_angle

    def maybe_connect_controller(self):
        if pygame.joystick.get_count() > 0 and self.joystick is None:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.controller_layout = get_controller_layout(self.joystick.get_name())
            print(f"Joystick connected: {self.joystick.get_name()}")

    def handle_events(self, events: List[pygame.event.Event]):
        for event in events:
            if event.type == pygame.JOYDEVICEADDED:
                self.maybe_connect_controller()
            elif event.type == pygame.JOYDEVICEREMOVED:
                if self.joystick is not None:
                    self.joystick.quit()
                self.joystick = None
                print("Joystick disconnected")

    def update(self):
        if self.joystick is None:
            self.control_input[:8] = [0.0] * 8
            return self.control_input

        pitch = 0
        roll = 0
        stabilization_pressed = False
        dpad_x = 0
        dpad_y = 0

        if self.joystick.get_numhats() > 0:
            dpad_x, dpad_y = self.joystick.get_hat(0)

        def control_active(control_name: str, button: Optional[int] = None) -> bool:
            mapping = self.controller_layout[control_name]
            if isinstance(mapping, int):
                return button == mapping if button is not None else False
            if mapping == "dpad_left":
                return dpad_x == -1
            if mapping == "dpad_right":
                return dpad_x == 1
            if mapping == "dpad_up":
                return dpad_y == 1
            if mapping == "dpad_down":
                return dpad_y == -1
            return False

        for button in range(self.joystick.get_numbuttons()):
            pressed = self.joystick.get_button(button)
            if not pressed:
                continue
            if control_active("claw_angle_increase", button) and self.claw_angle <= 179:
                self.claw_angle += 1
            elif control_active("claw_angle_decrease", button) and self.claw_angle >= 1:
                self.claw_angle -= 1
            if control_active("claw_rotate_increase", button) and self.claw_rotate <= 179:
                self.claw_rotate += 1
            elif control_active("claw_rotate_decrease", button) and self.claw_rotate >= 1:
                self.claw_rotate -= 1
            if control_active("pitch_positive", button):
                pitch = 1
            elif control_active("pitch_negative", button):
                pitch = -1
            if control_active("roll_positive", button):
                roll = 1
            elif control_active("roll_negative", button):
                roll = -1
            if control_active("claw_angle_preset_low", button):
                self.claw_angle = 65
            if control_active("claw_angle_preset_high", button):
                self.claw_angle = 100
            if control_active("syringe_open", button):
                self.syringe_angle = 180
            if control_active("syringe_close", button):
                self.syringe_angle = 0
            if control_active("camera_zero", button):
                self.camera_angle = 0
            if control_active("camera_max", button):
                self.camera_angle = 180
            if control_active("stabilization_toggle", button):
                stabilization_pressed = True

        if control_active("pitch_positive"):
            pitch = 1
        elif control_active("pitch_negative"):
            pitch = -1

        if control_active("roll_positive"):
            roll = 1
        elif control_active("roll_negative"):
            roll = -1

        if control_active("claw_angle_preset_low"):
            self.claw_angle = 65
        if control_active("claw_angle_preset_high"):
            self.claw_angle = 100
        if control_active("syringe_open"):
            self.syringe_angle = 180
        if control_active("syringe_close"):
            self.syringe_angle = 0

        if stabilization_pressed:
            if self.stabilization_debounce:
                self.stabilization_debounce = False
                self.enable_stabilization = not self.enable_stabilization
        else:
            self.stabilization_debounce = True

        yaw = self.joystick.get_axis(0)
        forward_backward = -self.joystick.get_axis(3)
        left_right = self.joystick.get_axis(2)
        up_down = self.joystick.get_axis(1)

        forward_backward = 0 if abs(forward_backward) < DEADBAND else forward_backward
        left_right = 0 if abs(left_right) < DEADBAND else left_right
        yaw = 0 if abs(yaw) < DEADBAND else yaw
        up_down = 0 if abs(up_down) < DEADBAND else up_down

        c = self.control_input
        c[0] = round(clamp(-forward_backward + left_right + up_down - pitch + yaw + roll), 3)
        c[1] = round(clamp(-forward_backward - left_right + up_down - pitch - yaw - roll), 3)
        c[2] = round(clamp(-forward_backward + left_right - up_down + pitch + yaw - roll), 3)
        c[3] = round(clamp(-forward_backward - left_right - up_down + pitch - yaw + roll), 3)
        c[4] = round(clamp(forward_backward + left_right + up_down + pitch - yaw + roll), 3)
        c[5] = round(clamp(forward_backward - left_right + up_down + pitch + yaw - roll), 3)
        c[6] = round(clamp(forward_backward + left_right - up_down - pitch - yaw - roll), 3)
        c[7] = round(clamp(forward_backward - left_right - up_down - pitch + yaw + roll), 3)

        c[8] = self.claw_angle
        c[9] = self.claw_rotate
        c[10] = self.syringe_angle
        c[11] = self.camera_angle
        c[12] = self.enable_stabilization
        return c


class ROVSimulator:
    def __init__(self):
        self.linear_drag = 3.8
        self.angular_drag = 6.5
        self.linear_quad_drag = 1.1
        self.angular_quad_drag = 2.4
        self.force_gain = 4.8
        self.torque_gain = 2.6

        self.net_body_force = [0.0, 0.0, 0.0]
        self.net_body_torque = [0.0, 0.0, 0.0]
        self.reset()

    def reset(self):
        self.pos = [0.0, 0.0, 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.rot = [0.0, 0.0, 0.0]  # pitch, yaw, roll (debug view derived from quaternion)
        self.rot_vel = [0.0, 0.0, 0.0]  # body-frame angular velocity (x, y, z)
        self.orientation = [1.0, 0.0, 0.0, 0.0]  # quaternion body->world
        self.net_body_force = [0.0, 0.0, 0.0]
        self.net_body_torque = [0.0, 0.0, 0.0]

    def rotate_body_to_world(self, vector: Sequence[float]) -> List[float]:
        return quat_rotate(vector, self.orientation)

    def update(self, thruster_input: List[float], dt: float):
        body_force = [0.0, 0.0, 0.0]
        body_torque = [0.0, 0.0, 0.0]

        for index, thruster in enumerate(THRUSTERS):
            power = clamp(thruster_input[index])
            direction = vec_normalize(thruster.positive_direction)
            thruster_force = vec_scale(direction, power * self.force_gain)
            body_force = vec_add(body_force, thruster_force)

            torque = vec_cross(thruster.position, thruster_force)
            body_torque = vec_add(body_torque, vec_scale(torque, self.torque_gain))

        self.net_body_force = body_force
        self.net_body_torque = body_torque

        world_force = self.rotate_body_to_world(body_force)
        for axis in range(3):
            self.vel[axis] += world_force[axis] * dt
            speed = abs(self.vel[axis])
            total_linear_drag = self.linear_drag + self.linear_quad_drag * speed
            self.vel[axis] *= max(0.0, 1.0 - total_linear_drag * dt)
            self.pos[axis] += self.vel[axis] * dt

        for axis in range(3):
            self.rot_vel[axis] += body_torque[axis] * dt
            angular_speed = abs(self.rot_vel[axis])
            total_angular_drag = self.angular_drag + self.angular_quad_drag * angular_speed
            self.rot_vel[axis] *= max(0.0, 1.0 - total_angular_drag * dt)

        omega_magnitude = math.sqrt(sum(component * component for component in self.rot_vel))
        if omega_magnitude > 1e-8:
            axis = [component / omega_magnitude for component in self.rot_vel]
            delta_q = quat_from_axis_angle(axis, omega_magnitude * dt)
            self.orientation = quat_normalize(quat_multiply(self.orientation, delta_q))

        self.rot = quat_to_euler(self.orientation)


class Camera:
    def __init__(self):
        self.reset()

    def reset(self):
        self.yaw = 0.45
        self.pitch = -0.35
        self.distance = 8.5

    def update(self, keys: pygame.key.ScancodeWrapper, dt: float):
        if keys[pygame.K_LEFT]:
            self.yaw -= CAMERA_YAW_SPEED * dt
        if keys[pygame.K_RIGHT]:
            self.yaw += CAMERA_YAW_SPEED * dt
        if keys[pygame.K_UP]:
            self.pitch = clamp(self.pitch + CAMERA_PITCH_SPEED * dt, -1.2, 1.2)
        if keys[pygame.K_DOWN]:
            self.pitch = clamp(self.pitch - CAMERA_PITCH_SPEED * dt, -1.2, 1.2)

        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            self.distance = clamp(self.distance - CAMERA_ZOOM_SPEED * dt, 4.0, 14.0)
        if keys[pygame.K_MINUS]:
            self.distance = clamp(self.distance + CAMERA_ZOOM_SPEED * dt, 4.0, 14.0)


def rotate_point(point: Tuple[float, float, float], rotation: Tuple[float, float, float]):
    x, y, z = point
    pitch, yaw, roll = rotation

    # Apply intrinsic body rotations as roll -> pitch -> yaw.
    # This yields body-to-world transform R = R_y(yaw) * R_x(pitch) * R_z(roll),
    # so pitch remains relative to the current yawed heading.
    cr, sr = math.cos(roll), math.sin(roll)
    x, y = x * cr - y * sr, x * sr + y * cr

    cp, sp = math.cos(pitch), math.sin(pitch)
    y, z = y * cp - z * sp, y * sp + z * cp

    cy, sy = math.cos(yaw), math.sin(yaw)
    x, z = x * cy + z * sy, -x * sy + z * cy
    return [x, y, z]


def world_to_screen(point_world: Sequence[float], camera: Camera) -> Tuple[int, int]:
    point = rotate_point(tuple(point_world), (camera.pitch, camera.yaw, 0.0))
    scale = camera.distance / (camera.distance + point[2] + 1e-4)
    x = int(point[0] * 120 * scale + WINDOW_SIZE[0] / 2)
    y = int(-point[1] * 120 * scale + WINDOW_SIZE[1] / 2)
    return x, y


def to_world(local_point: Tuple[float, float, float], sim: ROVSimulator) -> List[float]:
    rotated = sim.rotate_body_to_world(local_point)
    return vec_add(rotated, sim.pos)


def draw_rov(screen: pygame.Surface, sim: ROVSimulator, thrusters: List[float], camera: Camera):
    half = (1.3, 0.45, 0.9)
    vertices = [
        (-half[0], -half[1], -half[2]),
        (half[0], -half[1], -half[2]),
        (half[0], half[1], -half[2]),
        (-half[0], half[1], -half[2]),
        (-half[0], -half[1], half[2]),
        (half[0], -half[1], half[2]),
        (half[0], half[1], half[2]),
        (-half[0], half[1], half[2]),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    world_vertices = [to_world(v, sim) for v in vertices]
    projected = [world_to_screen(v, camera) for v in world_vertices]

    for edge in edges:
        pygame.draw.line(screen, (185, 220, 255), projected[edge[0]], projected[edge[1]], 2)

    for index, thruster in enumerate(THRUSTERS):
        center_world = to_world(thruster.position, sim)
        center_px = world_to_screen(center_world, camera)

        base_direction = vec_normalize(thruster.positive_direction)
        direction_world = sim.rotate_body_to_world(base_direction)

        color = (90, 220, 120) if thrusters[index] >= 0 else (235, 95, 95)
        intensity = int(100 + 155 * min(1.0, abs(thrusters[index])))
        color = (min(255, color[0] + intensity // 3), min(255, color[1] + intensity // 4), min(255, color[2] + intensity // 5))

        # Cylinder representation: draw a capsule shape for the body plus end caps.
        cyl_half_length = 0.22
        axis_start = vec_add(thruster.position, vec_scale(base_direction, -cyl_half_length))
        axis_end = vec_add(thruster.position, vec_scale(base_direction, cyl_half_length))
        axis_start_world = to_world(tuple(axis_start), sim)
        axis_end_world = to_world(tuple(axis_end), sim)

        start_px = world_to_screen(axis_start_world, camera)
        end_px = world_to_screen(axis_end_world, camera)
        radius = 5

        dx = end_px[0] - start_px[0]
        dy = end_px[1] - start_px[1]
        segment_len = math.hypot(dx, dy)
        if segment_len > 1e-4:
            nx = -dy / segment_len
            ny = dx / segment_len
            body_poly = [
                (int(start_px[0] + nx * radius), int(start_px[1] + ny * radius)),
                (int(end_px[0] + nx * radius), int(end_px[1] + ny * radius)),
                (int(end_px[0] - nx * radius), int(end_px[1] - ny * radius)),
                (int(start_px[0] - nx * radius), int(start_px[1] - ny * radius)),
            ]
            pygame.draw.polygon(screen, (165, 165, 172), body_poly)

        pygame.draw.circle(screen, (204, 204, 212), start_px, radius)
        pygame.draw.circle(screen, (204, 204, 212), end_px, radius)
        pygame.draw.circle(screen, (130, 130, 140), start_px, radius, 1)
        pygame.draw.circle(screen, (130, 130, 140), end_px, radius, 1)

        # Indicator shows direction of applied push force (opposite prop wash velocity).
        thrust_tip_world = vec_add(center_world, vec_scale(direction_world, thrusters[index] * 0.65))
        pygame.draw.line(screen, color, center_px, world_to_screen(thrust_tip_world, camera), 4)

        font = pygame.font.SysFont("Consolas", 18)
        label = font.render(thruster.motor_label, True, (245, 245, 150))
        screen.blit(label, (center_px[0] + 8, center_px[1] - 10))


def draw_hud(
    screen: pygame.Surface,
    control_input: List[float],
    connected: bool,
    sim: ROVSimulator,
    mpu_pitch_deg: float,
    mpu_roll_deg: float,
):
    font = pygame.font.SysFont("Consolas", 20)
    small = pygame.font.SysFont("Consolas", 16)
    status = "Joystick connected" if connected else "No joystick detected"
    status_color = (120, 255, 120) if connected else (255, 190, 120)
    screen.blit(font.render(status, True, status_color), (20, 20))

    for i in range(8):
        text = f"M{i + 1}: {control_input[i]:>6.3f}"
        screen.blit(font.render(text, True, (220, 230, 255)), (20, 60 + i * 24))

    telemetry = [
        f"Claw Angle: {control_input[8]:>5.1f}",
        f"Claw Rotate: {control_input[9]:>5.1f}",
        f"Syringe: {control_input[10]:>5.1f}",
        f"Camera Servo: {control_input[11]:>5.1f}",
        f"Stabilization: {'ON' if control_input[12] else 'OFF'}",
        f"MPU Pitch: {mpu_pitch_deg:>6.2f}",
        f"MPU Roll:  {mpu_roll_deg:>6.2f}",
    ]
    for i, text in enumerate(telemetry):
        screen.blit(font.render(text, True, (220, 230, 255)), (250, 60 + i * 24))

    force_text = f"Net Body Force  X/Y/Z: {sim.net_body_force[0]:>6.2f}, {sim.net_body_force[1]:>6.2f}, {sim.net_body_force[2]:>6.2f}"
    torque_text = f"Net Body Torque X/Y/Z: {sim.net_body_torque[0]:>6.2f}, {sim.net_body_torque[1]:>6.2f}, {sim.net_body_torque[2]:>6.2f}"
    screen.blit(small.render(force_text, True, (180, 255, 190)), (20, WINDOW_SIZE[1] - 54))
    screen.blit(small.render(torque_text, True, (180, 255, 190)), (20, WINDOW_SIZE[1] - 30))
    screen.blit(small.render("Camera: arrows rotate, +/- zoom, R resets sim", True, (190, 200, 210)), (20, WINDOW_SIZE[1] - 78))


def main():
    pygame.init()
    pygame.joystick.init()
    pygame.display.set_caption("ROV Simulator")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()

    inputs = InputModel()
    inputs.maybe_connect_controller()
    sim = ROVSimulator()
    camera = Camera()

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                sim.reset()
                camera.reset()

        inputs.handle_events(events)
        control_input = inputs.update()
        mpu_pitch_deg, mpu_roll_deg = apply_stabilization(control_input, sim)
        sim.update(control_input[:8], dt)
        camera.update(pygame.key.get_pressed(), dt)

        screen.fill((12, 15, 26))
        draw_rov(screen, sim, control_input[:8], camera)
        draw_hud(screen, control_input, inputs.joystick is not None, sim, mpu_pitch_deg, mpu_roll_deg)
        pygame.display.flip()

    if inputs.joystick is not None:
        inputs.joystick.quit()
    pygame.quit()


if __name__ == "__main__":
    main()
