"""第三问：药材烘房二维轴对称长时干燥模型 —— 唯一入口脚本。

本文件由原 q2_2d_model.py（物理库）、q3_2d_model.py（主模型）、solve_result3.py（入口）
与 6 个 q3_*.py 检验脚本合并而成：物理公式、离散格式与求解器逐字保留。

用法（在 Problem III/ 目录下执行）：

    python solve_problem3.py                     # 求解并写出 ../result3.xlsx（默认自适应 BDF）
    python solve_problem3.py --solver rk4        # 可选：固定步长二阶段隐式 RK4
    python solve_problem3.py --no-xlsx           # 只求解、不写交付件
    python solve_problem3.py --xlsx D:/result3.xlsx
    python solve_problem3.py --mode compare      # BDF 与 RK4 求解器对比
    python solve_problem3.py --mode grid         # 网格收敛性检验
    python solve_problem3.py --mode temporal     # 时间步长收敛性检验（RK4 路径）
    python solve_problem3.py --mode conservation # 全局水分守恒检验
    python solve_problem3.py --mode steady       # 均匀稳态检验
    python solve_problem3.py --mode rebound      # 表面含水率回升检验
    python solve_problem3.py --mode all          # 求解 + 全部六项检验

产出：

    ../result3.xlsx                     唯一的文件产出：每 60 s 的中截面径向水分浓度（四位小数）

求解与六项检验的关键数值（烘干时间、最终最大含水率、每 6 h 中截面含水率表、网格与时间
收敛阶、水分守恒误差、均匀稳态漂移、表面回升量、RK4 与 BDF 对照）一律打印到终端，
不写 npz、报告 md 或检验表格。

时间推进：BDF（rtol=2e-7，与问题四同一套）为生产解；RK4 为可复选路径。两条路径
在交付口径上的烘干时间逐位一致，终端输出会打印实际步数与右端求值次数。
"""
from __future__ import annotations

import argparse
import math
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from scipy.sparse import bmat, coo_matrix


# ============================================================================
# 物理库（原 q2_2d_model.py 前 430 行）
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent


PROJECT = BASE_DIR.parent


RADIUS = 0.02


LENGTH = 0.25


TEMPERATURE_0 = 28.0


MOISTURE_0 = 2.55


HEAT_TRANSFER_COEFFICIENT = 25.0


MASS_TRANSFER_COEFFICIENT = 8.0e-7


PLATEAU_START_TIME = 10800.0


SMOOTH_WIN, SMOOTH_ORDER = 31, 3


PHASE_BLEND = False


CRITICAL_WINDOW = 1200.0


CRITICAL_PERSISTENCE = 1800.0


PHASE_TRANSITION_DURATION = 1800.0


OUTPUT_RADII = np.arange(0.0, RADIUS + 0.0005, 0.001)


FULL_FIELD_TIMES = np.array([0, 1800, 3600, 5400, 7200, 9000, 10800])


TABLE_TIMES = np.array([1800, 3600, 5400, 7200, 9000, 10800])


def read_boundary_conditions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = PROJECT / "附件" / "附件1.xlsx"
    workbook = load_workbook(path, data_only=True, read_only=True)
    worksheet = workbook.active
    rows = [
        (float(row[0]), float(row[1]), float(row[2]))
        for row in worksheet.iter_rows(min_row=2, values_only=True)
        if row[0] is not None
    ]
    workbook.close()
    values = np.asarray(rows, dtype=float)
    return values[:, 0], values[:, 1], values[:, 2]


BOUNDARY_TIME, BOUNDARY_TEMPERATURE_RAW, BOUNDARY_MOISTURE_RAW = (
    read_boundary_conditions()
)


BOUNDARY_TEMPERATURE = savgol_filter(
    BOUNDARY_TEMPERATURE_RAW, SMOOTH_WIN, SMOOTH_ORDER
)


BOUNDARY_MOISTURE = savgol_filter(BOUNDARY_MOISTURE_RAW, SMOOTH_WIN, SMOOTH_ORDER)


def detect_critical_time() -> float:
    tail_mask = BOUNDARY_TIME >= PLATEAU_START_TIME
    tail_temperature = BOUNDARY_TEMPERATURE_RAW[tail_mask]
    plateau_temperature = float(np.median(tail_temperature))
    robust_sigma = float(
        1.4826
        * np.median(
            np.abs(tail_temperature - plateau_temperature)
        )
    )
    slope_limit = 3.0 * robust_sigma / 20.0

    def satisfies(candidate_index: int) -> bool:
        end_time = BOUNDARY_TIME[candidate_index]
        mask = (
            (BOUNDARY_TIME >= end_time - CRITICAL_WINDOW)
            & (BOUNDARY_TIME <= end_time)
        )
        if np.count_nonzero(mask) < 5:
            return False
        window_time_minutes = BOUNDARY_TIME[mask] / 60.0
        window_temperature = BOUNDARY_TEMPERATURE_RAW[mask]
        mean_temperature = float(np.mean(window_temperature))
        slope = float(
            np.polyfit(
                window_time_minutes, window_temperature, 1
            )[0]
        )
        return (
            abs(mean_temperature - plateau_temperature)
            <= 2.0 * robust_sigma
            and abs(slope) <= slope_limit
        )

    for candidate_index, candidate_time in enumerate(BOUNDARY_TIME):
        if candidate_time < CRITICAL_WINDOW:
            continue
        if not satisfies(candidate_index):
            continue
        persistent = True
        persistence_end = candidate_time + CRITICAL_PERSISTENCE
        for check_index in range(candidate_index, len(BOUNDARY_TIME)):
            if BOUNDARY_TIME[check_index] > persistence_end:
                break
            if not satisfies(check_index):
                persistent = False
                break
        if persistent:
            return float(candidate_time)
    raise RuntimeError("Could not determine the phase-switch time")


CRITICAL_TIME = detect_critical_time()


def build_phase_weight() -> np.ndarray:
    transition_start = CRITICAL_TIME - PHASE_TRANSITION_DURATION
    scaled = np.clip(
        (BOUNDARY_TIME - transition_start)
        / PHASE_TRANSITION_DURATION,
        0.0,
        1.0,
    )
    return scaled * scaled * (3.0 - 2.0 * scaled)


PHASE_WEIGHT = build_phase_weight()


AIR_END_TIME = float(BOUNDARY_TIME[-1])


PLATEAU_TEMPERATURE = float(
    np.mean(BOUNDARY_TEMPERATURE[BOUNDARY_TIME >= PLATEAU_START_TIME])
)


PLATEAU_MOISTURE = float(
    np.mean(BOUNDARY_MOISTURE[BOUNDARY_TIME >= PLATEAU_START_TIME])
)


DRYING_AIR_TEMPERATURE = PLATEAU_TEMPERATURE


DRYING_AIR_MOISTURE = PLATEAU_MOISTURE


def air_temperature(time: float) -> float:
    if time > AIR_END_TIME:
        return PLATEAU_TEMPERATURE
    return raw_air_temperature(time)


def air_moisture(time: float) -> float:
    if time > AIR_END_TIME:
        return PLATEAU_MOISTURE
    return float(np.interp(time, BOUNDARY_TIME, BOUNDARY_MOISTURE))


def raw_air_temperature(time: float) -> float:
    return float(np.interp(time, BOUNDARY_TIME, BOUNDARY_TEMPERATURE))


def phase_weight(time: float) -> float:
    return float(np.interp(time, BOUNDARY_TIME, PHASE_WEIGHT))


def moisture_diffusivity(
    concentration: np.ndarray,
    temperature_c: np.ndarray,
    time: float,
) -> np.ndarray:
    concentration = np.maximum(concentration, 1.0e-9)
    temperature_k = np.maximum(temperature_c + 273.15, 1.0)
    # 合并为单次 exp：exp(-0.45/C) * exp(-3850/T) 与 exp(-0.45/C - 3850/T)
    # 数学恒等，浮点差在 1 ulp 量级（已逐位量化，见验证输出）。
    appendix3 = 2.4e-3 * np.exp(
        -0.45 / concentration - 3850.0 / temperature_k
    )
    if not PHASE_BLEND:
        return appendix3
    appendix2 = 7.0e-9 * np.exp(-0.89 / concentration)
    weight = phase_weight(time)
    return (1.0 - weight) * appendix2 + weight * appendix3


def material_properties(
    concentration: np.ndarray,
    time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio = concentration / (concentration + 1.0)
    density = 650.0 + 128.0 * concentration
    heat_capacity = 1450.0 + 2736.0 * ratio
    conductivity = 0.21 + 0.38 * ratio
    if not PHASE_BLEND:
        return density, heat_capacity, conductivity
    weight = phase_weight(time)
    return (
        (1.0 - weight) * 820.0 + weight * density,
        (1.0 - weight) * 2600.0 + weight * heat_capacity,
        (1.0 - weight) * 0.36 + weight * conductivity,
    )


@lru_cache(maxsize=None)
def make_geometry(
    n_radial: int, n_axial: int
) -> tuple[
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build axisymmetric control-volume areas and volumes."""
    radii = np.linspace(0.0, RADIUS, n_radial + 1)
    axial_positions = np.linspace(0.0, LENGTH, n_axial + 1)
    dr = RADIUS / n_radial
    dz = LENGTH / n_axial

    radial_left = np.empty(n_radial + 1)
    radial_right = np.empty(n_radial + 1)
    radial_left[0] = 0.0
    radial_right[0] = 0.5 * (radii[0] + radii[1])
    radial_left[-1] = 0.5 * (radii[-2] + radii[-1])
    radial_right[-1] = RADIUS
    if n_radial > 1:
        radial_left[1:-1] = 0.5 * (radii[:-2] + radii[1:-1])
        radial_right[1:-1] = 0.5 * (radii[1:-1] + radii[2:])

    axial_left = np.empty(n_axial + 1)
    axial_right = np.empty(n_axial + 1)
    axial_left[0] = 0.0
    axial_right[0] = 0.5 * (
        axial_positions[0] + axial_positions[1]
    )
    axial_left[-1] = 0.5 * (
        axial_positions[-2] + axial_positions[-1]
    )
    axial_right[-1] = LENGTH
    if n_axial > 1:
        axial_left[1:-1] = 0.5 * (
            axial_positions[:-2] + axial_positions[1:-1]
        )
        axial_right[1:-1] = 0.5 * (
            axial_positions[1:-1] + axial_positions[2:]
        )

    radial_cell_area = 0.5 * (
        radial_right**2 - radial_left**2
    )
    axial_width = axial_right - axial_left
    volumes = radial_cell_area[:, None] * axial_width[None, :]

    radial_face_position = 0.5 * (radii[:-1] + radii[1:])
    radial_face_area = (
        radial_face_position[:, None] * axial_width[None, :]
    )
    axial_face_area = np.repeat(
        radial_cell_area[:, None], n_axial, axis=1
    )
    radial_boundary_area = RADIUS * axial_width
    axial_boundary_area = radial_cell_area
    return (
        dr,
        dz,
        volumes,
        radial_face_area,
        axial_face_area,
        radial_boundary_area,
        axial_boundary_area,
    )


_WORKSPACE: dict = {}


def workspace(n_radial: int, n_axial: int, tag: str) -> dict:
    key = (n_radial, n_axial, tag)
    space = _WORKSPACE.get(key)
    if space is None:
        nr1, nz1 = n_radial + 1, n_axial + 1
        space = {
            "d_face_r": np.empty((nr1 - 1, nz1)),
            "flux_r": np.empty((nr1 - 1, nz1)),
            "d_face_z": np.empty((nr1, nz1 - 1)),
            "flux_z": np.empty((nr1, nz1 - 1)),
            "surf_z": np.empty(nz1),
            "bnd_z": np.empty(nz1),
            "surf_r": np.empty(nr1),
            "bnd_r": np.empty(nr1),
            "bnd_r2": np.empty(nr1),
            "flux_heat": np.empty((nr1, nz1)),
            "flux_mass": np.empty((nr1, nz1)),
            "result": np.empty((nr1, nz1)),
        }
        _WORKSPACE[key] = space
    return space


def conservative_operator(
    values: np.ndarray,
    transport_property: np.ndarray,
    boundary_coefficient: float,
    ambient_value: float,
    geometry: tuple,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Conservative axisymmetric divergence of a diffusive flux.

    out 非空时写入调用方缓冲；内部临时数组按网格缓存，避免每步重复分配。
    浮点运算的分组顺序与原实现逐条一致，结果逐位相同。
    """
    (
        dr,
        dz,
        volumes,
        radial_face_area,
        axial_face_area,
        radial_boundary_area,
        axial_boundary_area,
    ) = geometry
    n_radial = values.shape[0] - 1
    n_axial = values.shape[1] - 1
    space = workspace(n_radial, n_axial, "operator")
    result = space["result"] if out is None else out
    result.fill(0.0)

    # 径向内部面：(A * 0.5(D_i+D_{i+1}) * (u_{i+1}-u_i)) / dr
    np.add(transport_property[:-1], transport_property[1:], out=space["d_face_r"])
    space["d_face_r"] *= 0.5
    np.multiply(radial_face_area, space["d_face_r"], out=space["flux_r"])
    np.subtract(values[1:], values[:-1], out=space["d_face_r"])
    space["flux_r"] *= space["d_face_r"]
    space["flux_r"] /= dr
    result[:-1] += space["flux_r"]
    result[1:] -= space["flux_r"]

    # 轴向内部面
    np.add(transport_property[:, :-1], transport_property[:, 1:], out=space["d_face_z"])
    space["d_face_z"] *= 0.5
    np.multiply(axial_face_area, space["d_face_z"], out=space["flux_z"])
    np.subtract(values[:, 1:], values[:, :-1], out=space["d_face_z"])
    space["flux_z"] *= space["d_face_z"]
    space["flux_z"] /= dz
    result[:, :-1] += space["flux_z"]
    result[:, 1:] -= space["flux_z"]

    result /= volumes

    # 侧面 r = R
    np.multiply(radial_boundary_area, boundary_coefficient, out=space["surf_z"])
    np.subtract(values[-1], ambient_value, out=space["bnd_z"])
    space["bnd_z"] *= space["surf_z"]
    space["bnd_z"] /= volumes[-1]
    result[-1] -= space["bnd_z"]

    # 两个端面 z = 0、z = L
    np.multiply(axial_boundary_area, boundary_coefficient, out=space["surf_r"])
    np.subtract(values[:, 0], ambient_value, out=space["bnd_r"])
    space["bnd_r"] *= space["surf_r"]
    space["bnd_r"] /= volumes[:, 0]
    result[:, 0] -= space["bnd_r"]

    np.subtract(values[:, -1], ambient_value, out=space["bnd_r2"])
    space["bnd_r2"] *= space["surf_r"]
    space["bnd_r2"] /= volumes[:, -1]
    result[:, -1] -= space["bnd_r2"]
    return result


def right_hand_side(
    state: np.ndarray,
    time: float,
    n_radial: int,
    n_axial: int,
    dr: float,
    dz: float,
) -> np.ndarray:
    node_count = (n_radial + 1) * (n_axial + 1)
    temperature = state[:node_count].reshape(n_radial + 1, n_axial + 1)
    concentration = state[node_count:].reshape(
        n_radial + 1, n_axial + 1
    )

    air_temperature_value = air_temperature(time)
    air_moisture_value = air_moisture(time)
    density, heat_capacity, conductivity = material_properties(
        concentration, time
    )
    diffusivity = moisture_diffusivity(concentration, temperature, time)
    geometry = make_geometry(n_radial, n_axial)

    space = workspace(n_radial, n_axial, "rhs")
    thermal_flux = conservative_operator(
        temperature,
        conductivity,
        HEAT_TRANSFER_COEFFICIENT,
        air_temperature_value,
        geometry,
        out=space["flux_heat"],
    )
    d_temperature = thermal_flux / (density * heat_capacity)

    d_concentration = conservative_operator(
        concentration,
        diffusivity,
        MASS_TRANSFER_COEFFICIENT,
        air_moisture_value,
        geometry,
        out=space["flux_mass"],
    )

    return np.concatenate(
        (d_temperature.ravel(), d_concentration.ravel())
    )


def interpolate_to_output_radii(
    radii: np.ndarray, profile: np.ndarray
) -> np.ndarray:
    return np.interp(OUTPUT_RADII, radii, profile)


# ============================================================================
# 二维长时模型与时间推进（原 q3_2d_model.py）
# ============================================================================

PHASE_TRANSITION_START = CRITICAL_TIME - PHASE_TRANSITION_DURATION


PHASE_TRANSITION_END = CRITICAL_TIME


THRESHOLD = 0.15


DT = 2.5


RECORD_INTERVAL = 60.0


MAX_DURATION = 10 * 24 * 3600


N_RADIAL = 24


N_AXIAL = 50


TABLE_INTERVAL = 6 * 3600


TABLE_RADII_CM = np.array([0.0, 0.5, 1.0, 1.5, 2.0])


SOLVER = "bdf"


BDF_RTOL = 2.0e-7


BDF_MAX_STEP = 300.0


IMPLICIT_TOLERANCE = 1.0e-7


IMPLICIT_MAX_ITERATIONS = 20


IMPLICIT_RELAXATION = 0.8


SQRT_THREE = np.sqrt(3.0)


RK4_C = np.array(
    [0.5 - SQRT_THREE / 6.0, 0.5 + SQRT_THREE / 6.0]
)


RK4_A = np.array(
    [
        [0.25, 0.25 - SQRT_THREE / 6.0],
        [0.25 + SQRT_THREE / 6.0, 0.25],
    ]
)


RK4_B = np.array([0.5, 0.5])


def make_state(n_radial: int, n_axial: int) -> np.ndarray:
    node_count = (n_radial + 1) * (n_axial + 1)
    return np.concatenate(
        (
            np.full(node_count, TEMPERATURE_0),
            np.full(node_count, MOISTURE_0),
        )
    )


def profiles_from_state(
    state: np.ndarray,
    output_radii: np.ndarray,
    radii: np.ndarray,
    axial_positions: np.ndarray,
    n_radial: int,
    n_axial: int,
) -> tuple[np.ndarray, np.ndarray]:
    node_count = (n_radial + 1) * (n_axial + 1)
    temperature = state[:node_count].reshape(n_radial + 1, n_axial + 1)
    concentration = state[node_count:].reshape(
        n_radial + 1, n_axial + 1
    )
    # 口径与问题一、二、四一致：取中截面 z = L/2 的径向剖面（半长 50 段时恰为 0.125 m 节点）。
    mid_index = (n_axial + 1) // 2
    midplane_temperature = temperature[:, mid_index]
    midplane_concentration = concentration[:, mid_index]
    return (
        np.interp(output_radii, radii, midplane_temperature),
        np.interp(output_radii, radii, midplane_concentration),
    )


def implicit_rk4_step(
    state: np.ndarray,
    time: float,
    dr: float,
    dz: float,
    n_radial: int | None = None,
    n_axial: int | None = None,
    dt: float | None = None,
) -> tuple[np.ndarray, int]:
    """Advance one step with the two-stage Gauss-Legendre implicit RK4.

    缺省时取模块常量（故 `--mode temporal` 这类按需覆盖 DT 的检验仍然有效）；
    局部同名变量遮蔽模块常量，函数体无需改动。
    """
    _globals = globals()
    N_RADIAL = _globals["N_RADIAL"] if n_radial is None else n_radial
    N_AXIAL = _globals["N_AXIAL"] if n_axial is None else n_axial
    DT = _globals["DT"] if dt is None else dt
    stage_1 = state.copy()
    stage_2 = state.copy()

    for iteration in range(1, IMPLICIT_MAX_ITERATIONS + 1):
        derivative_1 = right_hand_side(
            stage_1,
            time + RK4_C[0] * DT,
            N_RADIAL,
            N_AXIAL,
            dr,
            dz,
        )
        derivative_2 = right_hand_side(
            stage_2,
            time + RK4_C[1] * DT,
            N_RADIAL,
            N_AXIAL,
            dr,
            dz,
        )
        next_stage_1 = state + DT * (
            RK4_A[0, 0] * derivative_1
            + RK4_A[0, 1] * derivative_2
        )
        next_stage_2 = state + DT * (
            RK4_A[1, 0] * derivative_1
            + RK4_A[1, 1] * derivative_2
        )
        error = max(
            float(np.max(np.abs(next_stage_1 - stage_1))),
            float(np.max(np.abs(next_stage_2 - stage_2))),
        )
        stage_1 = (
            (1.0 - IMPLICIT_RELAXATION) * stage_1
            + IMPLICIT_RELAXATION * next_stage_1
        )
        stage_2 = (
            (1.0 - IMPLICIT_RELAXATION) * stage_2
            + IMPLICIT_RELAXATION * next_stage_2
        )
        if error < IMPLICIT_TOLERANCE:
            break
    else:
        raise RuntimeError(
            "Implicit RK4 fixed-point iteration did not converge"
        )

    derivative_1 = right_hand_side(
        stage_1,
        time + RK4_C[0] * DT,
        N_RADIAL,
        N_AXIAL,
        dr,
        dz,
    )
    derivative_2 = right_hand_side(
        stage_2,
        time + RK4_C[1] * DT,
        N_RADIAL,
        N_AXIAL,
        dr,
        dz,
    )
    next_state = state + DT * (
        RK4_B[0] * derivative_1 + RK4_B[1] * derivative_2
    )
    return next_state, iteration


def sparsity_pattern(n_radial: int, n_axial: int):
    """状态向量的雅可比稀疏结构：同一场按网格连通，两场互相耦合（T↔C）。"""
    nr1, nz1 = n_radial + 1, n_axial + 1
    ids = np.arange(nr1 * nz1).reshape(nr1, nz1)
    i = np.r_[ids[:-1].ravel(), ids[:, :-1].ravel()]
    j = np.r_[ids[1:].ravel(), ids[:, 1:].ravel()]
    rows = np.r_[ids.ravel(), i, j]
    cols = np.r_[ids.ravel(), j, i]
    pattern = coo_matrix(
        (np.ones(rows.size), (rows, cols)), shape=(nr1 * nz1, nr1 * nz1)
    ).tocsr()
    return bmat([[pattern, pattern], [pattern, pattern]], format="csr")


def simulate_bdf(
    n_radial: int | None = None,
    n_axial: int | None = None,
    rtol: float = BDF_RTOL,
    max_step: float = BDF_MAX_STEP,
    threshold: float | None = None,
    max_duration: float | None = None,
    record_interval: float | None = None,
) -> dict[str, object]:
    """用 scipy 的自适应 BDF 求解同一模型（与 solve_problem4.py 同一套做法）。

    返回结构与 simulate() 兼容，便于复用 write_result_workbook。
    达标判据同样取“严格低于阈值”的下一个整秒。
    """
    n_radial = N_RADIAL if n_radial is None else n_radial
    n_axial = N_AXIAL if n_axial is None else n_axial
    threshold = THRESHOLD if threshold is None else threshold
    max_duration = MAX_DURATION if max_duration is None else max_duration
    record_interval = RECORD_INTERVAL if record_interval is None else record_interval

    radii = np.linspace(0.0, RADIUS, n_radial + 1)
    axial_positions = np.linspace(0.0, LENGTH, n_axial + 1)
    output_radii = OUTPUT_RADII
    dr = RADIUS / n_radial
    dz = LENGTH / n_axial
    node_count = (n_radial + 1) * (n_axial + 1)
    mid_index = (n_axial + 1) // 2

    def residual(time, vector):
        return right_hand_side(vector, time, n_radial, n_axial, dr, dz)

    def crossed(time, vector):
        return float(vector[node_count:].max() - threshold)

    crossed.terminal = True
    crossed.direction = -1
    options = {
        "method": "BDF",
        "rtol": rtol,
        "atol": rtol * 0.01,
        "jac_sparsity": sparsity_pattern(n_radial, n_axial),
        "max_step": max_step,
    }

    start = make_state(n_radial, n_axial)
    first = solve_ivp(
        residual, (0.0, max_duration), start, events=crossed,
        dense_output=True, **options,
    )
    if first.t_events[0].size == 0:
        raise RuntimeError(
            "BDF 求解在 max_duration 内未达标，无法确定烘干结束时刻"
        )

    crossing_time = float(first.t_events[0][0])
    strict_stop_time = float(math.floor(crossing_time) + 1.0)

    tail = None
    if strict_stop_time > float(first.t[-1]) + 1.0e-9:
        tail = solve_ivp(
            residual, (float(first.t[-1]), strict_stop_time),
            first.y[:, -1], dense_output=True, **options,
        )

    def state_at(time):
        if tail is not None and time > float(first.t[-1]) + 1.0e-9:
            return tail.sol(time)
        return first.sol(time)

    times = np.r_[
        np.arange(0.0, strict_stop_time, record_interval), strict_stop_time
    ]
    temperature_profile = np.empty((times.size, output_radii.size))
    concentration_profile = np.empty((times.size, output_radii.size))
    for index, time in enumerate(times):
        vector = state_at(float(time))
        temperature = vector[:node_count].reshape(n_radial + 1, n_axial + 1)
        concentration = vector[node_count:].reshape(n_radial + 1, n_axial + 1)
        temperature_profile[index] = np.interp(
            output_radii, radii, temperature[:, mid_index]
        )
        concentration_profile[index] = np.interp(
            output_radii, radii, concentration[:, mid_index]
        )

    return {
        "time_s": times,
        "radius_cm": output_radii * 100.0,
        "temperature_c": temperature_profile,
        "moisture_kg_per_kg": concentration_profile,
        "final_time_s": strict_stop_time,
        "final_state": state_at(strict_stop_time),
        "radii": radii,
        "axial_positions": axial_positions,
        "n_radial": n_radial,
        "n_axial": n_axial,
        "solver": "bdf",
        "crossing_time_s": crossing_time,
        "steps": int(first.t.size + (0 if tail is None else tail.t.size - 1)),
        "nfev": int(first.nfev + (0 if tail is None else tail.nfev)),
        "average_iterations": float("nan"),
    }


def solve(solver: str | None = None, **kwargs) -> dict[str, object]:
    """按 SOLVER（或显式传入的 solver）选择时间推进。

    'bdf' → simulate_bdf（生产默认，变阶变步长自适应）；'rk4' → simulate（可选项，
    固定步长 + 阻尼固定点迭代）。kwargs 原样透传：n_radial/n_axial/threshold/
    max_duration/record_interval 两条路径都支持，dt 仅 RK4、rtol/max_step 仅 BDF。
    """
    name = (SOLVER if solver is None else solver).lower()
    if name == "bdf":
        return simulate_bdf(**kwargs)
    if name == "rk4":
        return simulate(**kwargs)
    raise ValueError(f"未知求解器：{name!r}（可选 'bdf' 或 'rk4'）")


def simulate(
    n_radial: int | None = None,
    n_axial: int | None = None,
    dt: float | None = None,
    threshold: float | None = None,
    max_duration: float | None = None,
    record_interval: float | None = None,
) -> dict[str, object]:
    """二阶段 Gauss-Legendre 隐式 RK4 推进（可选路径）。

    参数缺省时等于模块常量。函数体内以下同名的局部名遮蔽模块常量，因此函数体
    无需改动即可参数化，默认调用与原实现逐位一致。
    """
    _globals = globals()
    N_RADIAL = _globals["N_RADIAL"] if n_radial is None else n_radial
    N_AXIAL = _globals["N_AXIAL"] if n_axial is None else n_axial
    DT = _globals["DT"] if dt is None else dt
    THRESHOLD = _globals["THRESHOLD"] if threshold is None else threshold
    MAX_DURATION = _globals["MAX_DURATION"] if max_duration is None else max_duration
    RECORD_INTERVAL = _globals["RECORD_INTERVAL"] if record_interval is None else record_interval

    radii = np.linspace(0.0, RADIUS, N_RADIAL + 1)
    axial_positions = np.linspace(0.0, LENGTH, N_AXIAL + 1)
    output_radii = OUTPUT_RADII
    dr = RADIUS / N_RADIAL
    dz = LENGTH / N_AXIAL

    state = make_state(N_RADIAL, N_AXIAL)
    recorded_times = [0.0]
    recorded_temperature = []
    recorded_concentration = []
    temperature_profile, concentration_profile = profiles_from_state(
        state,
        output_radii,
        radii,
        axial_positions,
        N_RADIAL,
        N_AXIAL,
    )
    recorded_temperature.append(temperature_profile)
    recorded_concentration.append(concentration_profile)

    time = 0.0
    previous_time = 0.0
    previous_state = state.copy()
    previous_max_concentration = float(state[-len(state) // 2 :].max())
    steps_per_record = int(round(RECORD_INTERVAL / DT))
    step = 0
    final_state = state
    final_time = 0.0
    iteration_total = 0
    completed_steps = 0

    while time < MAX_DURATION:
        previous_time = time
        previous_state = state.copy()
        state, iterations = implicit_rk4_step(
            state, time, dr, dz,
            n_radial=N_RADIAL, n_axial=N_AXIAL, dt=DT,
        )
        iteration_total += iterations
        completed_steps += 1
        time += DT
        step += 1
        current_max_concentration = float(state[
            (N_RADIAL + 1) * (N_AXIAL + 1) :
        ].max())

        if step % steps_per_record == 0:
            temperature_profile, concentration_profile = (
                profiles_from_state(
                    state,
                    output_radii,
                    radii,
                    axial_positions,
                    N_RADIAL,
                    N_AXIAL,
                )
            )
            recorded_times.append(time)
            recorded_temperature.append(temperature_profile)
            recorded_concentration.append(concentration_profile)

        if current_max_concentration < THRESHOLD:
            # 达标判据与问题四一致：题目要求“低于 0.15”，故取交叉时刻之后的
            # 下一个整秒（该时刻最大含水率严格小于阈值）。
            denominator = (
                previous_max_concentration - current_max_concentration
            )
            if abs(denominator) < 1.0e-15:
                fraction = 0.0
            else:
                fraction = (
                    previous_max_concentration - THRESHOLD
                ) / denominator
            fraction = float(np.clip(fraction, 0.0, 1.0))
            crossing_time = previous_time + fraction * DT
            strict_stop_time = float(math.floor(crossing_time) + 1.0)
            # 推进到 strict_stop_time 之后，再做线性插值（保证是插值而非外推）。
            while time < strict_stop_time:
                previous_time = time
                previous_state = state.copy()
                state, iterations = implicit_rk4_step(state, time, dr, dz)
                iteration_total += iterations
                completed_steps += 1
                time += DT
                step += 1
            span = time - previous_time
            alpha = (
                0.0
                if span <= 0.0
                else float(
                    np.clip(
                        (strict_stop_time - previous_time) / span, 0.0, 1.0
                    )
                )
            )
            final_state = (1.0 - alpha) * previous_state + alpha * state
            final_time = strict_stop_time
            temperature_profile, concentration_profile = (
                profiles_from_state(
                    final_state,
                    output_radii,
                    radii,
                    axial_positions,
                    N_RADIAL,
                    N_AXIAL,
                )
            )
            if final_time - recorded_times[-1] > 1.0e-9:
                recorded_times.append(final_time)
                recorded_temperature.append(temperature_profile)
                recorded_concentration.append(concentration_profile)
            break

        previous_max_concentration = current_max_concentration

    return {
        "time_s": np.asarray(recorded_times),
        "radius_cm": output_radii * 100.0,
        "temperature_c": np.asarray(recorded_temperature),
        "moisture_kg_per_kg": np.asarray(recorded_concentration),
        "final_time_s": final_time,
        "final_state": final_state,
        "radii": radii,
        "axial_positions": axial_positions,
        "n_radial": N_RADIAL,
        "n_axial": N_AXIAL,
        "average_iterations": iteration_total / max(completed_steps, 1),
        "solver": "rk4",
        "steps": int(completed_steps),
        # 每次迭代 2 次右端求值，循环后另加 2 次（精确计数）
        "nfev": int(2 * iteration_total + 2 * completed_steps),
    }


def write_result_workbook(
    times: np.ndarray,
    concentrations: np.ndarray,
    output_path: "Path | None" = None,
) -> Path:
    template_path = PROJECT / "附件" / "附件3" / "result3.xlsx"
    output_path = (
        PROJECT / "result3.xlsx" if output_path is None else Path(output_path)
    )
    workbook = load_workbook(template_path)
    worksheet = workbook["Sheet1"]
    worksheet.delete_rows(2, worksheet.max_row)

    headers = ["时间\\到药材中心的距离"] + [
        float(f"{radius:.1f}") for radius in OUTPUT_RADII * 100.0
    ]
    for column, value in enumerate(headers, start=1):
        cell = worksheet.cell(row=1, column=column, value=value)
        cell.number_format = "0.0"

    for row_index, (time, profile) in enumerate(
        zip(times, concentrations), start=2
    ):
        time_cell = worksheet.cell(
            row=row_index, column=1, value=int(round(time))
        )
        time_cell.number_format = "0"
        for column, value in enumerate(profile, start=2):
            cell = worksheet.cell(
                row=row_index,
                column=column,
                value=float(round(value, 4)),
            )
            cell.number_format = "0.0000"

    worksheet.freeze_panes = "B2"
    workbook.save(output_path)
    workbook.close()
    return output_path


def run_solve(
    solver: str | None = None,
    write_xlsx: bool = True,
    xlsx_path: "Path | None" = None,
) -> None:
    result = solve(solver)
    times = result["time_s"]
    concentrations = result["moisture_kg_per_kg"]
    temperatures = result["temperature_c"]
    output_path = (
        write_result_workbook(times, concentrations, xlsx_path)
        if write_xlsx
        else None
    )


    table_indices = [
        int(np.argmin(np.abs(result["radius_cm"] - radius)))
        for radius in TABLE_RADII_CM
    ]
    table_rows = []
    for table_time in np.arange(
        TABLE_INTERVAL, times[-1], TABLE_INTERVAL
    ):
        index = int(np.argmin(np.abs(times - table_time)))
        table_rows.append(
            (
                times[index] / 3600.0,
                concentrations[index, table_indices],
            )
        )
    table_rows.append(
        (
            times[-1] / 3600.0,
            concentrations[-1, table_indices],
        )
    )


    print(f"dry time: {times[-1] / 3600.0:.6f} h")
    print(f"final max moisture: {concentrations[-1].max():.6f}")
    if result.get("solver") == "bdf":
        print(
            f"solver: BDF (rtol={BDF_RTOL:g}, max_step={BDF_MAX_STEP:g} s), "
            f"steps: {result['steps']}, nfev: {result['nfev']}"
        )
    else:
        print(
            f"solver: RK4 (dt={DT:g} s), average implicit iterations: "
            f"{result['average_iterations']:.3f}, nfev: {result['nfev']}"
        )
    print("\n每 6 h 的中截面含水率（kg/kg；径向 0 / 0.5 / 1.0 / 1.5 / 2.0 cm）：")
    for table_time_h, table_profile in table_rows:
        print(f"  {table_time_h:>8.4f} h  " + "  ".join(f"{value:.4f}" for value in table_profile))

    print(f"result workbook: {output_path}")


# ============================================================================
# 检验辅助
# ============================================================================

# ============================================================ 检验辅助：临时全局替换

@contextmanager
def _overrides(**values):
    """临时替换模块级全局（均匀稳态检验用），退出时逐字还原。

    原实现直接给模型模块的属性赋值再手工还原；合并为单文件后，用上下文管理器
    保证异常路径下也会还原，避免污染后续求解。
    """
    missing = object()
    saved = {name: globals().get(name, missing) for name in values}
    globals().update(values)
    try:
        yield
    finally:
        for name, original in saved.items():
            if original is missing:
                globals().pop(name, None)
            else:
                globals()[name] = original


# ============================================================================
# 六项检验（原 6 个 q3_*.py）
# ============================================================================

def check_compare() -> None:
    """RK4 与 BDF 求解器对比（结果打印到终端）。"""
    def main() -> None:
        started = time.perf_counter()

        print("跑 RK4（现行主求解器）…", flush=True)
        mark = time.perf_counter()
        rk4 = simulate()
        time_rk4 = time.perf_counter() - mark
        print("  完成 %.1f s" % time_rk4, flush=True)

        print("跑 BDF（scipy 自适应）…", flush=True)
        mark = time.perf_counter()
        bdf = simulate_bdf()
        time_bdf = time.perf_counter() - mark
        print("  完成 %.1f s" % time_bdf, flush=True)

        t_r = rk4["time_s"]
        t_b = bdf["time_s"]
        C_r, T_r = rk4["moisture_kg_per_kg"], rk4["temperature_c"]
        C_b, T_b = bdf["moisture_kg_per_kg"], bdf["temperature_c"]

        n = int(min(t_r.size, t_b.size))
        dT = float(np.abs(T_r[:n] - T_b[:n]).max())
        dC = float(np.abs(C_r[:n] - C_b[:n]).max())

        rounded_r = np.round(C_r[:n], 4)
        rounded_b = np.round(C_b[:n], 4)
        diff_cells = int(np.count_nonzero(rounded_r != rounded_b))
        rounded_gap = float(np.abs(rounded_r - rounded_b).max())

        print()
        print(f"烘干时长：RK4 = {rk4['final_time_s'] / 3600:.6f} h，BDF = {bdf['final_time_s'] / 3600:.6f} h")
        print(f"耗时：RK4 = {time_rk4:.1f} s，BDF = {time_bdf:.1f} s，加速 {time_rk4 / time_bdf:.1f} 倍")
        print(f"共同 60 s 网格前 {n} 行：max|ΔT| = {dT:.3e} °C，max|ΔC| = {dC:.3e} kg/kg")
        print(f"四位小数交付口径：{diff_cells} / {rounded_r.size} 个单元格不同（最大 {rounded_gap:.0e}）")
        print(f"内部步数：RK4 = {rk4['steps']} 步，BDF = {bdf['steps']} 步、nfev {bdf['nfev']}")
        print(f"总用时 {time.perf_counter() - started:.1f} s")


    main()


def check_grid() -> None:
    """网格收敛性检验（结果打印到终端）。"""
    REFINEMENT_RATIO = 1.5


    GRID_LEVELS = [
        ("粗网格", 10, 20),
        ("中网格", 15, 30),
        ("细网格", 20, 40),
    ]


    def main() -> None:
        records = []

        # 直接按参数调用生产求解器，不再改写模块全局常量（原先的 try/finally 全局
        # 改写既不可并行、也容易漏还原）。
        for name, n_radial, n_axial in GRID_LEVELS:
            start_time = time.perf_counter()
            result = solve(n_radial=n_radial, n_axial=n_axial)
            elapsed = time.perf_counter() - start_time
            times = result["time_s"]
            concentrations = result["moisture_kg_per_kg"]
            records.append(
                {
                    "name": name,
                    "n_radial": n_radial,
                    "n_axial": n_axial,
                    "nodes": (n_radial + 1) * (n_axial + 1),
                    "solver": str(result.get("solver", "?")),
                    "steps": int(result.get("steps", 0)),
                    "nfev": int(result.get("nfev", 0)),
                    "dry_hours": float(times[-1] / 3600.0),
                    "final_max": float(concentrations[-1].max()),
                    "elapsed": elapsed,
                }
            )

        coarse = records[0]["dry_hours"]
        medium = records[1]["dry_hours"]
        fine = records[2]["dry_hours"]
        numerator = fine - medium
        denominator = medium - coarse
        if (
            abs(numerator) < 1.0e-14
            or abs(denominator) < 1.0e-14
            or np.sign(numerator) != np.sign(denominator)
        ):
            order = float("nan")
        else:
            order = float(
                np.log(abs(denominator / numerator))
                / np.log(REFINEMENT_RATIO)
            )
        if not np.isfinite(order) or np.isclose(order, 0.0):
            richardson = float("nan")
        else:
            richardson = float(
                fine + (fine - medium) / (REFINEMENT_RATIO**order - 1)
            )

        for record in records:
            print(
                record["name"],
                f"grid={record['n_radial']}x{record['n_axial']}",
                f"dry={record['dry_hours']:.6f} h",
                f"steps={record['steps']}",
                f"elapsed={record['elapsed']:.2f} s",
            )
        print(f"observed order: {order:.6f}")
        print(f"Richardson dry time: {richardson:.6f} h")

    main()


def check_temporal() -> None:
    """时间步长收敛性检验（RK4 路径；结果打印到终端）。"""
    TEST_TIME = 6 * 3600.0


    TIME_STEPS = np.array([2.5, 1.25, 0.625])


    REFINEMENT_RATIO = 2.0


    def run_to_time(dt: float) -> dict[str, object]:
        radii = np.linspace(0.0, RADIUS, N_RADIAL + 1)
        axial_positions = np.linspace(
            0.0, LENGTH, N_AXIAL + 1
        )
        dr = RADIUS / N_RADIAL
        dz = LENGTH / N_AXIAL
        state = make_state(N_RADIAL, N_AXIAL)

        start = time.perf_counter()
        steps = int(round(TEST_TIME / dt))
        current_time = 0.0
        for _ in range(steps):
            state, _ = implicit_rk4_step(
                state, current_time, dr, dz, dt=dt
            )
            current_time += dt

        node_count = (N_RADIAL + 1) * (N_AXIAL + 1)
        concentration = state[node_count:].reshape(
            N_RADIAL + 1, N_AXIAL + 1
        )
        return {
            "dt": dt,
            "concentration": concentration,
            "elapsed": time.perf_counter() - start,
            "radii": radii,
            "axial_positions": axial_positions,
        }


    def weighted_l2_error(
        values: np.ndarray,
        reference: np.ndarray,
        radii: np.ndarray,
        axial_positions: np.ndarray,
    ) -> float:
        radial_area = np.trapezoid(
            radii[:, None] * np.ones_like(values), axial_positions, axis=1
        )
        weights = np.repeat(radial_area[:, None], values.shape[1], axis=1)
        numerator = np.sum(weights * (values - reference) ** 2)
        denominator = np.sum(weights)
        return float(np.sqrt(numerator / denominator))


    def main() -> None:
        results = [run_to_time(dt) for dt in TIME_STEPS]
        radii = results[-1]["radii"]
        axial_positions = results[-1]["axial_positions"]

        pair_l2_errors = []
        pair_max_errors = []
        for index in range(len(results) - 1):
            pair_l2_errors.append(
                weighted_l2_error(
                    results[index]["concentration"],
                    results[index + 1]["concentration"],
                    radii,
                    axial_positions,
                )
            )
            pair_max_errors.append(
                float(
                    np.max(
                        np.abs(
                            results[index]["concentration"]
                            - results[index + 1]["concentration"]
                        )
                    )
                )
            )

        observed_orders = []
        for index in range(len(pair_l2_errors) - 1):
            coarse_error = pair_l2_errors[index]
            fine_error = pair_l2_errors[index + 1]
            if fine_error <= 0.0:
                order = float("nan")
            else:
                order = float(
                    np.log(coarse_error / fine_error)
                    / np.log(REFINEMENT_RATIO)
                )
            observed_orders.append(order)


        for index, result in enumerate(results):
            if index < len(pair_l2_errors):
                l2_error = pair_l2_errors[index]
                max_error = pair_max_errors[index]
            else:
                l2_error = 0.0
                max_error = 0.0
            print(
                f"dt={result['dt']:.4f} "
                f"L2={l2_error:.6e} "
                f"max={max_error:.6e}"
            )
        print(f"observed orders: {observed_orders}")

    main()


def check_conservation() -> None:
    """全局水分守恒检验（结果打印到终端）。"""
    def moisture_content(state: np.ndarray) -> float:
        n_radial = N_RADIAL
        n_axial = N_AXIAL
        node_count = (n_radial + 1) * (n_axial + 1)
        concentration = state[node_count:].reshape(
            n_radial + 1, n_axial + 1
        )
        radii = np.linspace(0.0, RADIUS, n_radial + 1)
        axial_positions = np.linspace(0.0, LENGTH, n_axial + 1)
        radial_integral = np.trapezoid(
            concentration * radii[:, None], axial_positions, axis=1
        )
        return float(
            2.0
            * np.pi
            * np.trapezoid(radial_integral, radii)
        )


    def surface_flux(state: np.ndarray, time: float) -> float:
        n_radial = N_RADIAL
        n_axial = N_AXIAL
        node_count = (n_radial + 1) * (n_axial + 1)
        concentration = state[node_count:].reshape(
            n_radial + 1, n_axial + 1
        )
        radii = np.linspace(0.0, RADIUS, n_radial + 1)
        axial_positions = np.linspace(0.0, LENGTH, n_axial + 1)
        ambient_moisture = air_moisture(time)
        hm = MASS_TRANSFER_COEFFICIENT

        lateral = (
            2.0
            * np.pi
            * RADIUS
            * hm
            * np.trapezoid(
                ambient_moisture - concentration[-1], axial_positions
            )
        )
        lower_end = (
            2.0
            * np.pi
            * hm
            * np.trapezoid(
                radii * (ambient_moisture - concentration[:, 0]), radii
            )
        )
        upper_end = (
            2.0
            * np.pi
            * hm
            * np.trapezoid(
                radii * (ambient_moisture - concentration[:, -1]), radii
            )
        )
        return float(lateral + lower_end + upper_end)


    def main() -> None:
        radii = np.linspace(0.0, RADIUS, N_RADIAL + 1)
        axial_positions = np.linspace(
            0.0, LENGTH, N_AXIAL + 1
        )
        dr = RADIUS / N_RADIAL
        dz = LENGTH / N_AXIAL

        state = make_state(N_RADIAL, N_AXIAL)
        initial_moisture = moisture_content(state)
        cumulative_surface_flux = 0.0
        time = 0.0
        step = 0
        final_state = state
        final_time = 0.0

        while time < MAX_DURATION:
            previous_state = state.copy()
            previous_flux = surface_flux(previous_state, time)
            next_state, _ = implicit_rk4_step(
                state, time, dr, dz
            )
            time += DT
            next_flux = surface_flux(next_state, time)
            cumulative_surface_flux += (
                0.5 * (previous_flux + next_flux) * DT
            )
            state = next_state
            step += 1

            node_count = (N_RADIAL + 1) * (N_AXIAL + 1)
            maximum_concentration = float(state[node_count:].max())
            if maximum_concentration <= THRESHOLD:
                previous_maximum = float(previous_state[node_count:].max())
                denominator = previous_maximum - maximum_concentration
                fraction = (
                    0.0
                    if abs(denominator) < 1.0e-15
                    else (previous_maximum - THRESHOLD) / denominator
                )
                fraction = float(np.clip(fraction, 0.0, 1.0))
                final_state = (
                    (1.0 - fraction) * previous_state
                    + fraction * state
                )
                final_time = time - DT + fraction * DT
                crossing_flux = surface_flux(final_state, final_time)
                cumulative_surface_flux += (
                    fraction * DT *
                    0.5 * (previous_flux + crossing_flux)
                )
                break
            final_state = state
            final_time = time

        final_moisture = moisture_content(final_state)
        moisture_change = final_moisture - initial_moisture
        relative_error = abs(
            moisture_change - cumulative_surface_flux
        ) / max(abs(cumulative_surface_flux), 1.0e-16)
        normalized_error = abs(
            moisture_change - cumulative_surface_flux
        ) / max(abs(initial_moisture - final_moisture), 1.0e-16)


        print(f"initial moisture: {initial_moisture:.10e}")
        print(f"final moisture: {final_moisture:.10e}")
        print(f"moisture change: {moisture_change:.10e}")
        print(f"surface flux: {cumulative_surface_flux:.10e}")
        print(f"relative error: {relative_error:.6e}")
        print(f"normalized error: {normalized_error:.6e}")

    main()


def check_steady() -> None:
    """均匀稳态检验（结果打印到终端）。"""
    UNIFORM_TEMPERATURE = 50.0


    UNIFORM_MOISTURE = 0.05


    TEST_DURATION = 24 * 3600


    def main() -> None:
        radii = np.linspace(0.0, RADIUS, N_RADIAL + 1)
        axial_positions = np.linspace(
            0.0, LENGTH, N_AXIAL + 1
        )
        dr = RADIUS / N_RADIAL
        dz = LENGTH / N_AXIAL
        node_count = (N_RADIAL + 1) * (N_AXIAL + 1)

        state = np.concatenate(
            (
                np.full(node_count, UNIFORM_TEMPERATURE),
                np.full(node_count, UNIFORM_MOISTURE),
            )
        )
        with _overrides(
            air_temperature=lambda _: UNIFORM_TEMPERATURE,
            air_moisture=lambda _: UNIFORM_MOISTURE,
        ):
            initial_rhs = right_hand_side(
                state, 0.0, N_RADIAL, N_AXIAL, dr, dz
            )
            initial_max_rhs = float(np.max(np.abs(initial_rhs)))

            step_count = int(round(TEST_DURATION / DT))
            for _ in range(step_count):
                state, _ = implicit_rk4_step(
                    state, 0.0, dr, dz
                )

        temperature = state[:node_count].reshape(
            N_RADIAL + 1, N_AXIAL + 1
        )
        concentration = state[node_count:].reshape(
            N_RADIAL + 1, N_AXIAL + 1
        )
        maximum_temperature_drift = float(
            np.max(np.abs(temperature - UNIFORM_TEMPERATURE))
        )
        maximum_moisture_drift = float(
            np.max(np.abs(concentration - UNIFORM_MOISTURE))
        )


        print(f"initial RHS max: {initial_max_rhs:.6e}")
        print(
            f"24 h temperature drift: {maximum_temperature_drift:.6e} °C"
        )
        print(
            f"24 h moisture drift: {maximum_moisture_drift:.6e} kg/kg"
        )

    main()


def check_rebound() -> None:
    """表面含水率回升检验（结果打印到终端）。"""
    TEST_DURATION = 2 * 3600


    GRID_LEVELS = [
        ("粗网格", 10, 20),
        ("中网格", 15, 30),
        ("细网格", 20, 40),
    ]


    def run_case(
        name: str, n_radial: int, n_axial: int
    ) -> dict[str, object]:
        result = simulate(
            n_radial=n_radial,
            n_axial=n_axial,
            dt=2.5,
            max_duration=TEST_DURATION,
            threshold=-1.0,
        )

        times = result["time_s"]
        surface = result["moisture_kg_per_kg"][:, -1]
        differences = np.diff(surface)
        maximum_increase_index = int(np.argmax(differences))
        return {
            "name": name,
            "n_radial": n_radial,
            "n_axial": n_axial,
            "maximum_rebound": float(differences[maximum_increase_index]),
            "rebound_time": float(times[maximum_increase_index]),
            "surface_at_rebound": float(
                surface[maximum_increase_index]
            ),
            "surface_final": float(surface[-1]),
        }


    def main() -> None:
        results = [
            run_case(name, n_radial, n_axial)
            for name, n_radial, n_axial in GRID_LEVELS
        ]


        for result in results:
            print(
                result["name"],
                f"rebound={result['maximum_rebound']:.6f}",
                f"time={result['rebound_time']:.1f} s",
            )

    main()


# ============================================================================
# 命令行入口（原 solve_result3.py）
# ============================================================================

# ============================================================ 命令行入口

MODES = {
    "compare": ("RK4 与 BDF 求解器对比", check_compare),
    "grid": ("网格收敛性检验", check_grid),
    "temporal": ("时间步长收敛性检验（RK4 路径）", check_temporal),
    "conservation": ("全局水分守恒检验", check_conservation),
    "steady": ("均匀稳态检验", check_steady),
    "rebound": ("表面含水率回升检验", check_rebound),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="solve_problem3.py",
        description="第三问：二维轴对称长时干燥模型（求解 + 检验，唯一入口）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        default="solve",
        choices=["solve", *MODES, "all"],
        help="solve=求解并写交付件（默认）；all=求解 + 全部检验",
    )
    parser.add_argument(
        "--solver",
        choices=["bdf", "rk4"],
        default=None,
        help="时间推进方式，默认取模块常量 SOLVER（bdf）",
    )
    parser.add_argument("--no-xlsx", action="store_true", help="不写 result3.xlsx")
    parser.add_argument(
        "--xlsx", type=Path, default=None, help="把交付件写到指定路径（默认项目根 result3.xlsx）"
    )
    args = parser.parse_args()

    if args.mode in ("solve", "all"):
        run_solve(args.solver, write_xlsx=not args.no_xlsx, xlsx_path=args.xlsx)

    if args.mode == "all":
        for label, function in MODES.values():
            print(f"\n{'=' * 30} {label} {'=' * 30}")
            function()
    elif args.mode != "solve":
        MODES[args.mode][1]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())