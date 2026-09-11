"""问题二：计算三维温度、水分进程并写出 result2.xlsx。

运行：python solve_result2.py
依赖：numpy、scipy、openpyxl。题目附件须位于上级目录的“附件”文件夹。
"""
from pathlib import Path
import argparse
import sys
import time

WORKSPACE_DEPS = Path(__file__).resolve().parents[1] / ".p2deps"
if WORKSPACE_DEPS.exists():
    sys.path.insert(0, str(WORKSPACE_DEPS))

import numpy as np
import openpyxl
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import LinearOperator, cg
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parent          # .../CUMCM/Problem II
PROJECT = ROOT.parent                           # .../CUMCM
ATTACHMENTS = PROJECT / "附件"

# Savitzky-Golay 保形滤波参数（与论文一致：窗宽 31 点 = 31 min，三阶多项式）
SMOOTH_WIN, SMOOTH_ORDER = 31, 3


def read_air_data(smooth=True):
    """读入附件1；smooth=True 时先做 Savitzky-Golay 保形滤波。"""
    wb = openpyxl.load_workbook(ATTACHMENTS / "附件1.xlsx", read_only=True, data_only=True)
    data = np.asarray(list(wb.active.values)[1:], dtype=float)
    wb.close()
    if smooth:
        data[:, 1] = savgol_filter(data[:, 1], SMOOTH_WIN, SMOOTH_ORDER)
        data[:, 2] = savgol_filter(data[:, 2], SMOOTH_WIN, SMOOTH_ORDER)
    return data


def detect_critical_time(air, tail_start=10800, window=1200, persistence=1800):
    """由平台接近程度和升温斜率共同识别预热结束时刻。

    尾段中位数估计平台温度，MAD估计测温波动。候选时刻之前20分钟
    的均温须进入2倍MAD带，回归斜率须不超过“3倍MAD/窗长”，且
    两个条件此后连续保持30分钟。该判据用于离线分段，避免单点越阈。
    """
    t, temperature = air[:, 0], air[:, 1]
    sample_dt = float(np.median(np.diff(t)))
    tail = temperature[t >= tail_start]
    if len(tail) < 5:
        raise ValueError("附件1尾段数据不足，无法估计稳态")
    plateau = float(np.median(tail))
    sigma = float(1.4826 * np.median(np.abs(tail - plateau)))
    window_points = int(round(window / sample_dt)) + 1
    keep_points = int(round(persistence / sample_dt)) + 1
    mean = np.full(len(t), np.nan)
    slope = np.full(len(t), np.nan)
    for j in range(window_points - 1, len(t)):
        sl = slice(j - window_points + 1, j + 1)
        minutes = (t[sl] - t[sl][0]) / 60
        mean[j] = temperature[sl].mean()
        slope[j] = np.polyfit(minutes, temperature[sl], 1)[0]
    band = 2 * sigma
    slope_limit = 3 * sigma / (window / 60)
    stable = (np.abs(mean - plateau) <= band) & (np.abs(slope) <= slope_limit)
    for j in range(window_points - 1, len(t) - keep_points + 1):
        if np.all(stable[j : j + keep_points]):
            return float(t[j]), {
                "plateau": plateau,
                "sigma": sigma,
                "band": band,
                "slope_limit": slope_limit,
                "window_mean": float(mean[j]),
                "window_slope": float(slope[j]),
                "window": window,
                "persistence": persistence,
            }
    raise RuntimeError("附件1中未识别到满足持续性要求的恒温阶段")


def material(T, C, drying_stage):
    """临界时刻前用附件2，临界时刻后用附件3，不混合物性。"""
    if np.min(C) <= 0:
        raise RuntimeError("含水率出现非正值，计算停止")
    if drying_stage:
        rho = 650 + 128 * C
        cp = 1450 + 2736 * C / (C + 1)
        k = 0.21 + 0.38 * C / (C + 1)
        D = 2.4e-3 * np.exp(-0.45 / C - 3850 / (T + 273.15))
    else:
        rho = np.full_like(C, 820.0)
        cp = np.full_like(C, 2600.0)
        k = np.full_like(C, 0.36)
        D = 7e-9 * np.exp(-0.89 / C)
    return rho * cp, k, D


class CylinderGrid:
    """完整圆柱控制体：径向、周向、轴向均保留。"""
    def __init__(self, nr, nphi, nz):
        self.nr, self.nphi, self.nz = nr, nphi, nz
        self.shape = (nr, nphi, nz)
        self.dr, self.dphi, self.dz = 0.02 / nr, 2 * np.pi / nphi, 0.25 / nz
        self.r = (np.arange(nr) + 0.5) * self.dr
        self.z = (np.arange(nz) + 0.5) * self.dz - 0.125
        ids = np.arange(nr * nphi * nz).reshape(self.shape)
        self.volume = np.broadcast_to((self.r * self.dr * self.dphi * self.dz)[:, None, None], self.shape).ravel().copy()

        ii, jj, gg = [], [], []

        def connect(left, right, conductance):
            ii.extend(left.ravel())
            jj.extend(right.ravel())
            gg.extend(np.broadcast_to(conductance, left.shape).ravel())

        connect(ids[:-1], ids[1:], ((np.arange(1, nr) * self.dr) * self.dphi * self.dz / self.dr)[:, None, None])
        connect(ids, np.roll(ids, -1, axis=1), (self.dr * self.dz / (self.r * self.dphi))[:, None, None])
        connect(ids[:, :, :-1], ids[:, :, 1:], (self.r * self.dr * self.dphi / self.dz)[:, None, None])
        self.i, self.j, self.g = np.asarray(ii), np.asarray(jj), np.asarray(gg)

        # 侧面及两个端面是对流边界；轴线没有额外边界面。
        self.boundary_ids = np.r_[ids[-1].ravel(), ids[:, :, 0].ravel(), ids[:, :, -1].ravel()]
        end_area = np.broadcast_to((self.r * self.dr * self.dphi)[:, None], (nr, nphi)).ravel()
        self.boundary_area = np.r_[np.full(nphi * nz, 0.02 * self.dphi * self.dz), end_area, end_area]
        self.boundary_distance = np.r_[np.full(nphi * nz, self.dr / 2), np.full(2 * nr * nphi, self.dz / 2)]

    def matrix(self, coefficient, h):
        g = self.g * 2 * coefficient[self.i] * coefficient[self.j] / (coefficient[self.i] + coefficient[self.j])
        b = self.boundary_area / (self.boundary_distance / coefficient[self.boundary_ids] + 1 / h)
        diagonal = np.bincount(self.i, g, minlength=len(coefficient)) + np.bincount(self.j, g, minlength=len(coefficient))
        diagonal += np.bincount(self.boundary_ids, b, minlength=len(coefficient))
        K = coo_matrix((-np.r_[g, g], (np.r_[self.i, self.j], np.r_[self.j, self.i])), shape=(len(coefficient), len(coefficient))).tocsr()
        return K + diags(diagonal), np.bincount(self.boundary_ids, b, minlength=len(coefficient))

    def implicit_step(self, old, guess, storage, coefficient, h, ambient, dt):
        K, b = self.matrix(coefficient, h)
        m = storage * self.volume / dt
        A = K + diags(m)
        diagonal_inverse = 1 / A.diagonal()
        preconditioner = LinearOperator(A.shape, matvec=lambda x: diagonal_inverse * x)
        value, info = cg(A, m * old + b * ambient, x0=guess, M=preconditioner, rtol=1e-9, atol=1e-12, maxiter=900)
        if info != 0:
            raise RuntimeError(f"线性迭代未收敛（info={info}）")
        return value

    def middle_profile(self, field, coefficient, h, ambient):
        f = field.reshape(self.shape)
        c = coefficient.reshape(self.shape)
        profile = (f[:, 0, self.nz // 2 - 1] + f[:, 0, self.nz // 2]) / 2
        cp = (c[:, 0, self.nz // 2 - 1] + c[:, 0, self.nz // 2]) / 2
        center = (9 * profile[0] - profile[1]) / 8
        surface = ((cp[-1] / (self.dr / 2)) * profile[-1] + h * ambient) / (cp[-1] / (self.dr / 2) + h)
        return np.interp(np.arange(21) * 0.001, np.r_[0, self.r, 0.02], np.r_[center, profile, surface])


def solve(nr=20, nphi=6, nz=32, dt=5, end=10800, export=True):
    grid = CylinderGrid(nr, nphi, nz)
    # 对流边界条件：用 Savitzky-Golay 平滑后的环境数据。
    # 阶段分界时刻的统计判定：用原始数据——σ 应反映实测噪声；若改用平滑序列
    # 的 MAD，量到的只是滤波器自身的平坦度（σ 会被压小约 7 倍，t_c 随之后移）。
    air = read_air_data(smooth=True)
    critical_time, critical = detect_critical_time(read_air_data(smooth=False))
    print(
        f"识别临界时刻 {critical_time:.0f} s（{critical_time / 3600:.4f} h）："
        f"平台 {critical['plateau']:.4f} ℃，窗内斜率 {critical['window_slope']:.4f} ℃/min"
    )
    T = np.full(np.prod(grid.shape), 28.0)
    C = np.full_like(T, 2.55)
    recorded_t, recorded_T, recorded_C = [0.0], [np.full(21, 28.0)], [np.full(21, 2.55)]
    started = time.time()

    while recorded_t[-1] < end:
        old_time = recorded_t[-1]
        t = min(old_time + dt, end)
        if old_time < critical_time < t:
            t = critical_time
        actual_dt = t - old_time
        air_T = np.interp(t, air[:, 0], air[:, 1])
        air_C = np.interp(t, air[:, 0], air[:, 2])
        old_T, old_C = T.copy(), C.copy()
        drying_stage = old_time >= critical_time

        for _ in range(25):
            capacity, conductivity, _ = material(T, C, drying_stage)
            new_T = grid.implicit_step(old_T, T, capacity, conductivity, 25.0, air_T, actual_dt)
            _, _, diffusivity = material(new_T, C, drying_stage)
            new_C = grid.implicit_step(old_C, C, np.ones_like(C), diffusivity, 8e-7, air_C, actual_dt)
            change = max(np.max(np.abs(new_T - T)) / 50, np.max(np.abs(new_C - C)) / 2.55)
            T, C = new_T, new_C
            if change < 1e-8:
                break
        else:
            raise RuntimeError("热质耦合迭代未收敛")

        _, conductivity, diffusivity = material(T, C, drying_stage)
        recorded_t.append(t)
        recorded_T.append(grid.middle_profile(T, conductivity, 25.0, air_T))
        recorded_C.append(grid.middle_profile(C, diffusivity, 8e-7, air_C))

        if t % 1800 == 0:
            print(f"{t / 3600:.1f} h: 中心温度 {recorded_T[-1][0]:.3f} ℃，中心含水率 {recorded_C[-1][0]:.3f}")

    seconds = np.arange(1, end + 1, dtype=int)
    sampled_T = np.column_stack([np.interp(seconds, recorded_t, np.asarray(recorded_T)[:, j]) for j in range(21)])
    sampled_C = np.column_stack([np.interp(seconds, recorded_t, np.asarray(recorded_C)[:, j]) for j in range(21)])
    if export:
        write_result(seconds, sampled_T, sampled_C)
    print(f"完成，用时 {time.time() - started:.1f} s")
    return seconds, sampled_T, sampled_C


def write_result(seconds, temperature, moisture):
    """保持附件3的两张结果表结构，写到 CUMCM 根目录的 result2.xlsx。"""
    wb = openpyxl.load_workbook(ATTACHMENTS / "附件3" / "result2.xlsx")
    for name, values in (("温度", temperature), ("水分浓度", moisture)):
        ws = wb[name]
        for row in ws.iter_rows():
            for cell in row:
                cell.value = None
        ws.cell(1, 1, "时间/s；距中心距离/cm")
        for col, radius_cm in enumerate(np.arange(21) / 10, start=2):
            ws.cell(1, col, float(radius_cm))
        for row, t in enumerate(seconds, start=2):
            ws.cell(row, 1, int(t))
            for col, value in enumerate(values[row - 2], start=2):
                ws.cell(row, col, round(float(value), 4)).number_format = "0.0000"
        ws.freeze_panes = "B2"
    wb.save(PROJECT / "result2.xlsx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算问题二的三维温度、水分进程")
    parser.add_argument("--nr", type=int, default=20, help="径向控制体数")
    parser.add_argument("--nphi", type=int, default=6, help="周向控制体数")
    parser.add_argument("--nz", type=int, default=32, help="轴向控制体数")
    parser.add_argument("--dt", type=float, default=5, help="时间步长/s")
    parser.add_argument("--end", type=int, default=10800, help="结束时间/s")
    parser.add_argument("--no-export", action="store_true", help="只试算，不写 result2.xlsx")
    args = parser.parse_args()
    if args.nr < 2 or args.nphi < 3 or args.nz < 2 or args.dt <= 0:
        parser.error("网格数过小或时间步长不合法")
    solve(args.nr, args.nphi, args.nz, args.dt, args.end, not args.no_export)
