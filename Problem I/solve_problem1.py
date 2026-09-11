"""
问题1：预热平衡阶段 圆柱形药材的瞬态传热-传质耦合模型
====================================================
几何：无限长圆柱（轴向忽略），半径 R=0.02 m，一维径向 r∈[0,R]

控制方程（柱坐标）：
  传热：rho*cp * dT/dt = k * (1/r) d/dr( r dT/dr )
  传质：dC/dt          = (1/r) d/dr( r*D(C) dC/dr )

初始条件：T(r,0)=28 °C,  C(r,0)=2.55 kg/kg
边界条件：
  中心 r=0：对称  dT/dr=0, dC/dr=0
  表面 r=R：-k dT/dr = h (T - T_air(t))
           -D dC/dr = hm (C - C_air(t))
其中 T_air(t), C_air(t) 为附件1的烘房温度/含湿量。

数值：有限体积法(守恒形式) + theta 隐式格式
  温度：Crank-Nicolson (theta=0.5, 线性)
  水分：Crank-Nicolson (theta=0.5, 非线性 D(C) 用 Picard 迭代)
"""
import numpy as np
import openpyxl
import os, sys
from scipy.signal import savgol_filter

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Savitzky-Golay 保形滤波参数（与论文一致：窗宽 31 点 = 31 min，三阶多项式）
SMOOTH_WIN, SMOOTH_ORDER = 31, 3

BASE = os.path.dirname(os.path.abspath(__file__))          # .../CUMCM/Problem I
PROJECT = os.path.dirname(BASE)                            # .../CUMCM
ATTACH1 = os.path.join(PROJECT, "附件", "附件1.xlsx")
OUT_XLSX = os.path.join(PROJECT, "result1.xlsx")

# ---------------- 物理参数（附录2）----------------
R   = 0.02        # 半径 m
rho = 820.0       # 密度 kg/m^3
cp  = 2600.0      # 比热容 J/(kg·K)
k   = 0.36        # 热传导系数 W/(m·K)
h   = 25.0        # 对流换热系数 W/(m^2·K)
hm  = 8.0e-7      # 对流传质系数 m/s
T0  = 28.0        # 初始温度 °C
C0  = 2.55        # 初始水分浓度(干基) kg/kg


def D_coeff(C):
    """水分浓度扩散系数 D(C) m^2/s"""
    return 7.0e-9 * np.exp(-0.89 / C)


# ---------------- 读附件1：烘房温度/含湿量 ----------------
def load_attachment1(path, smooth=True):
    """读附件1 的烘房温度/含湿量；smooth=True 时先做 Savitzky-Golay 保形滤波。"""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))[1:] 
    t = np.array([r[0] for r in rows], dtype=float)
    Tair = np.array([r[1] for r in rows], dtype=float)
    Cair = np.array([r[2] for r in rows], dtype=float)
    wb.close()
    if smooth:
        Tair = savgol_filter(Tair, SMOOTH_WIN, SMOOTH_ORDER)
        Cair = savgol_filter(Cair, SMOOTH_WIN, SMOOTH_ORDER)
    return t, Tair, Cair


class DryingSim:
    """一维径向瞬态传热-传质求解器（有限体积 + theta 隐式）"""

    def __init__(self, R, rho, cp, k, h, hm, T0, C0, Dfunc, N=400):
        self.R, self.rho, self.cp, self.k = R, rho, cp, k
        self.h, self.hm, self.T0, self.C0 = h, hm, T0, C0
        self.Dfunc = Dfunc
        self.N = N
        self.dr = R / N
        self.r = np.linspace(0.0, R, N + 1)

        # 界面坐标（约去 pi）
        rp = (np.arange(N + 1) + 0.5) * self.dr
        rm = (np.arange(N + 1) - 0.5) * self.dr
        # 控制体体积 V'_i = r_{i+1/2}^2 - r_{i-1/2}^2
        self.V = np.zeros(N + 1)
        self.V[0]   = (self.dr / 2.0) ** 2
        self.V[1:N] = rp[1:N] ** 2 - rm[1:N] ** 2
        self.V[N]   = R ** 2 - rm[N] ** 2
        # 界面面积 A'_{i+1/2}=2*r_{i+1/2}
        self.Ap = 2.0 * rp
        self.Am = 2.0 * rm

    def _thomas(self, a, b, c, d):
        n = len(b)
        cp_ = np.zeros(n); dp_ = np.zeros(n)
        cp_[0] = c[0] / b[0]; dp_[0] = d[0] / b[0]
        for i in range(1, n):
            denom = b[i] - a[i] * cp_[i - 1]
            cp_[i] = c[i] / denom
            dp_[i] = (d[i] - a[i] * dp_[i - 1]) / denom
        x = np.zeros(n); x[-1] = dp_[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp_[i] - cp_[i] * x[i + 1]
        return x

    def solve_heat(self, T_old, dt, Tair_new, Tair_old, theta=0.5):
        """温度一步：Crank-Nicolson"""
        N = self.N; dr = self.dr
        m = self.rho * self.cp * self.V

        a = np.zeros(N + 1); b = np.zeros(N + 1); c = np.zeros(N + 1)
        a[1:N] =  self.k * self.Am[1:N] / dr
        c[1:N] =  self.k * self.Ap[1:N] / dr
        b[1:N] = -self.k * (self.Ap[1:N] + self.Am[1:N]) / dr
        c[0] = self.k * self.Ap[0] / dr
        b[0] = -self.k * self.Ap[0] / dr
        a[N] = self.k * self.Am[N] / dr
        b[N] = -self.k * self.Am[N] / dr - 2.0 * self.R * self.h

        S_next = np.zeros(N + 1); S_next[N] = 2.0 * self.R * self.h * Tair_new
        S_old  = np.zeros(N + 1); S_old[N]  = 2.0 * self.R * self.h * Tair_old

        ATold = np.zeros(N + 1)
        ATold[1:N] = a[1:N] * T_old[0:N - 1] + b[1:N] * T_old[1:N] + c[1:N] * T_old[2:N + 1]
        ATold[0]   = b[0] * T_old[0] + c[0] * T_old[1]
        ATold[N]   = a[N] * T_old[N - 1] + b[N] * T_old[N]

        mdt = m / dt
        lhs_b = mdt - theta * b
        lhs_a = -theta * a
        lhs_c = -theta * c
        rhs   = mdt * T_old + (1.0 - theta) * ATold + theta * S_next + (1.0 - theta) * S_old
        return self._thomas(lhs_a, lhs_b, lhs_c, rhs)

    def solve_mass(self, C_old, dt, Cair_new, Cair_old, theta=0.5):
        """水分一步：theta 隐式(Crank-Nicolson theta=0.5) + Picard 迭代"""
        N = self.N; dr = self.dr
        m = self.V.copy()

        S_next = np.zeros(N + 1); S_next[N] = 2.0 * self.R * self.hm * Cair_new
        S_old  = np.zeros(N + 1); S_old[N]  = 2.0 * self.R * self.hm * Cair_old

        def build(C):
            Cmid = 0.5 * (C[:-1] + C[1:])
            Dp = self.Dfunc(Cmid)
            Dm = np.zeros(N + 1); Dm[1:] = Dp
            a = np.zeros(N + 1); b = np.zeros(N + 1); c = np.zeros(N + 1)
            a[1:N] =  Dm[1:N] * self.Am[1:N] / dr
            c[1:N] =  Dp[1:N] * self.Ap[1:N] / dr
            b[1:N] = -Dp[1:N] * self.Ap[1:N] / dr - Dm[1:N] * self.Am[1:N] / dr
            c[0] = Dp[0] * self.Ap[0] / dr
            b[0] = -Dp[0] * self.Ap[0] / dr
            a[N] = Dm[N] * self.Am[N] / dr
            b[N] = -Dm[N] * self.Am[N] / dr - 2.0 * self.R * self.hm
            return a, b, c

        def apply(a, b, c, u):
            w = np.zeros(N + 1)
            w[1:N] = a[1:N] * u[0:N - 1] + b[1:N] * u[1:N] + c[1:N] * u[2:N + 1]
            w[0] = b[0] * u[0] + c[0] * u[1]
            w[N] = a[N] * u[N - 1] + b[N] * u[N]
            return w

        a0, b0, c0 = build(C_old)
        LCold = apply(a0, b0, c0, C_old)

        mdt = m / dt
        C_new = C_old.copy()
        for _ in range(3):   # Picard 迭代
            a, b, c = build(C_new)
            lhs_b = mdt - theta * b
            lhs_a = -theta * a
            lhs_c = -theta * c
            rhs = mdt * C_old + (1.0 - theta) * LCold + theta * S_next + (1.0 - theta) * S_old
            C_new = self._thomas(lhs_a, lhs_b, lhs_c, rhs)
        return C_new


def run(N=400, dt=0.5, t_end=1800.0, Tair_interp=None, Cair_interp=None):
    """推进 t∈[0,t_end]，返回 (t_out, T_hist, C_hist)"""
    sim = DryingSim(R, rho, cp, k, h, hm, T0, C0, D_coeff, N=N)
    n_steps = int(round(t_end / dt))
    step_per_s = max(1, int(round(1.0 / dt)))
    t_out = np.arange(1, int(t_end) + 1)
    T_hist = np.zeros((len(t_out), N + 1))
    C_hist = np.zeros((len(t_out), N + 1))

    T = np.full(N + 1, T0); C = np.full(N + 1, C0)
    Tair_prev, Cair_prev = Tair_interp(0.0), Cair_interp(0.0)
    out_idx = 0
    for n in range(1, n_steps + 1):
        t_new = n * dt
        Tair_new = Tair_interp(t_new); Cair_new = Cair_interp(t_new)
        T = sim.solve_heat(T, dt, Tair_new, Tair_prev, theta=0.5)
        C = sim.solve_mass(C, dt, Cair_new, Cair_prev, theta=0.5)
        Tair_prev, Cair_prev = Tair_new, Cair_new
        if n % step_per_s == 0 and out_idx < len(t_out):
            T_hist[out_idx] = T; C_hist[out_idx] = C; out_idx += 1
    return t_out, T_hist, C_hist


def main():
    t_air, T_air, C_air = load_attachment1(ATTACH1)
    Tair_interp = lambda tt: np.interp(tt, t_air, T_air)
    Cair_interp = lambda tt: np.interp(tt, t_air, C_air)

    N = 1600
    dt = 0.25
    t_out, T_hist, C_hist = run(N=N, dt=dt, t_end=1800.0,
                                Tair_interp=Tair_interp, Cair_interp=Cair_interp)

    dr = R / N
    r_out_cm = np.arange(0.0, 2.0001, 0.1)
    r_out_idx = (r_out_cm / 100.0 / dr).round().astype(int)

    t_table = [100, 300, 600, 900, 1200, 1500, 1800]
    r_table = [0.0, 0.5, 1.0, 1.5, 2.0]
    r_tbl_idx = [int(x / 100.0 / dr) for x in r_table]

    print("\n===== 表1 药材温度(°C) =====")
    print("时间/s\t" + "\t".join(f"r={x}cm" for x in r_table))
    for tt in t_table:
        i = int(tt) - 1
        vals = [T_hist[i, idx] for idx in r_tbl_idx]
        print(f"{tt}\t" + "\t".join(f"{v:.4f}" for v in vals))

    print("\n===== 表2 药材水分浓度(kg/kg) =====")
    print("时间/s\t" + "\t".join(f"r={x}cm" for x in r_table))
    for tt in t_table:
        i = int(tt) - 1
        vals = [C_hist[i, idx] for idx in r_tbl_idx]
        print(f"{tt}\t" + "\t".join(f"{v:.4f}" for v in vals))

    # 写 result1.xlsx（以附件3 模板为底，保留其表头样式与列宽）
    template = os.path.join(PROJECT, "附件", "附件3", "result1.xlsx")
    wb = openpyxl.load_workbook(template)
    for name in ("温度", "水分浓度"):
        ws = wb[name]
        for row in ws.iter_rows():
            for cell in row:
                cell.value = None
        ws.cell(1, 1, "时间\\到药材中心的距离")
        for j, x in enumerate(r_out_cm):
            ws.cell(1, 2 + j, round(float(x), 1))
    ws1, ws2 = wb["温度"], wb["水分浓度"]
    for i, tt in enumerate(t_out):
        ws1.cell(2 + i, 1, int(tt)); ws2.cell(2 + i, 1, int(tt))
        for j, idx in enumerate(r_out_idx):
            c1 = ws1.cell(2 + i, 2 + j, round(float(T_hist[i, idx]), 4))
            c1.number_format = "0.0000"
            c2 = ws2.cell(2 + i, 2 + j, round(float(C_hist[i, idx]), 4))
            c2.number_format = "0.0000"
    wb.save(OUT_XLSX)
    print("\nsaved ->", OUT_XLSX)


if __name__ == "__main__":
    main()
