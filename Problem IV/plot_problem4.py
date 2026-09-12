# -*- coding: utf-8 -*-
"""问题四绘图：中截面水分浓度时空分布 + 径向剖面（供论文图使用）。

    python plot_problem4.py                 # 生产配置 96x80，约 2 min
    python plot_problem4.py --cached        # 复用 plot_cache.npz

左图：物理坐标 (t, r) 下的中截面水分浓度热力图，白色区域为收缩后已不存在的部分，
      白线为随时间收缩的药材表面 R(t)；
右图：若干典型时刻的中截面径向剖面，每条曲线终止于该时刻的药材表面。
"""
import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from solve_problem4 import DryingModel                      # noqa: E402

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

NR, NZ = 96, 80
RTOL, MAX_STEP = 2e-7, 300.0
PROFILE_TIMES = [6, 12, 24, 36, 48]
CACHE = HERE / "plot_cache.npz"
OUT = HERE.parent.parent.parent / "论文" / "graphics"


def find_project():
    """定位含 附件/ 的项目根目录。"""
    for p in HERE.parents:
        if (p / "附件" / "附件1.xlsx").exists():
            return p
    for p in HERE.parents:
        if (p / "A题" / "附件" / "附件1.xlsx").exists():
            return p / "A题"
    raise FileNotFoundError("未找到含 附件/附件1.xlsx 的项目根目录")


def compute():
    m = DryingModel(find_project(), NR, NZ, shrink=True, preheat=False, smooth=True)
    y0 = np.r_[np.full(m.n, 28.0), np.full(m.n, 2.55)]

    def event(t, y):
        return float(np.max(y[m.n:])) - 0.15
    event.terminal, event.direction = True, -1

    segs, y = [], y0
    for start, end, step in [(0, 14400, min(MAX_STEP, 60)), (14400, 259200, MAX_STEP)]:
        sol = solve_ivp(m.rhs, (start, end), y, method="BDF", rtol=RTOL,
                        atol=RTOL * 0.01, jac_sparsity=m.sparsity,
                        max_step=step, events=event, dense_output=True)
        assert sol.success, sol.message
        segs.append(sol)
        y = sol.y[:, -1]
        if sol.t_events[0].size:
            break
    crossing = float(segs[-1].t_events[0][0])
    stop = float(np.floor(crossing) + 1)
    final = solve_ivp(m.rhs, (crossing, stop), y, method="BDF", rtol=RTOL,
                      atol=RTOL * 0.01, jac_sparsity=m.sparsity, dense_output=True)

    def state(t):
        if t > crossing:
            return final.sol(t)
        return segs[0].sol(t) if t <= 14400 else segs[1].sol(t)

    # 中截面剖面插值到固定物理半径网格，r > R(t) 置 NaN
    r_grid = np.linspace(0.0, 0.021, 43)
    times = np.r_[np.arange(0.0, stop, 600.0), stop]
    mid = np.full((len(times), len(r_grid)), np.nan)
    radii = np.array([m.radius(t) for t in times])
    for k, t in enumerate(times):
        C = state(t)[m.n:].reshape(m.shape)
        prof = C[:, 0]                                  # z=0 即中截面
        rr = m.x * radii[k]
        vals = np.interp(r_grid, rr, prof, left=prof[0], right=np.nan)
        vals[r_grid > radii[k]] = np.nan
        mid[k] = vals

    prof_r = np.linspace(0.0, 0.0135, 28)
    profiles = {}
    for h in PROFILE_TIMES:
        j = int(np.argmin(np.abs(times - h * 3600.0)))
        profiles[h] = mid[j][:len(prof_r)] if False else np.interp(
            prof_r, r_grid, mid[j], left=mid[j][0], right=np.nan)
    np.savez_compressed(CACHE, times=times, mid=mid, radii=radii,
                        r_grid=r_grid, prof_r=prof_r, stop=stop,
                        **{"p%d" % h: profiles[h] for h in PROFILE_TIMES},
                        final_prof=np.interp(prof_r, r_grid, mid[-1],
                                             left=mid[-1][0], right=np.nan))
    return stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true")
    args = ap.parse_args()
    if not args.cached or not CACHE.exists():
        t0 = time.time()
        stop = compute()
        print("计算完成 %.0f s, t_dry = %.4f h" % (time.time() - t0, stop / 3600))

    d = np.load(CACHE)
    times, mid, radii, r_grid, prof_r = (d["times"], d["mid"], d["radii"],
                                         d["r_grid"], d["prof_r"])
    stop = float(d["stop"])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    ax = axes[0]
    pcm = ax.pcolormesh(times / 3600.0, r_grid * 100.0, mid.T, shading="auto",
                        cmap="viridis", vmin=0.0, vmax=2.6)
    ax.fill_between(times / 3600.0, radii * 100.0, 2.15, color="white", zorder=3)
    ax.plot(times / 3600.0, radii * 100.0, color="white", lw=2.2, zorder=4,
            label="药材表面 $R(t)$")
    ax.set_xlim(0, stop / 3600.0)
    ax.set_ylim(0, 2.15)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("到药材中心的距离 / cm")
    ax.set_title("(a) 中截面水分浓度与收缩的表面", fontsize=11)
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label("水分浓度 / (kg/kg)")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)

    ax = axes[1]
    cmap = plt.get_cmap("plasma")
    for k, h in enumerate(PROFILE_TIMES):
        ax.plot(prof_r * 100.0, d["p%d" % h], color=cmap(k / 6.0), lw=1.6,
                label="%d h" % h)
    ax.plot(prof_r * 100.0, d["final_prof"], color="k", lw=2.0,
            label="结束 %.1f h" % (stop / 3600.0))
    ax.axhline(0.15, color="red", ls="--", lw=1.0)
    ax.text(0.05, 0.175, "达标判据 0.15 kg/kg", color="red", fontsize=8)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 2.75)
    ax.set_xlabel("到药材中心的距离 / cm")
    ax.set_ylabel("水分浓度 / (kg/kg)")
    ax.set_title("(b) 各时刻中截面径向剖面", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, ncol=2)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig_q4_moisture_field.png"
    fig.savefig(path, dpi=150)
    print("图已保存 ->", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
