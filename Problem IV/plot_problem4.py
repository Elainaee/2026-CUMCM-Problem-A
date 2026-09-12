# -*- coding: utf-8 -*-
"""问题四绘图：中截面水分浓度时空分布 + 径向剖面（供论文图使用）。

    python plot_problem4.py                 # 生产配置 96x80，约 2 min

左图：物理坐标 (t, r) 下的中截面水分浓度热力图，白色区域为收缩后已不存在的部分，
      白线为随时间收缩的药材表面 R(t)；
右图：若干典型时刻的中截面径向剖面，每条曲线终止于该时刻的药材表面。
"""
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from solve_problem4 import DryingModel                      # noqa: E402

NR, NZ = 96, 80
RTOL, MAX_STEP = 2e-7, 300.0
PROFILE_TIMES = [6, 12, 24, 36, 48]
OUT = HERE.parent / "graphics" / "fig_q4_moisture_field.png"


def apply_windows_kaleido_cleanup_workaround():
    """兼容 Choreographer 在 Windows 上误报 taskkill 超时的问题。"""
    if os.name != "nt":
        return

    try:
        import choreographer.browser_async as browser_async
    except ImportError:
        return

    original_kill = browser_async.kill
    if getattr(original_kill, "_cumcm_timeout_workaround", False):
        return

    def kill_without_taskkill_timeout(process):
        try:
            original_kill(process)
        except subprocess.TimeoutExpired:
            # taskkill 偶尔在目标 Chrome 已退出后仍不返回。忽略其自身的
            # 超时，随后仍由 Choreographer 检查浏览器是否真正关闭。
            pass

    kill_without_taskkill_timeout._cumcm_timeout_workaround = True
    browser_async.kill = kill_without_taskkill_timeout


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
    return {
        "times": times,
        "mid": mid,
        "radii": radii,
        "r_grid": r_grid,
        "prof_r": prof_r,
        "stop": stop,
        "profiles": profiles,
        "final_prof": np.interp(
            prof_r, r_grid, mid[-1], left=mid[-1][0], right=np.nan
        ),
    }


def main():
    t0 = time.time()
    data = compute()
    stop = data["stop"]
    print("计算完成 %.0f s, t_dry = %.4f h" % (time.time() - t0, stop / 3600))

    times = data["times"]
    mid = data["mid"]
    radii = data["radii"]
    r_grid = data["r_grid"]
    prof_r = data["prof_r"]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.52, 0.48],
        horizontal_spacing=0.15,
        subplot_titles=[
            "(a) 中截面水分浓度与收缩的表面",
            "(b) 各时刻中截面径向剖面",
        ],
    )
    fig.add_trace(
        go.Heatmap(
            x=times / 3600.0,
            y=r_grid * 100.0,
            z=mid.T,
            colorscale="Viridis",
            zmin=0.0,
            zmax=2.6,
            colorbar=dict(
                title=dict(text="水分浓度<br>/(kg/kg)"),
                x=0.47,
                len=0.92,
                thickness=14,
            ),
            hovertemplate=(
                "时间 = %{x:.2f} h<br>距离 = %{y:.3f} cm"
                "<br>水分浓度 = %{z:.4f} kg/kg<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=times / 3600.0,
            y=radii * 100.0,
            mode="lines",
            line=dict(color="white", width=2.2),
            name="药材表面 R(t)",
            showlegend=False,
            hovertemplate="时间 = %{x:.2f} h<br>表面半径 = %{y:.3f} cm<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=0.02 * stop / 3600.0,
        y=0.08,
        text="白线：药材表面 R(t)",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#BBBBBB",
        font=dict(size=11, color="#333333"),
        xanchor="left",
        row=1,
        col=1,
    )

    for k, h in enumerate(PROFILE_TIMES):
        color = sample_colorscale("Plasma", [k / 6.0])[0]
        fig.add_trace(
            go.Scatter(
                x=prof_r * 100.0,
                y=data["profiles"][h],
                mode="lines",
                line=dict(color=color, width=1.8),
                name=f"{h} h",
                hovertemplate=(
                    f"{h} h<br>距离 = %{{x:.3f}} cm"
                    "<br>水分浓度 = %{y:.4f} kg/kg<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=prof_r * 100.0,
            y=data["final_prof"],
            mode="lines",
            line=dict(color="black", width=2.2),
            name="结束 %.1f h" % (stop / 3600.0),
            hovertemplate=(
                "结束时<br>距离 = %{x:.3f} cm"
                "<br>水分浓度 = %{y:.4f} kg/kg<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=0.15,
        line=dict(color="red", width=1.2, dash="dash"),
        annotation_text="达标判据 0.15 kg/kg",
        annotation_position="top left",
        annotation_font=dict(color="red", size=10),
        row=1,
        col=2,
    )

    fig.update_xaxes(
        title_text="时间 / h", range=[0, stop / 3600.0], row=1, col=1
    )
    fig.update_yaxes(
        title_text="到药材中心的距离 / cm", range=[0, 2.15], row=1, col=1
    )
    fig.update_xaxes(
        title_text="到药材中心的距离 / cm", range=[0, 2.0], row=1, col=2
    )
    fig.update_yaxes(
        title_text="水分浓度 / (kg/kg)", range=[0, 2.75], row=1, col=2
    )
    fig.update_layout(
        width=1300,
        height=460,
        margin=dict(l=70, r=35, t=70, b=60),
        font=dict(family="Microsoft YaHei, SimHei, sans-serif", size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            x=0.99,
            y=0.98,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#BBBBBB",
            borderwidth=1,
            font=dict(size=10),
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E0E0E0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E0E0E0", zeroline=False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    apply_windows_kaleido_cleanup_workaround()
    fig.write_image(OUT, scale=2)
    print("图已保存 ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
