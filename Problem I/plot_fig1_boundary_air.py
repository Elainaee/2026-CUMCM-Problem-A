# -*- coding: utf-8 -*-
"""
图：烘房温度与水分浓度在前半小时（0~1800 s）的变化 —— Plotly 版本
================================================================
数据源：../附件/附件1.xlsx（241 点，t = 0~14400 s，步长 60 s）
        本图只画 t <= 1800 s 的 31 个点

图内信息：
  (a) 烘房温度     (b) 烘房水分浓度
  每个子图 2 条：原始采样点（圆点）与 Savitzky-Golay 保形滤波曲线
  —— 平滑在全段 241 点上进行后再截取，避免端点被多项式外推带偏

输出：../graphics/fig_q1_boundary_air.png
用法：python "Problem I/plot_fig1_boundary_air.py"
"""
import os

import numpy as np
import openpyxl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

# ---------------- 全局绘图规范 ----------------
FONT = "Microsoft YaHei, SimHei, sans-serif"
INK = "#333333"
GRID = "#E0E0E0"
PALETTE = {"T": "#4472C4", "C": "#ED7D31"}

WINDOW_S = 1800          # 前半小时
SMOOTH_WIN, SMOOTH_ORDER = 31, 3   # Savitzky-Golay 保形滤波（与论文一致）

BASE = os.path.dirname(os.path.abspath(__file__))          # .../CUMCM/Problem I
PROJECT = os.path.dirname(BASE)                            # .../CUMCM
ATTACH = os.path.join(PROJECT, "附件", "附件1.xlsx")
GRAPHICS = os.path.join(PROJECT, "graphics")
SRC = ATTACH
OUT = os.path.join(GRAPHICS, "fig_q1_boundary_air.png")


def load_attachment1(path, t_max=WINDOW_S):
    """返回 (t, T原始, T平滑, C原始, C平滑)，均截取到 t <= t_max。"""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    rows = [r for r in list(ws.iter_rows(values_only=True))[1:] if r[0] is not None]
    wb.close()
    t_all = np.array([float(r[0]) for r in rows])
    T_all = np.array([float(r[1]) for r in rows])
    C_all = np.array([float(r[2]) for r in rows])
    # SG 平滑必须在全段上做，再截取
    T_sm = savgol_filter(T_all, SMOOTH_WIN, SMOOTH_ORDER)
    C_sm = savgol_filter(C_all, SMOOTH_WIN, SMOOTH_ORDER)
    m = t_all <= t_max
    return t_all[m], T_all[m], T_sm[m], C_all[m], C_sm[m]


def main():
    t, Tair, Tair_s, Cair, Cair_s = load_attachment1(SRC)

    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.12,
        subplot_titles=["(a) 烘房温度", "(b) 烘房水分浓度"],
    )

    # ---- (a) 温度 ----
    fig.add_trace(go.Scatter(
        x=t, y=Tair, mode="markers",
        marker=dict(color=PALETTE["T"], size=5, opacity=0.45),
        name="附件1 原始数据（未平滑）",
        hovertemplate="t = %{x:.0f} s<br>T<sub>air</sub> = %{y:.3f} °C<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=Tair_s, mode="lines",
        line=dict(color=PALETTE["T"], width=2.2),
        name=f"Savitzky–Golay 平滑（窗宽 {SMOOTH_WIN} 点）",
        hovertemplate="t = %{x:.0f} s<br>T<sub>air</sub> = %{y:.3f} °C<extra></extra>",
    ), row=1, col=1)

    # ---- (b) 水分浓度 ----
    fig.add_trace(go.Scatter(
        x=t, y=Cair, mode="markers",
        marker=dict(color=PALETTE["C"], size=5, opacity=0.45),
        name="附件1 原始数据（未平滑）", showlegend=False,
        hovertemplate="t = %{x:.0f} s<br>C<sub>air</sub> = %{y:.5f} kg/kg<extra></extra>",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=t, y=Cair_s, mode="lines",
        line=dict(color=PALETTE["C"], width=2.2),
        name=f"Savitzky–Golay 平滑（窗宽 {SMOOTH_WIN} 点）", showlegend=False,
        hovertemplate="t = %{x:.0f} s<br>C<sub>air</sub> = %{y:.5f} kg/kg<extra></extra>",
    ), row=1, col=2)

    # ---------------- 坐标轴 ----------------
    for col in (1, 2):
        fig.update_xaxes(
            title_text="时间 t / s", range=[0, WINDOW_S],
            tickmode="linear", tick0=0, dtick=300,
            showgrid=True, gridcolor=GRID, gridwidth=1,
            zeroline=False, linecolor="#999999", ticks="outside",
            row=1, col=col,
        )
    fig.update_yaxes(
        title_text="烘房温度 T<sub>air</sub> / °C", range=[27.5, 42.5],
        tickmode="linear", tick0=28, dtick=2,
        showgrid=True, gridcolor=GRID, gridwidth=1,
        zeroline=False, linecolor="#999999", row=1, col=1,
    )
    fig.update_yaxes(
        title_text="烘房水分浓度 C<sub>air</sub> / (kg/kg)", range=[0.0190, 0.0340],
        tickmode="linear", tick0=0.020, dtick=0.002, tickformat=".3f",
        showgrid=True, gridcolor=GRID, gridwidth=1,
        zeroline=False, linecolor="#999999", row=1, col=2,
    )

    # ---------------- 版面 ----------------
    fig.update_layout(
        width=1240, height=560,
        margin=dict(l=80, r=35, t=80, b=110),
        font=dict(family=FONT, size=13, color=INK),
        legend=dict(
            orientation="h", x=0.5, y=-0.17,
            xanchor="center", yanchor="top",
            font=dict(size=13, family=FONT, color=INK),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    for ann in fig.layout.annotations:
        if ann.text and ann.text.startswith("("):
            ann.font = dict(size=15, family=FONT, color=INK)

    os.makedirs(GRAPHICS, exist_ok=True)
    fig.write_image(OUT, scale=2)

    print("图已生成：")
    print("  ", OUT)
    print(f"\n数据点 {t.size} 个（t = {t[0]:.0f} ~ {t[-1]:.0f} s，步长 60 s）")
    print(f"  烘房温度     : 原始 {Tair[0]:.3f} -> {Tair[-1]:.3f} °C | "
          f"平滑 {Tair_s[0]:.3f} -> {Tair_s[-1]:.3f} °C | "
          f"平滑前后最大差 {np.abs(Tair_s - Tair).max():.4f} °C")
    print(f"  烘房水分浓度 : 原始 {Cair[0]:.5f} -> {Cair[-1]:.5f} kg/kg | "
          f"平滑 {Cair_s[0]:.5f} -> {Cair_s[-1]:.5f} kg/kg | "
          f"平滑前后最大差 {np.abs(Cair_s - Cair).max():.5f} kg/kg")


if __name__ == "__main__":
    main()
