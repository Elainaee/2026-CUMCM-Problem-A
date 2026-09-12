# -*- coding: utf-8 -*-
"""
问题一 结果曲线图
================================================================
数据源：../result1.xlsx
        工作表「温度」与「水分浓度」，A 列为时间 t / s，第 1 行为到药材中心的距离 r / cm

图内信息：
  (a) 药材温度随时间变化      (b) 药材水分浓度随时间变化
  每个子图 4 条曲线，对应 r = 0.5 / 1.0 / 1.5 / 2.0 cm，各用不同颜色

输出：../graphics/fig_q1_curves.png
用法：python "Problem I/plot_curves.py"
"""
import os
import subprocess

import numpy as np
import openpyxl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------- 全局绘图规范 ----------------
FONT = "Microsoft YaHei, SimHei, sans-serif"
INK = "#333333"
GRID = "#E0E0E0"

# 要画的 4 个半径位置（cm）与各自的颜色（Office 主题色系，由内到外递进）
RADII_CM = [0.5, 1.0, 1.5, 2.0]
COLORS = {
    0.5: "#4472C4",   # 蓝
    1.0: "#ED7D31",   # 橙
    1.5: "#70AD47",   # 绿
    2.0: "#C00000",   # 红
}

BASE = os.path.dirname(os.path.abspath(__file__))          # .../CUMCM/Problem I
PROJECT = os.path.dirname(BASE)                            # .../CUMCM
GRAPHICS = os.path.join(PROJECT, "graphics")
SRC = os.path.join(PROJECT, "result1.xlsx")
OUT = os.path.join(GRAPHICS, "fig_q1_curves.png")


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


def load_sheet(path, sheet_name, radii_cm):
    """返回 (t, {r: 数值序列})。按数值匹配表头，容错字符串/浮点/百分比等差异。"""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))

    header = rows[0]
    col_of = {}
    for j, cell in enumerate(header):
        if cell is None or j == 0:
            continue
        try:
            val = float(cell)
        except (TypeError, ValueError):
            continue
        for r in radii_cm:
            if abs(val - r) < 1e-9:
                col_of[r] = j
    missing = [r for r in radii_cm if r not in col_of]
    if missing:
        raise KeyError(f"工作表「{sheet_name}」表头找不到这些半径列：{missing}")

    body = [row for row in rows[1:] if row[0] is not None]
    t = np.array([float(row[0]) for row in body])
    series = {
        r: np.array([float(row[col_of[r]]) for row in body])
        for r in radii_cm
    }
    return t, series


def main():
    t, temp = load_sheet(SRC, "温度", RADII_CM)
    _, moist = load_sheet(SRC, "水分浓度", RADII_CM)

    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.11,
        subplot_titles=["(a) 药材温度", "(b) 药材水分浓度"],
    )

    for r in RADII_CM:
        fig.add_trace(
            go.Scatter(
                x=t, y=temp[r], mode="lines",
                line=dict(color=COLORS[r], width=2),
                name=f"r = {r:.1f} cm",
                legendgroup=f"r{r}", showlegend=True,
                hovertemplate=(f"t = %{{x:.0f}} s<br>T = %{{y:.4f}} °C"
                               f"<extra>r = {r:.1f} cm</extra>"),
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t, y=moist[r], mode="lines",
                line=dict(color=COLORS[r], width=2),
                name=f"r = {r:.1f} cm",
                legendgroup=f"r{r}", showlegend=False,
                hovertemplate=(f"t = %{{x:.0f}} s<br>C = %{{y:.4f}} kg/kg"
                               f"<extra>r = {r:.1f} cm</extra>"),
            ), row=1, col=2,
        )

    # ---------------- 坐标轴 ----------------
    axis_common = dict(
        tickmode="linear", tick0=0, dtick=300,
        showgrid=True, gridcolor=GRID, gridwidth=1,
        zeroline=False, linecolor="#999999", ticks="outside",
    )
    fig.update_xaxes(title_text="时间 t / s", range=[0, 1800], **axis_common, row=1, col=1)
    fig.update_xaxes(title_text="时间 t / s", range=[0, 1800], **axis_common, row=1, col=2)

    fig.update_yaxes(
        title_text="温度 T / °C", range=[27.5, 37.5],
        tickmode="linear", tick0=28, dtick=1,
        showgrid=True, gridcolor=GRID, gridwidth=1,
        zeroline=False, linecolor="#999999", row=1, col=1,
    )
    fig.update_yaxes(
        title_text="水分浓度 C / (kg/kg)", range=[1.40, 2.62],
        tickmode="linear", tick0=1.4, dtick=0.2,
        showgrid=True, gridcolor=GRID, gridwidth=1,
        zeroline=False, linecolor="#999999", row=1, col=2,
    )

    # ---------------- 版面 ----------------
    fig.update_layout(
        width=1240, height=560,
        margin=dict(l=75, r=35, t=80, b=110),
        font=dict(family=FONT, size=13, color=INK),
        legend=dict(
            orientation="h", x=0.5, y=-0.17,
            xanchor="center", yanchor="top",
            font=dict(size=13, family=FONT, color=INK),
            bgcolor="rgba(0,0,0,0)",
            title=dict(text="到药材中心的距离 r：", font=dict(size=13, family=FONT, color=INK),
                       side="left"),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    for ann in fig.layout.annotations:
        if ann.text and ann.text.startswith("("):
            ann.font = dict(size=15, family=FONT, color=INK)

    os.makedirs(GRAPHICS, exist_ok=True)
    apply_windows_kaleido_cleanup_workaround()
    fig.write_image(OUT, scale=2)

    print("曲线图已生成：")
    print("  ", OUT)
    print("\n数验（各半径的起止值）：")
    for r in RADII_CM:
        print(f"  r = {r:.1f} cm : "
              f"T {temp[r][0]:.4f} -> {temp[r][-1]:.4f} °C | "
              f"C {moist[r][0]:.4f} -> {moist[r][-1]:.4f} kg/kg")


if __name__ == "__main__":
    main()
