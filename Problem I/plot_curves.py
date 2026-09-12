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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl

# ---------------- 全局绘图规范 ----------------
FONT = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
INK = "#333333"
GRID = "#E0E0E0"

plt.rcParams["font.sans-serif"] = FONT
plt.rcParams["axes.unicode_minus"] = False

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

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    ax_temp, ax_moist = axes

    for r in RADII_CM:
        ax_temp.plot(
            t, temp[r], color=COLORS[r], linewidth=2,
            label=f"r = {r:.1f} cm",
        )
        ax_moist.plot(
            t, moist[r], color=COLORS[r], linewidth=2,
            label=f"r = {r:.1f} cm",
        )

    # ---------------- 坐标轴 ----------------
    for ax in axes:
        ax.set_xlim(0, 1800)
        ax.set_xticks(np.arange(0, 1801, 300))
        ax.set_xlabel("时间 t / s")
        ax.grid(True, color=GRID, linewidth=1)
        ax.tick_params(direction="out", colors=INK)
        for spine in ax.spines.values():
            spine.set_color("#999999")

    ax_temp.set_title("(a) 药材温度", fontsize=15, color=INK)
    ax_temp.set_ylabel("温度 T / °C")
    ax_temp.set_ylim(27.5, 37.5)
    ax_temp.set_yticks(np.arange(28, 38, 1))

    ax_moist.set_title("(b) 药材水分浓度", fontsize=15, color=INK)
    ax_moist.set_ylabel("水分浓度 C / (kg/kg)")
    ax_moist.set_ylim(1.40, 2.62)
    ax_moist.set_yticks(np.arange(1.4, 2.61, 0.2))

    # ---------------- 版面 ----------------
    handles, labels = ax_temp.get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=4, loc="lower center",
        bbox_to_anchor=(0.58, 0.015), frameon=False, fontsize=13,
    )
    fig.text(0.125, 0.032, "到药材中心的距离 r：", fontsize=13, color=INK)
    fig.subplots_adjust(left=0.075, right=0.97, top=0.88, bottom=0.22, wspace=0.22)

    os.makedirs(GRAPHICS, exist_ok=True)
    # 直接由 Matplotlib 写入 PNG，避免 Kaleido 在 Windows 上清理
    # Chrome 子进程时偶发的 “Couldn't close or kill browser subprocess”。
    tmp_out = f"{OUT}.tmp"
    try:
        fig.savefig(tmp_out, format="png", dpi=200, facecolor="white")
        os.replace(tmp_out, OUT)
    finally:
        plt.close(fig)
        if os.path.exists(tmp_out):
            os.remove(tmp_out)

    print("曲线图已生成：")
    print("  ", OUT)
    print("\n数验（各半径的起止值）：")
    for r in RADII_CM:
        print(f"  r = {r:.1f} cm : "
              f"T {temp[r][0]:.4f} -> {temp[r][-1]:.4f} °C | "
              f"C {moist[r][0]:.4f} -> {moist[r][-1]:.4f} kg/kg")


if __name__ == "__main__":
    main()
