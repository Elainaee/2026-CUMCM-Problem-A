"""四问共用的面积分、误差标准、干物质质量归一化守恒检查与交付件写出。"""
import os
from pathlib import Path

import numpy as np

RTOL = 2e-8
ATOL_T = 1e-7       # deg C，时间积分内部容差，不等于总误差
ATOL_C = 1e-10      # kg/kg
NONLINEAR_RTOL = 1e-9
NONLINEAR_MAXITER = 50
PROFILE_C_TOL = 5e-5  # 交付表四位小数对应的目标误差；需网格/时间检验
PROFILE_T_TOL = 5e-5
DRY_TIME_TOL_S = 1.0
MASS_BALANCE_TOL = 1e-5
INSTANT_BALANCE_TOL = 1e-10
MATERIAL_MODE = "staged"  # 用户指定方案；specified=全程使用题面该问附录。
# 环境数据（附件1）在前4h按60s间隔给出；该窗口内所有入口共用同一最大步长上限，
# 之后放开到各入口自己的max_step，避免第二、三、四问时间步策略不一致。
ENV_FINE_INTERVAL_S = 60.0
ENV_FINE_UNTIL_S = 14400.0
GX, GW = np.polynomial.legendre.leggauss(5)
GX, GW = (GX + 1) / 2, GW / 2


def face_mean(function, left_states, right_states):
    """连续物性沿相邻节点状态的5点Gauss积分均值；支持D(C,T)和k(C)。"""
    result = np.zeros_like(np.asarray(left_states[0]), dtype=float)
    for x, w in zip(GX, GW):
        args = [(1-x)*a+x*b for a, b in zip(left_states, right_states)]
        result += w * function(*args)
    return result


def component_atol(node_count):
    return np.r_[np.full(node_count, ATOL_T), np.full(node_count, ATOL_C)]


def mass_balance(segments, node_count, dry_weights, inflow_rate):
    """每kg干物质的水质量收支。dry_weights为固定材料单元的干质量份额。

    均匀径向收缩且干物质不流失：rho_d(t)=rho_d(0)/J(t)。
    inflow_rate必须独立从边界计算，单位kg水/(kg干物质*s)。
    对每个已接受的BDF时间步用5点Gauss积分，避免跨事件重复累计。
    不用热物性经验rho(C)/(1+C)重新定义各单元干质量。
    """
    weights = np.asarray(dry_weights).ravel()
    weights = weights / weights.sum()
    initial = float(weights @ segments[0].y[node_count:, 0])
    final = float(weights @ segments[-1].y[node_count:, -1])
    accumulated = 0.0
    for segment in segments:
        for left, right in zip(segment.t[:-1], segment.t[1:]):
            for x, w in zip(GX, GW):
                t = left + x*(right-left)
                accumulated += (right-left)*w*inflow_rate(t, segment.sol(t))
    error = abs(final-initial-accumulated)
    normalized = error / max(abs(final-initial), 1e-12)
    return dict(initial_water_per_kg_dry=initial, final_water_per_kg_dry=final,
                integrated_inflow_per_kg_dry=float(accumulated),
                absolute_balance_error=float(error), relative_balance_error=float(normalized),
                balance_pass=bool(normalized <= MASS_BALANCE_TOL))


def ensure_deliverable_writable(target):
    """长算前预检交付件：被 Excel 等程序锁定时立即报错，避免算完写不进去。"""
    target = Path(target)
    if not target.exists():
        return target
    try:
        with target.open("r+b"):
            pass
    except PermissionError:
        raise SystemExit("交付件 %s 当前不可写：可能正被 Excel 等程序打开，\n"
                         "也可能当前账号对该文件没有写权限；请处理后重跑。" % target)
    return target


def save_deliverable(workbook, target):
    """把交付件原子写到 target：只产生 target 这一个结果文件。

    先写同目录临时文件再替换目标；替换失败（文件被占用）时删除临时文件并报错，
    不生成 _pending 或对照副本。
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    workbook.save(temporary)
    workbook.close()
    try:
        os.replace(temporary, target)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise SystemExit("无法写出 %s：文件被占用，或当前账号没有写权限，请处理后重跑。"
                         % target)
    print("已写出 %s" % target, flush=True)
    return target
