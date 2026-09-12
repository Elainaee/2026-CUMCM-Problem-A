"""问题四：二维轴对称半长度收缩模型（数值规范共用 drying_common.py）。

敏感性/对照/网格检查跑法只算校验量、不落盘（validate_models.py 用 --validation_only 取内存结果）：
    python solve_problem4.py --no-shrink                  # 不收缩对照
    python solve_problem4.py --material-mode specified    # 全程附录4对照
坐标：xi=r/R(t), z 为中截面到端面的距离，长度保持 25 cm。
干基含水率跟随固体材料运动，采用均匀径向收缩，不加入体积浓度稀释项。
环境边界条件（附件1）先做 Savitzky-Golay 保形滤波（窗宽31点、三阶），与问题一、二一致。
"""
from pathlib import Path
import argparse
import sys
import json
import os
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.sparse import coo_matrix, bmat
from scipy.signal import savgol_filter
from openpyxl import load_workbook
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drying_common import (face_mean, component_atol, mass_balance, RTOL, MATERIAL_MODE,
                          ENV_FINE_INTERVAL_S, ENV_FINE_UNTIL_S,
                          ensure_deliverable_writable, save_deliverable)

# 环境边界条件的 Savitzky-Golay 保形滤波参数，与问题一、二同一套
SMOOTH_WIN, SMOOTH_ORDER = 31, 3


def read_attachment(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    a = np.array(list(wb.active.values)[1:], dtype=float)
    wb.close()
    assert np.isfinite(a).all() and np.all(np.diff(a[:, 0]) > 0)
    return a


class DryingModel:
    def __init__(self, project, nr=320, nz=40, shrink=True, preheat=False, smooth=True, material_mode=MATERIAL_MODE):
        self.air = read_attachment(project / '附件/附件1.xlsx')
        if smooth:
            # 环境边界条件先做 Savitzky-Golay 保形滤波（窗宽31点即31 min、三阶多项式）
            self.air[:, 1] = savgol_filter(self.air[:, 1], SMOOTH_WIN, SMOOTH_ORDER)
            self.air[:, 2] = savgol_filter(self.air[:, 2], SMOOTH_WIN, SMOOTH_ORDER)
        self.radius_data = read_attachment(project / '附件/附件2.xlsx')
        assert np.all(np.diff(self.radius_data[:, 1]) <= 0)
        self.rfit = PchipInterpolator(self.radius_data[:, 0], self.radius_data[:, 1] / 100)
        self.shrink, self.preheat, self.smooth = shrink, preheat, smooth
        self.material_mode = "smooth" if preheat else material_mode
        tail = self.air[self.air[:, 0] >= 10800, 1:]
        self.plateau = np.mean(tail, axis=0)
        self.nr, self.nz = nr, nz
        self.x = np.linspace(0, 1, nr + 1)
        self.z = np.linspace(0, .125, nz + 1)
        self.dx, self.dz = 1 / nr, .125 / nz
        faces = np.r_[0, (self.x[:-1] + self.x[1:]) / 2, 1]
        self.rweight = np.diff(faces**2) / 2
        self.zweight = np.full(nz + 1, self.dz)
        self.zweight[[0, -1]] /= 2
        self.shape = (nr + 1, nz + 1)
        self.n = (nr + 1) * (nz + 1)
        ids = np.arange(self.n).reshape(self.shape)
        i = np.r_[ids[:-1].ravel(), ids[:, :-1].ravel()]
        j = np.r_[ids[1:].ravel(), ids[:, 1:].ravel()]
        rows, cols = np.r_[ids.ravel(), i, j], np.r_[ids.ravel(), j, i]
        S = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.n, self.n)).tocsr()
        self.sparsity = bmat([[S, S], [S, S]], format='csr')

    def radius(self, t):
        if not self.shrink:
            return .02
        # 附件覆盖到72h，主计算不允许依靠无依据的半径外推。
        return float(self.rfit(np.clip(t, 0, self.radius_data[-1, 0])))

    def ambient(self, t):
        if t > self.air[-1, 0]:
            return self.plateau
        return np.array([np.interp(t, self.air[:, 0], self.air[:, j]) for j in (1, 2)])

    def phase_weight(self, t):
        if self.material_mode == "specified":
            return 1.
        if self.material_mode == "staged":
            return float(t >= 6780.)
        x = np.clip((t-4980.)/1800.,0.,1.)
        return x*x*(3-2*x)

    def material(self, t, T, C):
        c = np.maximum(C, 1e-9)  # 仅保护牛顿迭代的试探值；最终场单独验证。
        rho = 760 + 90 * c
        cp = 1850 + 2150 * c / (1 + c)
        k = .12 + .20 * c / (1 + c)
        D = 4.2e-4 * np.exp(-.30 / c - 3850 / (T + 273.15))
        w = self.phase_weight(t)
        rho = (1-w)*820 + w*rho
        cp = (1-w)*2600 + w*cp
        k = (1-w)*.36 + w*k
        D = (1-w)*7e-9*np.exp(-.89/c) + w*D
        return rho*cp, k, D

    def transport_faces(self, t, T, C, kind="mass"):
        index = 2 if kind == "mass" else 1
        function = lambda c,temp: self.material(t,temp,c)[index]
        return tuple(face_mean(function,(C[a],T[a]),(C[b],T[b]))
            for a,b in ((np.s_[:-1,:],np.s_[1:,:]),(np.s_[:,:-1],np.s_[:,1:])))

    def moisture_faces(self,t,T,C):
        return self.transport_faces(t,T,C,"mass")

    def boundary_inflow(self,t,y):
        C=y[self.n:].reshape(self.shape)
        R=self.radius(t)
        ca=self.ambient(t)[1]
        net=8e-7*(np.sum((ca-C[-1])*R*self.zweight)+
                  np.sum((ca-C[:,-1])*self.rweight*R**2))
        volume=R**2*self.rweight.sum()*self.zweight.sum()
        return float(net/volume)

    def diffusion(self, f, a, h, ambient, R, faces=None):
        """节点控制体通量，省略各式相同的周向因子2pi。"""
        if faces is None and not np.all(a == a.flat[0]):
            raise ValueError("非均匀物性必须显式传入积分平均面系数")
        radial_area = self.rweight * R**2
        V = radial_area[:, None] * self.zweight[None, :]
        q = np.zeros_like(f)
        af = faces[0] if faces is not None else np.broadcast_to(a[:-1], f[:-1].shape)
        flux = af * (f[1:] - f[:-1]) * ((self.x[:-1]+self.x[1:])/2/self.dx)[:, None] * self.zweight
        q[:-1] += flux
        q[1:] -= flux
        af = faces[1] if faces is not None else np.broadcast_to(a[:, :-1], f[:, :-1].shape)
        flux = af * (f[:, 1:] - f[:, :-1]) / self.dz * radial_area[:, None]
        q[:, :-1] += flux
        q[:, 1:] -= flux
        q[-1] += h*(ambient-f[-1])*R*self.zweight
        q[:, -1] += h*(ambient-f[:, -1])*radial_area
        return q / V

    def rhs(self, t, y):
        T, C = y[:self.n].reshape(self.shape), y[self.n:].reshape(self.shape)
        cap, k, D = self.material(t, T, C)
        Ta, Ca = self.ambient(t)
        R = self.radius(t)
        return np.r_[(self.diffusion(T, k, 25, Ta, R, self.transport_faces(t,T,C,"heat"))/cap).ravel(),
                     self.diffusion(C, D, 8e-7, Ca, R, self.moisture_faces(t,T,C)).ravel()]

    def profiles(self, y):
        C = y[self.n:].reshape(self.shape)
        return C @ self.zweight / .125, C[:, 0], float(C.max()), float(C.min())

    def checks(self, t, y):
        T, C = y[:self.n].reshape(self.shape), y[self.n:].reshape(self.shape)
        _, _, D = self.material(t, T, C)
        R = self.radius(t)
        Ca = self.ambient(t)[1]
        V = self.rweight[:, None]*R**2*self.zweight
        dC = self.diffusion(C, D, 8e-7, Ca, R, self.moisture_faces(t,T,C))
        outward = 8e-7*(np.sum((C[-1]-Ca)*R*self.zweight) + np.sum((C[:, -1]-Ca)*self.rweight*R**2))
        balance = float(abs(np.sum(V*dC)+outward)/max(abs(outward), 1e-30))
        eqT, eqC = np.full(self.shape, 50.), np.full(self.shape, .05)
        _, k, D = self.material(t, eqT, eqC)
        eq = max(np.abs(self.diffusion(eqT,k,25,50.,R)).max(), np.abs(self.diffusion(eqC,D,8e-7,.05,R)).max())
        return {'relative_instantaneous_flux_residual':balance, 'uniform_equilibrium_rhs':float(eq)}


def run(args):
    started = time.perf_counter()
    target = xlsx_target(args) 
    if target is not None:
        ensure_deliverable_writable(target)
    m = DryingModel(args.project, args.nr, args.nz, not args.no_shrink, args.preheat, not args.no_smooth, args.material_mode)
    y0 = np.r_[np.full(m.n,28.), np.full(m.n,2.55)]
    def event(t, y):
        return float(np.max(y[m.n:])) - .15
    event.terminal, event.direction = True, -1
    # 前4h环境数据每60s变化，按共用规范限制内部步长；后续仍连续继承全部状态。
    segments = []
    y = y0
    bounds=sorted(set([0.,ENV_FINE_UNTIL_S,259200.]+([6780.] if m.material_mode=="staged" else [])))
    for start,end in zip(bounds[:-1],bounds[1:]):
        step=min(args.max_step,ENV_FINE_INTERVAL_S) if start<ENV_FINE_UNTIL_S else args.max_step
        def rhs(t,y):
            tt=np.nextafter(t,-np.inf) if end==6780. and t>=end else t
            return m.rhs(tt,y)
        sol = solve_ivp(rhs,(start,end),y,method='BDF',rtol=args.rtol,atol=component_atol(m.n)*(args.rtol/RTOL),
                        jac_sparsity=m.sparsity,max_step=step,events=event,dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        segments.append(sol)
        y = sol.y[:,-1]
        print(f'grid={args.nr}x{args.nz}, t={sol.t[-1]/3600:.6f} h, Cmax={y[m.n:].max():.9f}',flush=True)
        if sol.t_events[0].size:
            break
    if not segments[-1].t_events[0].size:
        raise RuntimeError('72小时内未达标，附件2不足以确定更晚半径，未生成伪造完成结果。')
    crossing = float(segments[-1].t_events[0][0])
    # 等于阈值是临界时刻；另算下一整秒以满足严格小于。
    stop = float(np.floor(crossing)+1)
    if stop > 259200:
        raise RuntimeError("严格达标时刻超出半径数据覆盖时长")
    final = solve_ivp(m.rhs,(crossing,stop),y,method='BDF',rtol=args.rtol,atol=component_atol(m.n)*(args.rtol/RTOL),
                     jac_sparsity=m.sparsity,dense_output=True)
    assert final.success and final.y[m.n:,-1].max() < .15
    def state(t):
        if t > crossing:
            return final.sol(t)
        for segment in segments:
            if t <= segment.t[-1]+1e-9:
                return segment.sol(t)
        raise ValueError("时间超出积分区间")
    summary = {'nr':args.nr,'nz_half':args.nz,'rtol':args.rtol,'max_step_s':args.max_step,
               'smooth_env':bool(m.smooth),'smooth_win':SMOOTH_WIN,'smooth_order':SMOOTH_ORDER,
               'shrink':not args.no_shrink,'preheat_appendix2':m.material_mode!='specified',
               'material_mode':m.material_mode,'profile_mode':'midplane','schema_version':2,
               'conservation':mass_balance(segments+[final],m.n,
                   m.rweight[:,None]*m.zweight[None,:],m.boundary_inflow),
               'threshold_crossing_s':crossing,'threshold_crossing_h':crossing/3600,
               'strict_stop_s':stop,'strict_stop_h':stop/3600,
               'radius_stop_cm':m.radius(stop)*100,'max_C_stop':float(final.y[m.n:,-1].max()),
               'air_plateau_T_C':m.plateau.tolist(),'seconds_runtime':time.perf_counter()-started,
               'max_C_one_second_before_crossing':float(state(crossing-1)[m.n:].max()),
               'checks':m.checks(stop,final.y[:,-1]),
               'nfev':sum(s.nfev for s in segments)}
    if getattr(args, 'validation_only', False):
        times=np.r_[np.arange(0.,stop,60.),stop]
        temperatures,moistures=[],[]
        # 与第三问共用的探针：归一化半径 xi=r/R(t)=0,0.05,...,1，共21列。
        # radius_cm 记录参考构型半径 xi*2cm；收缩时实际半径再乘 R(t)/2cm。
        xi=np.linspace(0.,1.,21)
        for t in times:
            vector=state(t)
            T=vector[:m.n].reshape(m.shape)[:,0]
            C=vector[m.n:].reshape(m.shape)[:,0]
            temperatures.append(np.interp(xi,m.x,T))
            moistures.append(np.interp(xi,m.x,C))
        return dict(time_s=times,xi=xi,radius_cm=xi*2.,
            temperature_c=np.asarray(temperatures),
            moisture_kg_per_kg=np.asarray(moistures),crossing_time_s=crossing,
            conservation=summary['conservation'],summary=summary)
    times = np.r_[np.arange(60,stop,60),stop]
    radii_cm = np.arange(21)/10
    records = []
    for t in times:
        avg, mid, cmax, cmin = m.profiles(state(t))
        Rcm = m.radius(t)*100
        # 口径：输出中截面 z=L/2 的径向剖面（此前用 avg 轴向平均）
        vals = [float(np.interp(r/Rcm,m.x,mid)) if r <= Rcm+1e-10 else None for r in radii_cm]
        records.append([float(t)]+vals+[float(mid[-1])])
    table6=[]
    for t in np.r_[np.arange(21600,stop,21600),stop]:
        avg, mid, cmax, cmin = m.profiles(state(t))
        Rcm = m.radius(t)*100
        # 口径：表6 同用中截面；“药材表面”列 = mid[-1] = C(R(t), L/2, t)
        vals = [float(np.interp(r/Rcm,m.x,mid)) if r <= Rcm+1e-10 else None for r in (0,.5,1,1.5)]
        table6.append([float(t/3600)]+vals+[float(mid[-1]),Rcm,cmax])
    payload={'headers':['时间/s；距离/cm']+radii_cm.tolist()+['药材表面'],
             'rows':records,
             'table6_headers':['时间/h','0 cm','0.5 cm','1.0 cm','1.5 cm','药材表面','半径/cm','全场最大含水率'],
             'table6':table6}
    target = xlsx_target(args)
    if target is None:
        print('未写交付件：本次为敏感性/对照跑法，只算校验量、不落盘',flush=True)
    else:
        export_result4(payload, args.project, target)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def xlsx_target(args):
    """交付件 result4.xlsx 的写出路径；敏感性/对照跑法返回 None。"""
    variants = []
    if args.no_shrink:
        variants.append('no_shrink')
    if getattr(args, 'preheat', False):
        variants.append('smooth_material')
    if args.material_mode != MATERIAL_MODE:
        variants.append(args.material_mode)
    if args.no_smooth:
        variants.append('no_smooth')
    if variants:      # 对照/敏感性跑法只算校验量
        return None
    return args.project/'result4.xlsx'


def export_result4(payload, project, out):
    """按“附件3/result4.xlsx”模板写出交付件：四位小数、冻结首行首列、超出半径留空。"""
    headers, rows = payload['headers'], payload['rows']
    if len(headers) != len(rows[0]):
        raise SystemExit('表头与数据列数不符：%d vs %d' % (len(headers), len(rows[0])))
    wb = load_workbook(project/'附件'/'附件3'/'result4.xlsx')
    ws = wb['Sheet1']
    for row in ws.iter_rows():
        for c in row:
            c.value = None
    for j, h in enumerate(headers, start=1):
        ws.cell(1, j, h)
    for i, rec in enumerate(rows, start=2):
        ws.cell(i, 1, int(round(rec[0])))
        for j, v in enumerate(rec[1:], start=2):
            if v is None:
                continue                      # 超出当前半径：留空
            ws.cell(i, j, round(float(v), 4)).number_format = '0.0000'
    ws.freeze_panes = 'B2'
    ws.column_dimensions['A'].width = 19.625
    for col in 'BCDEFGHIJKLMNOPQRSTUVW':
        ws.column_dimensions[col].width = 9.625
    shape = (ws.max_row, ws.max_column)
    saved = save_deliverable(wb, out)
    print('  行数=%d 列数=%d' % shape, flush=True)
    return saved


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--nr',type=int,default=320,help='径向单元数；取20的倍数时交付表1mm列正好是节点')
    p.add_argument('--nz',type=int,default=40,help='半长度方向的单元数')
    p.add_argument('--rtol',type=float,default=RTOL)
    p.add_argument('--max-step',type=float,default=300)
    p.add_argument('--no-shrink',action='store_true',help='对照：不收缩')
    p.add_argument('--material-mode',choices=['specified','staged','smooth'],default=MATERIAL_MODE,help='specified=全程附录4；staged=6780s附录2到4；smooth=平滑对照')
    p.add_argument('--preheat',action='store_true',help='敏感性：使用附录2预热物性的平滑过渡方案')
    p.add_argument('--no-smooth',action='store_true',help='敏感性：环境输入不做SG平滑，直接用附件1原值')
    a=p.parse_args()
    if a.nr<4 or a.nz<4 or a.rtol<=0 or a.max_step<=0:
        p.error('网格、容差和步长必须为正，网格至少4。')
    run(a)
