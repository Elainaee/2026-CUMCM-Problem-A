"""问题四：收缩圆柱的热质耦合模型（独立于前三问脚本）。

python solve_problem4.py --project "E:/Working Space/CUMCM" --out work/q4 --nr 96 --nz 80
依赖 numpy scipy openpyxl。本脚本是问题四的唯一入口：求解 → 写 --out 目录（结果 JSON、表6 CSV、
诊断 CSV、summary JSON）→ 直接写出交付件 result4.xlsx（以附件3模板为底、四位小数、冻结首行首列）。

只重出 Excel、不重新求解（秒级）：
    python solve_problem4.py --from-json work/q4/result4_data.json [--xlsx 输出路径]

默认只在“主方案”跑法下写交付件（--out 为 work/q4 且未加 --check-only/--preheat/--no-smooth），
以免敏感性、网格检查跑覆盖 result4.xlsx；需要时用 --xlsx 显式指定输出路径，--no-xlsx 关闭。
坐标：xi=r/R(t), z 为中截面到端面的距离，长度保持 25 cm。
干基含水率跟随固体材料运动，采用均匀径向收缩，不加入体积浓度稀释项。
环境边界条件（附件1）先做 Savitzky-Golay 保形滤波（窗宽31点、三阶），与问题一、二一致。
"""
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.sparse import coo_matrix, bmat
from scipy.signal import savgol_filter
from openpyxl import load_workbook

# 环境边界条件的 Savitzky-Golay 保形滤波参数，与问题一、二同一套
SMOOTH_WIN, SMOOTH_ORDER = 31, 3


def read_attachment(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    a = np.array(list(wb.active.values)[1:], dtype=float)
    wb.close()
    assert np.isfinite(a).all() and np.all(np.diff(a[:, 0]) > 0)
    return a


class DryingModel:
    def __init__(self, project, nr=48, nz=40, shrink=True, preheat=False, smooth=True):
        self.air = read_attachment(project / '附件/附件1.xlsx')
        if smooth:
            # 环境边界条件先做 Savitzky-Golay 保形滤波（窗宽31点即31 min、三阶多项式），
            # 与问题一、二同一套处理；平滑后的序列才用于插值，平台统计也取自平滑序列。
            self.air[:, 1] = savgol_filter(self.air[:, 1], SMOOTH_WIN, SMOOTH_ORDER)
            self.air[:, 2] = savgol_filter(self.air[:, 2], SMOOTH_WIN, SMOOTH_ORDER)
        self.radius_data = read_attachment(project / '附件/附件2.xlsx')
        assert np.all(np.diff(self.radius_data[:, 1]) <= 0)
        self.rfit = PchipInterpolator(self.radius_data[:, 0], self.radius_data[:, 1] / 100)
        self.shrink, self.preheat, self.smooth = shrink, preheat, smooth
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
        gx, gw = np.polynomial.legendre.leggauss(5)
        self.gx, self.gw = (gx+1)/2, gw/2

    def radius(self, t):
        if not self.shrink:
            return .02
        # 附件覆盖到72h，主计算不允许依靠无依据的半径外推。
        return float(self.rfit(np.clip(t, 0, self.radius_data[-1, 0])))

    def ambient(self, t):
        if t > self.air[-1, 0]:
            return self.plateau
        return np.array([np.interp(t, self.air[:, 0], self.air[:, j]) for j in (1, 2)])

    def material(self, t, T, C):
        c = np.maximum(C, 1e-9)  # 仅保护牛顿迭代的试探值；最终场单独验证。
        rho = 760 + 90 * c
        cp = 1850 + 2150 * c / (1 + c)
        k = .12 + .20 * c / (1 + c)
        D = 4.2e-4 * np.exp(-.30 / c - 3850 / (T + 273.15))
        if self.preheat:
            s = np.clip((t - 4980) / 1800, 0, 1)
            w = s*s*(3 - 2*s)
            rho = (1-w)*820 + w*rho
            cp = (1-w)*2600 + w*cp
            k = (1-w)*.36 + w*k
            D = (1-w)*7e-9*np.exp(-.89/c) + w*D
        return rho*cp, k, D

    def moisture_faces(self, t, T, C):
        """Kirchhoff积分均值：对连续非线性D沿相邻节点状态积分。

        恒温时等价于 [Phi(Cj)-Phi(Ci)]/(Cj-Ci), Phi'=D。
        用5点Gauss积分避免相近浓度相减的消减误差。
        """
        out=[]
        for left,right in [(np.s_[:-1,:],np.s_[1:,:]),(np.s_[:,:-1],np.s_[:,1:])]:
            result=np.zeros_like(C[left])
            for x,w in zip(self.gx,self.gw):
                c=np.maximum((1-x)*C[left]+x*C[right],1e-9)
                temp=(1-x)*T[left]+x*T[right]
                d=4.2e-4*np.exp(-.30/c-3850/(temp+273.15))
                if self.preheat:
                    s=np.clip((t-4980)/1800,0,1);blend=s*s*(3-2*s)
                    d=(1-blend)*7e-9*np.exp(-.89/c)+blend*d
                result+=w*d
            out.append(result)
        return out

    def diffusion(self, f, a, h, ambient, R, faces=None):
        """节点控制体通量，省略各式相同的周向因子2pi。"""
        radial_area = self.rweight * R**2
        V = radial_area[:, None] * self.zweight[None, :]
        q = np.zeros_like(f)
        af = 2*a[:-1]*a[1:] / (a[:-1]+a[1:]) if faces is None else faces[0]
        flux = af * (f[1:] - f[:-1]) * ((self.x[:-1]+self.x[1:])/2/self.dx)[:, None] * self.zweight
        q[:-1] += flux
        q[1:] -= flux
        af = 2*a[:, :-1]*a[:, 1:] / (a[:, :-1]+a[:, 1:]) if faces is None else faces[1]
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
        return np.r_[(self.diffusion(T, k, 25, Ta, R)/cap).ravel(),
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
    m = DryingModel(args.project, args.nr, args.nz, not args.no_shrink, args.preheat, not args.no_smooth)
    y0 = np.r_[np.full(m.n,28.), np.full(m.n,2.55)]
    def event(t, y):
        return float(np.max(y[m.n:])) - .15
    event.terminal, event.direction = True, -1
    # 前4小时边界数据每60s变化，限制内部步长；后续仍连续继承全部状态。
    segments = []
    y = y0
    for start, end, step in [(0,14400,min(args.max_step,60)), (14400,259200,args.max_step)]:
        sol = solve_ivp(m.rhs,(start,end),y,method='BDF',rtol=args.rtol,atol=args.rtol*.01,
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
    final = solve_ivp(m.rhs,(crossing,stop),y,method='BDF',rtol=args.rtol,atol=args.rtol*.01,
                     jac_sparsity=m.sparsity,dense_output=True)
    assert final.success and final.y[m.n:,-1].max() < .15
    def state(t):
        if t > crossing:
            return final.sol(t)
        return segments[0].sol(t) if t <= 14400 else segments[1].sol(t)
    summary = {'nr':args.nr,'nz_half':args.nz,'rtol':args.rtol,'max_step_s':args.max_step,
               'smooth_env':bool(m.smooth),'smooth_win':SMOOTH_WIN,'smooth_order':SMOOTH_ORDER,
               'shrink':not args.no_shrink,'preheat_appendix2':args.preheat,
               'threshold_crossing_s':crossing,'threshold_crossing_h':crossing/3600,
               'strict_stop_s':stop,'strict_stop_h':stop/3600,
               'radius_stop_cm':m.radius(stop)*100,'max_C_stop':float(final.y[m.n:,-1].max()),
               'air_plateau_T_C':m.plateau.tolist(),'seconds_runtime':time.perf_counter()-started,
               'max_C_one_second_before_crossing':float(state(crossing-1)[m.n:].max()),
               'checks':m.checks(stop,final.y[:,-1]),
               'nfev':sum(s.nfev for s in segments)}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.check_only:
        print(json.dumps(summary,ensure_ascii=False),flush=True)
        return
    times = np.r_[np.arange(60,stop,60),stop]
    radii_cm = np.arange(21)/10
    records, diagnostics = [], []
    for t in times:
        avg, mid, cmax, cmin = m.profiles(state(t))
        Rcm = m.radius(t)*100
        # 口径：输出中截面 z=L/2 的径向剖面（此前用 avg 轴向平均）
        vals = [float(np.interp(r/Rcm,m.x,mid)) if r <= Rcm+1e-10 else None for r in radii_cm]
        records.append([float(t)]+vals+[float(mid[-1])])
        diagnostics.append([float(t),Rcm,cmax,cmin,float(mid[0]),float(avg[0])])
    table6=[]
    for t in np.r_[np.arange(21600,stop,21600),stop]:
        avg, mid, cmax, cmin = m.profiles(state(t))
        Rcm = m.radius(t)*100
        # 口径：表6 同用中截面；“药材表面”列 = mid[-1] = C(R(t), L/2, t)
        vals = [float(np.interp(r/Rcm,m.x,mid)) if r <= Rcm+1e-10 else None for r in (0,.5,1,1.5)]
        table6.append([float(t/3600)]+vals+[float(mid[-1]),Rcm,cmax])
    payload={'summary':summary,'headers':['时间/s；距离/cm']+radii_cm.tolist()+['药材表面'],
             'rows':records,'diagnostics':diagnostics,
             'table6_headers':['时间/h','0 cm','0.5 cm','1.0 cm','1.5 cm','药材表面','半径/cm','全场最大含水率'],
             'table6':table6}
    (args.out/'result4_data.json').write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    for filename,headers,rows in [('table6.csv',payload['table6_headers'],table6),
          ('diagnostics.csv',['time_s','radius_cm','Cmax','Cmin','center_midplane','center_axial_average'],diagnostics)]:
        with (args.out/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(headers);w.writerows(rows)
    target = xlsx_target(args)
    if target is None:
        why = '--no-xlsx' if args.no_xlsx else '非主方案跑法或带敏感性开关，需要时用 --xlsx 指定'
        print('未写 Excel（%s）' % why, flush=True)
    else:
        export_result4(payload, args.project, target)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def xlsx_target(args):
    """交付件 result4.xlsx 的写出路径；返回 None 表示本次不写。"""
    if args.no_xlsx:
        return None
    if args.xlsx is not None:
        return Path(args.xlsx)
    if args.check_only or args.preheat or args.no_smooth:
        return None
    main_out = Path(__file__).resolve().parent/'work'/'q4'
    if args.out.as_posix().rstrip('/')=='work/q4' or args.out.resolve()==main_out:
        return args.project/'result4.xlsx'
    return None


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
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print('已写出 %s  行数=%d 列数=%d' % (out, ws.max_row, ws.max_column), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--out',type=Path,default=Path('work/q4'))
    p.add_argument('--nr',type=int,default=72)
    p.add_argument('--nz',type=int,default=60,help='半长度方向的单元数')
    p.add_argument('--rtol',type=float,default=2e-7)
    p.add_argument('--max-step',type=float,default=300)
    p.add_argument('--no-shrink',action='store_true')
    p.add_argument('--preheat',action='store_true',help='敏感性：沿用第三问的预热物性平滑切换')
    p.add_argument('--no-smooth',action='store_true',help='敏感性：环境输入不做SG平滑，直接用附件1原值')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--xlsx',type=Path,default=None,help='显式指定 result4.xlsx 的输出路径')
    p.add_argument('--no-xlsx',action='store_true',help='本次不写 Excel')
    p.add_argument('--from-json',type=Path,default=None,
                   help='不求解，直接由已有 JSON 重出 result4.xlsx（秒级）')
    a=p.parse_args()
    if a.from_json is not None:
        export_result4(json.loads(a.from_json.read_text(encoding='utf-8')),
                       a.project, a.xlsx or (a.project/'result4.xlsx'))
    else:
        if a.nr<4 or a.nz<4 or a.rtol<=0 or a.max_step<=0:
            p.error('网格、容差和步长必须为正，网格至少4。')
        run(a)
