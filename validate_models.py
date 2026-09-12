"""统一验证入口；默认快速回归，--full 追加分层网格与时间精度检查，不写正式Excel。

误差标准是分层的：初始边界层(0~600s)只记录不设四位小数门槛，交付时段(t>=6h)
才用 5e-5 含水率/温度门槛，并对三网格序列做 Richardson 误差估计。烘干时间差
只在“同一网格缩小时间容差”时要求 <=1s；空间加密时它是被检验的输出之一。
含早期边界层的全程 C/T 对照只记录、不参与判据：5e-5 门槛只对交付时段(t>=6h)生效。
"""
from pathlib import Path
import argparse
import importlib.util
import sys
import time
from types import SimpleNamespace
import numpy as np
from drying_common import (face_mean, RTOL, PROFILE_C_TOL, PROFILE_T_TOL,
                           DRY_TIME_TOL_S, MASS_BALANCE_TOL)

ROOT=Path(__file__).resolve().parent
def load(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def quick():
    q1=load('validation_q1','Problem I/solve_problem1.py')
    q2=load('validation_q2','Problem II/solve_result2.py')
    q3=q2.model_core()
    q4=load('validation_q4','Problem IV/solve_problem4.py')
    got=face_mean(lambda c:c*c,(np.array([1.]),),(np.array([3.]),))
    np.testing.assert_allclose(got,[13/3],rtol=1e-14)
    print('PASS: 面积分解析多项式验证',flush=True)
    # Exact reduction: same constitutive equations and fixed radius => Q3/Q4 RHS agree.
    nr,nz=6,7
    m=q4.DryingModel(ROOT,nr,nz,shrink=False)
    def common_material(t,T,C):
        rho,cp,k=q3.material_properties(C,t)
        return rho*cp,k,q3.moisture_diffusivity(C,T,t)
    m.material=common_material
    y=q3.make_state(nr,nz)
    y[:m.n]=np.linspace(29.,49.,m.n)
    y[m.n:]=np.linspace(.2,2.4,m.n)
    for t in (0.,6779.999,6780.,15000.):
        rhs=q3.right_hand_side(y,t,nr,nz,.02/nr,.125/nz)
        np.testing.assert_allclose(rhs,m.rhs(t,y),rtol=1e-11,atol=1e-12)
        weights=q3.make_geometry(nr,nz)[2]
        lhs=np.sum(rhs[m.n:].reshape(m.shape)*weights)/weights.sum()
        np.testing.assert_allclose(lhs,q3.boundary_inflow(t,y,nr,nz),rtol=1e-11,atol=1e-15)
    print('PASS: 同物性不收缩时第三/四问逐节点RHS一致；半域边界收支一致',flush=True)
    # Actual shrink: dry-mass weighting differs from moving-volume concentration integral.
    shrink=q4.DryingModel(ROOT,nr,nz)
    for t in (0.,7200.,100000.):
        rhs=shrink.rhs(t,y)[m.n:].reshape(m.shape)
        weights=shrink.rweight[:,None]*shrink.zweight[None,:]
        np.testing.assert_allclose(np.sum(weights*rhs)/weights.sum(),shrink.boundary_inflow(t,y),rtol=1e-11,atol=1e-15)
    for nz in (5,6):
        z=np.linspace(0,.125,nz+1);r=np.linspace(0,.02,nr+1)
        c=np.broadcast_to(z*z,(nr+1,nz+1)).copy()
        field=np.r_[np.full(c.size,28.),c.ravel()]
        _,profile=q3.profiles_from_state(field,r,r,z,nr,nz)
        np.testing.assert_array_equal(profile,np.zeros(nr+1))
    print('PASS: 收缩材料干质量权重恒定；奇偶网格中截面精确提取',flush=True)
    with q3._overrides(air_temperature=lambda t:50.,air_moisture=lambda t:.05):
        uniform=np.r_[np.full(m.n,50.),np.full(m.n,.05)]
        np.testing.assert_allclose(q3.right_hand_side(uniform,8000,nr,7,.02/nr,.125/7),0.,atol=1e-14)
    # Non-default dt now lands exactly on every requested second.
    _,T,C=q1.run(N=20,dt=.6,t_end=3.,Tair_interp=lambda t:50.,Cair_interp=lambda t:.05)
    assert np.all(T>=28.-1e-10) and np.all(C>0.)
    assert q1.run.last_balance['relative_balance_error']<MASS_BALANCE_TOL
    q2t,q2T,q2C=q2.solve(nr=6,nz=5,end=100,export=False)
    r=q3.simulate_bdf(n_radial=6,n_axial=5,max_duration=100,record_interval=1.,stop_at_threshold=False)
    np.testing.assert_allclose(q2T,r['temperature_c'][1:],rtol=0,atol=0)
    np.testing.assert_allclose(q2C,r['moisture_kg_per_kg'][1:],rtol=0,atol=0)
    assert r['conservation']['balance_pass']
    print('PASS: 稳态、第一问任意步长记录与守恒、第二/三问相同时段逐值一致',flush=True)
    from types import SimpleNamespace
    base=dict(preheat=False,no_smooth=False,no_shrink=False,
        project=ROOT,material_mode=q4.MATERIAL_MODE)
    assert q4.xlsx_target(SimpleNamespace(**base))==ROOT/'result4.xlsx'
    for flag in ('preheat','no_smooth','no_shrink'):
        assert q4.xlsx_target(SimpleNamespace(**(base|{flag:True}))) is None
    assert q4.xlsx_target(SimpleNamespace(**(base|{'material_mode':'specified'}))) is None
    print('PASS: 第四问只导出主方案 result4.xlsx；敏感性/对照跑法不落盘',flush=True)
    for p in ROOT.rglob('*.py'):
        compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
    return q1,q2,q3,q4


# 交付表所用归一化半径探针（xi=r/R(t)，对应 0、0.5、1、1.5、2 cm）
XI_PROBE=np.linspace(0.,1.,21)

# 分层时间窗：初始边界层与交付时段分开判定，避免“全程最大差”被最初几十秒主导。
LAYER_WINDOWS=(('0-100s',0.,100.),('100-600s',100.,600.),('600s-6h',600.,21600.),
               ('t>=6h',21600.,None),('全程',0.,None))
# Richardson 估计的收敛阶；三层网格实测比值≈4，见 --full 日志中的 ratio。
RICHD_ORDER=2.

# 第一问交付网格是 N=3200、dt=0.25（见 solve_problem1.main），因此空间序列必须包含它。
Q1_GRIDS=((800,.25),(1600,.25),(3200,.25),(3200,.125))
# 第二至四问交付网格是 320 径向 × 40 半轴向；保留 40/80/160 序列用于收敛阶，
# 时间对照在交付的 320 网格上把 rtol 缩小 10 倍，轴向对照取 160x20/40/80。
Q34_CONFIGS=((40,40,RTOL,300.),(80,40,RTOL,300.),(160,40,RTOL,300.),
             (320,40,RTOL,300.),(320,40,RTOL/10,150.),
             (160,20,RTOL,300.),(160,80,RTOL,300.))


def on_probe(result,key,xi=XI_PROBE):
    """把任意径向探针列插值到统一归一化半径 xi；避免不同列坐标被当成同一位置。"""
    values=np.asarray(result[key],dtype=float)
    if values.ndim!=2:
        raise ValueError('%s 不是二维输出' % key)
    src=np.asarray(result['xi'],dtype=float) if 'xi' in result else np.linspace(0.,1.,values.shape[1])
    if src.ndim!=1 or src.size!=values.shape[1]:
        raise ValueError('%s 的 xi 列数与输出不一致' % key)
    if src.size==xi.size and np.array_equal(src,xi):
        return values
    return np.stack([np.interp(xi,src,row) for row in values])


def profile_error(a,b,t_from=0.0,t_to=None):
    """公共时间轴、公共归一化半径上的最大差，并给出最大差出现的时刻与半径。

    两组结果的时间步可能不同（例如不同网格的BDF自适应步长），因此先取并集时间轴，
    再各自插值；这样不会把“不同时刻”当成同一时刻比较。
    """
    ta=np.asarray(a['time_s'],dtype=float)
    tb=np.asarray(b['time_s'],dtype=float)
    hi=min(ta[-1],tb[-1]) if t_to is None else min(float(t_to),ta[-1],tb[-1])
    grid=np.unique(np.r_[ta[(ta>=t_from)&(ta<=hi)],tb[(tb>=t_from)&(tb<=hi)]])
    if grid.size==0:
        raise ValueError('时间窗 [%g,%g] 内没有可比较时刻' % (t_from,hi))
    out={'n_times':int(grid.size),'t_from':float(t_from),'t_to':float(hi)}
    for key,name in (('temperature_c','T'),('moisture_kg_per_kg','C')):
        av,bv=on_probe(a,key),on_probe(b,key)
        ai=np.column_stack([np.interp(grid,ta,av[:,j]) for j in range(av.shape[1])])
        bi=np.column_stack([np.interp(grid,tb,bv[:,j]) for j in range(bv.shape[1])])
        diff=np.abs(ai-bi)
        row,col=np.unravel_index(int(np.argmax(diff)),diff.shape)
        out[name]=float(diff[row,col])
        out[name+'_t_s']=float(grid[row])
        out[name+'_xi']=float(XI_PROBE[col])
    return out


def compare(label,a,b,t_from=0.0,t_to=None,dry=True,tol_t=None,tol_c=None,tol_dry=None,flags=True):
    """按时间窗比较两组输出；flags=False 时只记录，不给出通过与否。"""
    tol_c=PROFILE_C_TOL if tol_c is None else tol_c
    tol_t=PROFILE_T_TOL if tol_t is None else tol_t
    tol_dry=DRY_TIME_TOL_S if tol_dry is None else tol_dry
    err=profile_error(a,b,t_from,t_to)
    if flags:
        err['C_pass']=err['C']<=tol_c
        err['T_pass']=err['T']<=tol_t
    if dry and a.get('crossing_time_s') is not None and b.get('crossing_time_s') is not None:
        err['dry_s']=abs(a['crossing_time_s']-b['crossing_time_s'])
        if flags:
            err['dry_pass']=bool(err['dry_s']<=tol_dry)
    print(label,err,flush=True)
    return err


def spatial_verdict(label,coarse,mid,fine,t_from=21600.,t_to=None):
    """三网格加密：报告相邻两级差与 Richardson 估计的最细网格(即交付网格)误差。

    二阶方法在步长减半时相邻差按 1/4 递减；用最细一级差 / (2^p-1) 估计最细网格
    相对极限解的误差。门槛仍是交付四位小数对应的 5e-5。
    """
    d1=profile_error(coarse,mid,t_from,t_to)
    d2=profile_error(mid,fine,t_from,t_to)
    factor=2.**RICHD_ORDER-1.
    est={'n_times':d2['n_times'],'t_from':d2['t_from'],'t_to':d2['t_to'],
         'C':d2['C']/factor,'T':d2['T']/factor,
         'C_t_s':d2['C_t_s'],'C_xi':d2['C_xi'],'T_t_s':d2['T_t_s'],'T_xi':d2['T_xi']}
    est['C_ratio']=float(d1['C']/d2['C']) if d2['C']>0 else float('inf')
    est['T_ratio']=float(d1['T']/d2['T']) if d2['T']>0 else float('inf')
    est['C_pass']=bool(est['C']<=PROFILE_C_TOL)
    est['T_pass']=bool(est['T']<=PROFILE_T_TOL)
    print(label,est,flush=True)
    return est


def layer_report(label,a,b,dry=False):
    """逐时间窗记录一组加密的两级差（不设门槛，供论文分层说明）。

    数据时长不足以覆盖某个窗口时记为 skipped，而不是抛错：第一问只算到 1800s，
    不存在 t>=6h 的交付时段。
    """
    rows={}
    for name,lo,hi in LAYER_WINDOWS:
        ta=np.asarray(a['time_s'],dtype=float)
        tb=np.asarray(b['time_s'],dtype=float)
        hi_eff=min(ta[-1],tb[-1]) if hi is None else min(hi,ta[-1],tb[-1])
        if hi_eff<=lo:
            rows[name]={'t_from':float(lo),'t_to':float(hi_eff),'n_times':0,'skipped':True}
            print('%s [%s]' % (label,name),rows[name],flush=True)
            continue
        rows[name]=compare('%s [%s]' % (label,name),a,b,t_from=lo,t_to=hi,
                           dry=(dry and name=='全程'),flags=False)
    return rows


def q1_block(q1):
    """第一问：交付网格是 N=3200、dt=0.25，四个入口中只有它是显式时间步。"""
    gates=[]
    air_t,air_T,air_C=q1.load_attachment1(q1.ATTACH1)
    runs={}
    for n,dt in Q1_GRIDS:
        t0=time.perf_counter()
        t,T,C=q1.run(N=n,dt=dt,t_end=1800,
            Tair_interp=lambda t:np.interp(t,air_t,air_T),
            Cair_interp=lambda t:np.interp(t,air_t,air_C))
        ix=np.rint(np.arange(21)*n/20).astype(int)
        runs[(n,dt)]=dict(time_s=t,temperature_c=T[:,ix],moisture_kg_per_kg=C[:,ix],
                          xi=np.arange(21)/20.)
        print('Q1 N=%d dt=%g wall %.1fs' % (n,dt,time.perf_counter()-t0),
              q1.run.last_balance,flush=True)
    a,b,c=runs[(800,.25)],runs[(1600,.25)],runs[(3200,.25)]
    layer_report('Q1空间800->1600',a,b)
    layer_report('Q1空间1600->3200',b,c)
    # 交付表从 t=100s 起；最初 1s 的表面边界层只记录，不并入四位小数门槛。
    gates.append(spatial_verdict('Q1空间800/1600/3200 t>=100s 门槛(交付网格3200)',a,b,c,t_from=100.))
    gates.append(compare('Q1空间1600->3200 [0,1s] 边界层(只记录)',b,c,t_from=0.,t_to=1.,dry=False,flags=False))
    gates.append(compare('Q1时间 dt0.25->0.125(3200网格) t>=100s 门槛',c,runs[(3200,.125)],
                         t_from=100.,dry=False))
    return gates


def q3_block(q3):
    gates=[]

    # ---------------- 第三问：径向40/80/160/320(nz=40，交付列均为节点)+轴向与时间对照 ----------------
    q3_runs={}
    for nr,nz,rtol,max_step in Q34_CONFIGS:
        t0=time.perf_counter()
        r=q3.simulate_bdf(n_radial=nr,n_axial=nz,rtol=rtol,max_step=max_step)
        q3_runs[(nr,nz,rtol)]=r
        print('Q3 %dx%d rtol=%g wall %.1fs crossing_h %.6f balance %.2e' % (
            nr,nz,rtol,time.perf_counter()-t0,r['crossing_time_s']/3600,
            r['conservation']['relative_balance_error']),flush=True)
    c3,m3,f3,d3=(q3_runs[(40,40,RTOL)],q3_runs[(80,40,RTOL)],
                 q3_runs[(160,40,RTOL)],q3_runs[(320,40,RTOL)])
    layer_report('Q3径向40->80',c3,m3,dry=True)
    layer_report('Q3径向80->160',m3,f3,dry=True)
    layer_report('Q3径向160->320',f3,d3,dry=True)
    gates.append(spatial_verdict('Q3径向80/160/320 t>=6h 门槛(交付网格320x40)',m3,f3,d3))
    layer_report('Q3轴向160x20->160x40',q3_runs[(160,20,RTOL)],f3)
    layer_report('Q3轴向160x40->160x80',f3,q3_runs[(160,80,RTOL)])
    gates.append(compare('Q3轴向160x40->160x80 t>=6h 门槛',f3,q3_runs[(160,80,RTOL)],t_from=21600.))
    gates.append(compare('Q3时间 rtol/10 t>=6h 门槛',d3,q3_runs[(320,40,RTOL/10)],t_from=21600.))
    # 全程对照里 C/T 只记录：分层标准规定 5e-5 门槛只对交付时段(t>=6h)生效，
    # 全域最大差会被最初数小时的表面边界层主导；烘干时间仍按 1s 单独判定。
    whole=compare('Q3时间 rtol/10 全程(含<=1s烘干时间；C/T 只记录)',d3,
                  q3_runs[(320,40,RTOL/10)],flags=False)
    gates.append({'dry_s':float(whole['dry_s']),
                  'dry_pass':bool(whole['dry_s']<=DRY_TIME_TOL_S)})
    return gates


def q4_block(q4):
    gates=[]
    # ---------------- 第四问：同网格口径；收缩半径由附件2插值 ----------------
    from types import SimpleNamespace
    q4_runs={}
    for nr,nz,rtol,max_step in Q34_CONFIGS:
        args=SimpleNamespace(project=ROOT,nr=nr,nz=nz,no_shrink=False,preheat=False,
            no_smooth=False,material_mode=q4.MATERIAL_MODE,rtol=rtol,max_step=max_step,
            validation_only=True)
        t0=time.perf_counter()
        r=q4.run(args)
        q4_runs[(nr,nz,rtol)]=r
        print('Q4 %dx%d rtol=%g wall %.1fs crossing_h %.6f balance %.2e' % (
            nr,nz,rtol,time.perf_counter()-t0,r['crossing_time_s']/3600,
            r['conservation']['relative_balance_error']),flush=True)
    c4,m4,f4,d4=(q4_runs[(40,40,RTOL)],q4_runs[(80,40,RTOL)],
                 q4_runs[(160,40,RTOL)],q4_runs[(320,40,RTOL)])
    layer_report('Q4径向40->80',c4,m4,dry=True)
    layer_report('Q4径向80->160',m4,f4,dry=True)
    layer_report('Q4径向160->320',f4,d4,dry=True)
    gates.append(spatial_verdict('Q4径向80/160/320 t>=6h 门槛(交付网格320x40)',m4,f4,d4))
    layer_report('Q4轴向160x20->160x40',q4_runs[(160,20,RTOL)],f4)
    layer_report('Q4轴向160x40->160x80',f4,q4_runs[(160,80,RTOL)])
    gates.append(compare('Q4轴向160x40->160x80 t>=6h 门槛',f4,q4_runs[(160,80,RTOL)],t_from=21600.))
    gates.append(compare('Q4时间 rtol/10 t>=6h 门槛',d4,q4_runs[(320,40,RTOL/10)],t_from=21600.))
    # 同第三问：全程 C/T 只记录，只有烘干时间参与 1s 判定。
    whole=compare('Q4时间 rtol/10 全程(含<=1s烘干时间；C/T 只记录)',d4,
                  q4_runs[(320,40,RTOL/10)],flags=False)
    gates.append({'dry_s':float(whole['dry_s']),
                  'dry_pass':bool(whole['dry_s']<=DRY_TIME_TOL_S)})
    return gates


def cross_block(q3,q4):
    gates=[]
    # ---------------- 跨实现对照：同一物性、不收缩、同网格同容差 ----------------
    original=q4.DryingModel.material
    def common_material(self,t,T,C):
        rho,cp,k=q3.material_properties(C,t)
        return rho*cp,k,q3.moisture_diffusivity(C,T,t)
    q4.DryingModel.material=common_material
    try:
        cross_args=SimpleNamespace(project=ROOT,nr=80,nz=40,no_shrink=True,preheat=False,
            no_smooth=False,material_mode='specified',rtol=RTOL,max_step=300.,validation_only=True)
        cross4=q4.run(cross_args)
    finally:
        q4.DryingModel.material=original
    # 记录间隔必须与第四问 validation_only 的 60 s 一致：两组输出时间轴不同会在
    # profile_error 的并集时间轴上引入线性插值误差，早期边界层可达 0.1 kg/kg 量级。
    cross3=q3.simulate_bdf(n_radial=80,n_axial=40,max_duration=10800,record_interval=60.,
                           stop_at_threshold=False)
    gates.append(compare('第三/四问同物性对照(前3h)',cross3,cross4))
    return gates


def full(q1,q2,q3,q4,stages=('q1','q3','q4','cross')):
    gates=[]
    for stage in stages:
        t0=time.perf_counter()
        print('\n---------- stage %s ----------' % stage,flush=True)
        if stage=='q1':
            gates+=q1_block(q1)
        elif stage=='q3':
            gates+=q3_block(q3)
        elif stage=='q4':
            gates+=q4_block(q4)
        elif stage=='cross':
            gates+=cross_block(q3,q4)
        print('---------- stage %s wall %.1fs ----------' % (stage,time.perf_counter()-t0),flush=True)
    # 守恒与四舍五入不是同一件事：守恒通过不代表四位小数收敛。
    accepted=all(value for gate in gates for key,value in gate.items() if key.endswith('_pass'))
    failed=[(key,value) for gate in gates for key,value in gate.items()
            if key.endswith('_pass') and not value]
    if failed:
        print('未通过的门槛:',failed,flush=True)
    print('整体精度目标:', 'PASS' if accepted else 'NOT MET；不得宣称所有输出四位小数均已收敛',flush=True)
    return accepted


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full',action='store_true')
    p.add_argument('--stage',default='all',
                   choices=['all','q1','q3','q4','cross'],
                   help='只跑某一组网格/时间检查，便于分进程并行与分段留档')
    a=p.parse_args()
    modules=quick()
    stages=('q1','q3','q4','cross') if a.stage=='all' else (a.stage,)
    if a.full and not full(*modules,stages=stages):
        raise SystemExit(2)
