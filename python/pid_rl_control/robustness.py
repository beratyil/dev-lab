"""Robustness test of the PID and RL controllers in 10 destabilising scenarios.
In each scenario a 'severity' parameter is swept and the closed loop is simulated at every point."""
import numpy as np, pickle, math, time
from plant_pid import DT, U_MAX

# ---------------- Controllers (support arbitrary dt) ----------------
class PIDc:
    kp, ki, kd = 12.0, 6.0, 5.0
    def reset(s): s.i, s.yp = 0.0, 0.0
    def __call__(s, r, y, dt):
        e = r - y; d = -(y - s.yp)/dt
        u_raw = s.kp*e + s.ki*(s.i + e*dt) + s.kd*d
        u = min(max(u_raw, -U_MAX), U_MAX)
        if u == u_raw: s.i += e*dt
        s.yp = y
        return u

W = pickle.load(open("data/results.pkl", "rb"))["pi"]          # network selected at iteration 102
def rl_raw(o):
    h = np.tanh(o @ W[0] + W[1]); h = np.tanh(h @ W[2] + W[3]); return (h @ W[4] + W[5])[..., 0]
class RLc:
    def reset(s): s.yp = 0.0
    def __call__(s, r, y, dt):
        o = np.array([r, r - y, (y - s.yp)/dt]); s.yp = y
        return U_MAX*float(np.clip(np.tanh(rl_raw(o)), -1, 1))

# ---------------- Generalised plant + loop ----------------
T_SEG, REFS = 30.0, [1.0, -0.5, 0.8]
N_SEG = int(T_SEG/DT); N = 3*N_SEG

def simulate(ctrl, p):
    m, c, k = p.get("m", 1.0), p.get("c", 0.4), p.get("k", 1.0)
    g = p.get("g", 1.0); tau = p.get("tau", 0.0); wf = p.get("wf"); R = p.get("rate"); q = p.get("q", 0.0)
    h = 0.0025 if (wf and wf > 20) else 0.01                     # RK4 sub-step
    dsub = int(round(p.get("delay", 0.0)/h))                     # dead time (sub-step resolution)
    nTs = int(round(p.get("Ts", DT)/h))                          # controller call period
    if wf:                                                       # flexible mode: actuator on m1, sensor on m2
        m1, m2 = 0.8*m, 0.2*m; s_ = 1/m1 + 1/m2
        ks, cs = wf**2/s_, 2*0.02*wf/s_
    lag = 1 - math.exp(-h/tau) if tau > 0 else 1.0
    Nsub = int(round(3*T_SEG/h)); per_seg = Nsub//3; hpd = int(round(DT/h))
    ysub = np.full(Nsub, np.nan); usub = np.full(Nsub, np.nan)
    x1 = v1 = x2 = v2 = ua = u = 0.0; diverged = False
    ctrl.reset()
    for n in range(Nsub):
        r = REFS[n // per_seg]
        y = x2 if wf else x1
        ysub[n] = y
        if n % nTs == 0:
            ym = ysub[n - dsub] if n >= dsub else 0.0
            if q > 0: ym = q*round(ym/q)                         # quantisation
            u = ctrl(r, ym, nTs*h)
        usub[n] = u
        if R: ua += max(-R*h, min(R*h, u - ua))                  # rate limit
        else: ua += lag*(u - ua)                                 # first-order actuator (tau=0 -> ideal)
        F = g*ua                                                 # actuator gain
        if wf:
            def f(x1, v1, x2, v2):
                fs = ks*(x1 - x2) + cs*(v1 - v2)
                return v1, (F - c*v1 - k*x1 - fs)/m1, v2, fs/m2
        else:
            def f(x1, v1, x2, v2): return v1, (F - c*v1 - k*x1)/m, 0.0, 0.0
        a = f(x1, v1, x2, v2)
        b = f(x1+.5*h*a[0], v1+.5*h*a[1], x2+.5*h*a[2], v2+.5*h*a[3])
        cc = f(x1+.5*h*b[0], v1+.5*h*b[1], x2+.5*h*b[2], v2+.5*h*b[3])
        dd = f(x1+h*cc[0], v1+h*cc[1], x2+h*cc[2], v2+h*cc[3])
        x1 += h/6*(a[0]+2*b[0]+2*cc[0]+dd[0]); v1 += h/6*(a[1]+2*b[1]+2*cc[1]+dd[1])
        x2 += h/6*(a[2]+2*b[2]+2*cc[2]+dd[2]); v2 += h/6*(a[3]+2*b[3]+2*cc[3]+dd[3])
        yy = x2 if wf else x1
        if not math.isfinite(yy) or abs(yy) > 50: diverged = True; break
    return ysub[::hpd], usub[::hpd], diverged

OSC_TH = 0.05       # 5 cm peak-to-peak = 5% of the reference
def classify(ys, diverged):
    """DIVERGED / OSCILLATING (sustained) / WEAK (decaying but slowly) / STABLE"""
    if diverged: return "DIVERGED", np.inf
    pp = lambda w: w.max() - w.min(); i10, i20 = int(10/DT), int(20/DT)
    best = (0.0, 1.0)
    for sg in range(3):
        o = sg*N_SEG
        A2 = pp(ys[o+i20:o+N_SEG]); A1 = pp(ys[o+i10:o+i20])
        if A2 > best[0]: best = (A2, A2/max(A1, 1e-9))
    A, ratio = best
    if np.abs(ys).max() > 5: return "DIVERGED", A
    if A >= OSC_TH and ratio >= 0.75: return "OSCILLATING", A
    if A >= OSC_TH: return "WEAK", A
    return "STABLE", A

FAIL = ("OSCILLATING", "DIVERGED")
def is_fail(C, key, v): return classify(*simulate(C, {key: v})[::2])[0] in FAIL

def threshold(Cls, key, grid, log=False):
    """First loss of stability in the sweep, refined by bisection between the two neighbouring points."""
    prev = grid[0]
    for v in grid:
        if is_fail(Cls(), key, float(v)):
            if v == grid[0]: return float(v)
            lo, hi = float(prev), float(v)
            for _ in range(10):
                mid = math.sqrt(lo*hi) if log else (lo + hi)/2
                if is_fail(Cls(), key, mid): hi = mid
                else: lo = mid
            return hi
        prev = v
    return None

# ---------------- 10 scenarios ----------------
SCEN = [
 ("delay", "1) Dead time (measurement delay)", "delay [s]", np.round(np.arange(0, 1.001, 0.02), 3)),
 ("g",     "2) Actuator gain increase", "gain factor [×]", np.geomspace(1, 60, 31)),
 ("m",     "3) Mass increase (extra load)", "mass [kg]", np.geomspace(1, 60, 31)),
 ("c",     "4) Negative damping (energy pumping)", "damping c [N·s/m]", np.linspace(0.4, -15, 32)),
 ("k",     "5) Negative spring (open-loop unstable plant)", "spring k [N/m]", np.linspace(1, -14, 31)),
 ("tau",   "6) Actuator lag (first order)", "time constant τ [s]", np.round(np.linspace(0, 3, 31), 3)),
 ("wf",    "7) Unmodelled flexible mode (resonance)", "mode frequency [rad/s] (lower = worse)", np.geomspace(60, 0.5, 31)),
 ("rate",  "8) Actuator rate limit", "max. rate [N/s] (lower = worse)", np.geomspace(300, 0.5, 31)),
 ("q",     "9) Sensor quantisation (resolution)", "resolution [m]", np.round(np.linspace(0, 0.6, 31), 3)),
 ("Ts",    "10) Sampling period increase", "control period [s]", np.round(np.arange(0.05, 1.001, 0.025), 3)),
]

LOG = {"g", "m", "wf", "rate"}

if __name__ == "__main__":
    t0 = time.time(); res = {}
    for key, title, unit, grid in SCEN:
        rows = []
        for v in grid:
            for name, Cls in (("PID", PIDc), ("RL", RLc)):
                ys, us, dv = simulate(Cls(), {key: float(v)})
                cls, A = classify(ys, dv)
                rows.append((float(v), name, cls, A))
        th = {nm: threshold(Cls, key, grid, key in LOG) for nm, Cls in (("PID", PIDc), ("RL", RLc))}
        res[key] = dict(rows=rows, th=th)
        print(f"{title:46s} loss of stability -> PID: {th['PID']}   RL: {th['RL']}   ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(res, open("data/robust_results.pkl", "wb"))
