import numpy as np

# ---------------- PLANT: mass-spring-damper ----------------
# m*x'' + c*x' + k*x = u      (y = x, position is measured)
M, C, K = 1.0, 0.4, 1.0        # wn = 1 rad/s, zeta = 0.2 -> oscillatory
U_MAX = 10.0                   # actuator limit [N]
DT = 0.05                      # control period (20 Hz)
SUB = 5                        # plant sub-steps (RK4, 0.01 s)

def plant_step(x, v, u):
    h = DT / SUB
    f = lambda x, v: (v, (u - C*v - K*x) / M)
    for _ in range(SUB):
        k1x, k1v = f(x, v)
        k2x, k2v = f(x + .5*h*k1x, v + .5*h*k1v)
        k3x, k3v = f(x + .5*h*k2x, v + .5*h*k2v)
        k4x, k4v = f(x + h*k3x, v + h*k3v)
        x = x + h/6*(k1x + 2*k2x + 2*k3x + k4x)
        v = v + h/6*(k1v + 2*k2v + 2*k3v + k4v)
    return x, v

# ---------------- TEST REFERENCE ----------------
T_TEST = 30.0
t_test = np.arange(int(T_TEST/DT)) * DT
r_test = np.where(t_test < 10, 1.0, np.where(t_test < 20, -0.5, 0.8))

# ---------------- PID (derivative on measurement, anti-windup) ----------------
class PID:
    def __init__(s, kp, ki, kd): s.kp, s.ki, s.kd = kp, ki, kd
    def reset(s, y0=0.0): s.i, s.y_prev = 0.0, y0
    def __call__(s, r, y):
        e = r - y
        d = -(y - s.y_prev) / DT
        u_raw = s.kp*e + s.ki*(s.i + e*DT) + s.kd*d
        u = float(np.clip(u_raw, -U_MAX, U_MAX))
        if u == u_raw: s.i += e*DT          # freeze integrator while saturated
        s.y_prev = y
        return u

def simulate(ctrl, r_seq):
    x = v = 0.0
    ys, us = [], []
    ctrl.reset(0.0)
    for r in r_seq:
        y = x
        u = ctrl(r, y)
        ys.append(y); us.append(u)
        x, v = plant_step(x, v, u)
    return np.array(ys), np.array(us)

def metrics(y, u, r):
    e = r - y
    return dict(IAE=np.sum(np.abs(e))*DT, effort=np.sum(u**2)*DT)
