"""Extra analyses (produce the validation and interpretation numbers in the README):
  1) Linearisation of the RL network around the operating point -> equivalent Kp, Kd, feedforward
  2) Discrete-time linear theory for the PID -> phase margin, delay margin, gain margin
  3) Effect of saturation -> stability limits with small (5 cm) and large (1 m) steps
Requires: numpy, scipy, data/results.pkl
"""
import numpy as np
from scipy.signal import cont2discrete
import robustness as Rb
from plant_pid import U_MAX

# ---------------- 1) RL equivalent gains ----------------
print("1) Linearisation of the RL network (at e=0, dy=0)")
u = lambda o: U_MAX*np.tanh(Rb.rl_raw(np.array(o, dtype=float)))
for r in [1.0, -0.5, 0.8, 0.0]:
    o, eps = np.array([r, 0.0, 0.0]), 1e-4
    J = [(u(o + eps*np.eye(3)[i]) - u(o - eps*np.eye(3)[i]))/(2*eps) for i in range(3)]
    print(f"   r={r:5.1f}  u={u(o):6.3f}  feedforward du/dr={J[0]:5.2f}  "
          f"Kp_eq=du/de={J[1]:6.2f}  Kd_eq=-du/d(dy)={-J[2]:5.2f}")
print("   (for comparison: PID Kp=12, Ki=6, Kd=5)\n")

# ---------------- 2) PID discrete-time margins ----------------
def loop_tf(m=1.0, c=0.4, k=1.0, T=0.05):
    A = np.array([[0, 1], [-k/m, -c/m]]); B = np.array([[0], [1/m]]); C = np.array([[1, 0]])
    Ad, Bd, Cd, _, _ = cont2discrete((A, B, C, np.zeros((1, 1))), T, "zoh")
    w = np.linspace(0.01, np.pi/T*0.999, 6000); z = np.exp(1j*w*T)
    P = np.array([(Cd @ np.linalg.solve(zz*np.eye(2) - Ad, Bd))[0, 0] for zz in z])
    Cz = 12 + 6*T*z/(z - 1) + 5*(z - 1)/(T*z)          # same discretisation as plant_pid.PID
    return w, Cz*P
w, L = loop_tf()
i = np.argmin(abs(abs(L) - 1)); pm = np.angle(L[i], deg=True) + 180
ph = np.unwrap(np.angle(L)); j = np.argmin(abs(ph + np.pi))
print("2) PID discrete-time linear analysis (20 Hz, ZOH)")
print(f"   crossover frequency {w[i]:.2f} rad/s, phase margin {pm:.1f}°, "
      f"delay margin ≈ {np.deg2rad(pm)/w[i]:.3f} s, gain margin ≈ {1/abs(L[j]):.2f}×\n")

# ---------------- 3) Effect of saturation ----------------
print("3) Stability limit: large (1 m) vs small (5 cm, unsaturated) steps")
orig = list(Rb.REFS)
cases = [("c", "neg. damping", np.linspace(0.4, -15, 32), False),
         ("k", "neg. spring", np.linspace(1, -14, 31), False),
         ("delay", "dead time", np.round(np.arange(0, 1.001, 0.02), 3), False),
         ("m", "mass", np.geomspace(1, 60, 31), True)]
for key, name, grid, lg in cases:
    out = []
    for nm, Cls in (("PID", Rb.PIDc), ("RL", Rb.RLc)):
        Rb.REFS[:] = orig; Rb.OSC_TH = 0.05
        big = Rb.threshold(Cls, key, grid, lg)
        Rb.REFS[:] = [0.05, -0.025, 0.04]; Rb.OSC_TH = 0.0025     # threshold is scaled by 1/20 too
        small = Rb.threshold(Cls, key, grid, lg)
        f = lambda v: "no failure" if v is None else f"{v:.3f}"
        out.append(f"{nm}: large={f(big)}, small={f(small)}")
    print(f"   {name:14s} " + "   |   ".join(out))
Rb.REFS[:] = orig; Rb.OSC_TH = 0.05
