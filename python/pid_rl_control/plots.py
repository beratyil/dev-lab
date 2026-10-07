import numpy as np, pickle, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from plant_pid import *
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3})
D = pickle.load(open("data/results.pkl", "rb")); log, snaps = D["log"], D["snaps"]
pid = PID(12, 6, 5); y_pid, u_pid = simulate(pid, r_test); iae_pid = metrics(y_pid, u_pid, r_test)["IAE"]

# ---- Figure 1: plant + PID ----
x = v = 0.0; yo = []
for _ in t_test[:400]: yo.append(x); x, v = plant_step(x, v, 1.0)
fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=False)
ax[0].plot(t_test[:400], yo, c="gray"); ax[0].axhline(1, ls="--", c="k", lw=.8)
ax[0].set_title("Plant alone: u = 1 N step (open loop) — oscillatory, ζ = 0.2")
ax[0].set_ylabel("position y [m]")
ax[1].plot(t_test, r_test, "k--", lw=1, label="reference r"); ax[1].plot(t_test, y_pid, c="C0", lw=2, label="output y (PID)")
ax[1].set_title(f"Closed-loop PID (Kp=12, Ki=6, Kd=5) — IAE = {iae_pid:.2f}"); ax[1].set_ylabel("position y [m]"); ax[1].legend(loc="upper right")
ax[2].plot(t_test, u_pid, c="C1"); ax[2].set_ylabel("force u [N]"); ax[2].set_xlabel("time [s]"); ax[2].set_title("PID control signal")
fig.tight_layout(); fig.savefig("figures/1_plant_and_pid.png", dpi=130)

# ---- Figure 2: training curve ----
it = np.array(log["it"]); bi = snaps["best"][3]
fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
sm = lambda a, k=7: np.convolve(a, np.ones(k)/k, mode="valid")
ax[0].plot(it, log["ret"], c="C2", alpha=.3); ax[0].plot(it[6:], sm(log["ret"]), c="C2", lw=2)
ax[0].set_ylabel("episode reward"); ax[0].set_title("PPO training — mean reward of training episodes"); ax[0].set_ylim(-120, 0)
ax[1].plot(it, log["iae"], c="C3", label="test profile IAE (RL)")
ax[1].axhline(iae_pid, c="C0", ls="--", label=f"PID IAE = {iae_pid:.2f}")
ax[1].axvline(bi, c="k", ls=":", label=f"selected model (it {bi}, by validation)")
ax[1].set_ylim(0, 8); ax[1].set_xlabel("PPO iteration (1 iter = 24 episodes × 20 s)"); ax[1].set_ylabel("IAE (lower = better)"); ax[1].legend()
fig.tight_layout(); fig.savefig("figures/2_training_curve.png", dpi=130)

# ---- Figure 3: output during training ----
keys = [0, 5, 20, 80, "best", 240]
fig, ax = plt.subplots(3, 2, figsize=(11, 9), sharex=True, sharey=True); ax = ax.ravel()
for a, k in zip(ax, keys):
    ys, us, iae = snaps[k][:3]
    a.plot(t_test, r_test, "k--", lw=1); a.plot(t_test, ys, c="C3", lw=2)
    ttl = f"iteration {snaps[k][3]} (selected)" if k == "best" else f"iteration {k}" + (" — empty model" if k == 0 else "")
    a.set_title(f"{ttl}   IAE={iae:.2f}"); a.set_ylim(-1, 1.5)
for a in ax[4:]: a.set_xlabel("time [s]")
fig.suptitle("Plant output (y) under the RL controller as training progresses", y=.995); fig.tight_layout(); fig.savefig("figures/3_output_during_training.png", dpi=130)

# ---- Figure 4: comparison ----
yb, ub, iaeb, _ = snaps["best"]
fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ax[0].plot(t_test, r_test, "k--", lw=1, label="reference"); ax[0].plot(t_test, y_pid, c="C0", lw=2, label=f"PID (IAE {iae_pid:.2f})")
ax[0].plot(t_test, yb, c="C3", lw=2, label=f"RL (IAE {iaeb:.2f})"); ax[0].set_ylabel("position y [m]"); ax[0].legend(loc="upper right")
ax[0].set_title("Same test scenario: PID vs RL-trained neural network")
ax[1].plot(t_test, u_pid, c="C0", label="PID"); ax[1].plot(t_test, ub, c="C3", label="RL", alpha=.85)
ax[1].set_ylabel("force u [N]"); ax[1].set_xlabel("time [s]"); ax[1].legend()
fig.tight_layout(); fig.savefig("figures/4_pid_vs_rl.png", dpi=130)

# ---- numerical summary ----
def step_stats(y, u):
    out = []
    for a, b, r0, r1 in [(0, 200, 0, 1.0), (200, 400, 1.0, -0.5), (400, 600, -0.5, 0.8)]:
        seg = (y[a:b] - r0)/(r1 - r0); os_ = (seg.max() - 1)*100
        bad = np.where(np.abs(seg - 1) > .02)[0]; ts = (bad[-1] + 1)*DT if len(bad) else 0
        out.append((round(os_, 1), round(ts, 2), round(abs(r1 - y[b-1]), 4)))
    tv = np.abs(np.diff(u)).sum()
    return out, round(tv, 1), round(np.sum(u**2)*DT, 1)
print("PID (overshoot %, settling s, final error):", step_stats(y_pid, u_pid))
print("RL  (overshoot %, settling s, final error):", step_stats(yb, ub))
