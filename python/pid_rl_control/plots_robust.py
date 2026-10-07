import numpy as np, pickle, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from robustness import *
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3})
res = pickle.load(open("data/robust_results.pkl", "rb"))
CP, CR = "C0", "C3"
SHOW = {"delay": .13, "g": 6.5, "m": 20, "c": -1.9, "k": -5.0, "tau": .25, "wf": 25, "rate": 25, "q": .2, "Ts": .2}
TR = {"STABLE": "stable", "WEAK": "weakly damped", "OSCILLATING": "sustained oscillation", "DIVERGED": "DIVERGED"}

# ---- Figure 5: oscillation amplitude vs severity ----
fig, axs = plt.subplots(5, 2, figsize=(11, 15)); axs = axs.ravel()
for ax, (key, title, unit, grid) in zip(axs, SCEN):
    rows, th = res[key]["rows"], res[key]["th"]
    for nm, col in (("PID", CP), ("RL", CR)):
        R_ = [r_ for r_ in rows if r_[1] == nm]
        v = np.array([r_[0] for r_ in R_]); A = np.array([r_[3] for r_ in R_]); cl = np.array([r_[2] for r_ in R_])
        div = cl == "DIVERGED"; Ap = np.clip(np.where(div, np.nan, A), 1e-4, 12)
        ax.plot(v, Ap, "-", c=col, lw=1.3, label=nm)
        for c_, mk, kw in (("STABLE", ".", {}), ("WEAK", "o", dict(mfc="white")), ("OSCILLATING", "o", {})):
            sel = cl == c_; ax.plot(v[sel], Ap[sel], mk, c=col, ms=4, **kw)
        ax.plot(v[div], np.full(div.sum(), 30), "x", c=col, ms=6, mew=1.8)
        if th[nm] is not None: ax.axvline(th[nm], c=col, ls=":", lw=1.5)
    ax.axhline(OSC_TH, c="k", ls="--", lw=.8); ax.axhspan(15, 60, color="0.92", zorder=0)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 60)
    if key in LOG: ax.set_xscale("log")
    if key in ("wf", "rate", "c", "k"): ax.invert_xaxis()
    ax.set_title(title, fontsize=10.5); ax.set_xlabel(unit + "  → more severe", fontsize=9)
    ax.set_ylabel("sustained oscillation [m]", fontsize=9)
axs[0].legend(loc="center right", fontsize=9)
fig.suptitle("Sustained oscillation measurement (last 10 s, peak-to-peak)\n● sustained  ○ slowly decaying  × diverged   ┄ 5 cm threshold   ⋮ loss-of-stability point", y=.998, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, .985)); fig.savefig("figures/5_robustness_measurement.png", dpi=120)

# ---- Figure 6: time responses at a representative severity ----
t = np.arange(N)*DT; r = np.repeat(REFS, N_SEG)
fig, axs = plt.subplots(5, 2, figsize=(11, 15), sharex=True); axs = axs.ravel()
for ax, (key, title, unit, grid) in zip(axs, SCEN):
    v = SHOW[key]; lab = []
    ax.plot(t, r, "k--", lw=.8)
    for nm, Cls, col in (("PID", PIDc, CP), ("RL", RLc, CR)):
        ys, us, dv = simulate(Cls(), {key: float(v)}); cl, A = classify(ys, dv)
        ax.plot(t, np.clip(ys, -3, 3), c=col, lw=1.2, alpha=.85); lab.append(f"{nm}: {TR[cl]}")
    ax.set_ylim(-2.2, 2.8); ax.set_title(f"{title}\n{unit.split(' (')[0]} = {v}   |   " + "  ·  ".join(lab), fontsize=9.5)
for a in axs[-2:]: a.set_xlabel("time [s]")
fig.suptitle("Position output y(t) at a representative severity — blue PID, red RL, dashed reference", y=.997, fontsize=11)
fig.tight_layout(); fig.savefig("figures/6_time_responses.png", dpi=120)
