"""Trains an empty (randomly initialised) neural network as a controller with PPO.
The agent sees ONLY the plant output (y) and the reference; the reward comes ONLY from the output error.
It never has access to the PID's control signal u or to the plant's internal states (velocity etc.)."""
import numpy as np, pickle, time
from plant_pid import plant_step, DT, U_MAX, t_test, r_test

rng = np.random.default_rng(0)

# ---------------- Environment (vectorised, N parallel episodes) ----------------
class TrackEnv:
    def __init__(s, n, T=20.0):
        s.n, s.steps = n, int(T/DT)
    def _make_refs(s):
        R = np.zeros((s.steps, s.n))
        for j in range(s.n):
            k = 0
            while k < s.steps:
                L = rng.integers(int(3/DT), int(8/DT))
                R[k:k+L, j] = rng.uniform(-1, 1); k += L
        return R
    def obs(s):
        r = s.R[s.k]
        return np.stack([r, r - s.x, (s.x - s.y_prev)/DT], 1)   # all derived from the output
    def reset(s):
        s.R = s._make_refs(); s.k = 0
        s.x = np.zeros(s.n); s.v = np.zeros(s.n); s.y_prev = s.x.copy(); s.u_prev = np.zeros(s.n)
        return s.obs()
    def step(s, a_norm):
        u = U_MAX*np.clip(a_norm, -1, 1)
        r = s.R[s.k]
        s.y_prev = s.x.copy()
        s.x, s.v = plant_step(s.x, s.v, u)
        e = r - s.x
        rew = -(e**2 + 0.1*np.abs(e)) - 0.002*((u - s.u_prev)/U_MAX)**2*100
        s.u_prev = u; s.k += 1
        done = s.k >= s.steps
        return (None if done else s.obs()), rew, done

# ---------------- Small MLP with hand-written backprop ----------------
class MLP:
    def __init__(s, sizes, out_gain):
        s.P = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            g = np.sqrt(2) if i < len(sizes)-2 else out_gain
            q, _ = np.linalg.qr(rng.normal(size=(max(a, b), max(a, b))))
            s.P += [g*q[:a, :b], np.zeros(b)]
    def forward(s, x):
        s.cache = [x]; h = x; n = len(s.P)//2
        for i in range(n):
            h = h @ s.P[2*i] + s.P[2*i+1]
            if i < n-1: h = np.tanh(h)
            s.cache.append(h)
        return h
    def backward(s, g):
        n = len(s.P)//2; grads = [None]*len(s.P)
        for i in reversed(range(n)):
            if i < n-1: g = g*(1 - s.cache[i+1]**2)
            grads[2*i] = s.cache[i].T @ g; grads[2*i+1] = g.sum(0)
            g = g @ s.P[2*i].T
        return grads

class Adam:
    def __init__(s, params, lr):
        s.p, s.lr, s.t = params, lr, 0
        s.m = [np.zeros_like(p) for p in params]; s.v = [np.zeros_like(p) for p in params]
    def step(s, grads, max_norm=0.5):
        gn = np.sqrt(sum((g**2).sum() for g in grads))
        if gn > max_norm: grads = [g*max_norm/gn for g in grads]
        s.t += 1
        for i, g in enumerate(grads):
            s.m[i] = .9*s.m[i] + .1*g; s.v[i] = .999*s.v[i] + .001*g*g
            mh = s.m[i]/(1-.9**s.t); vh = s.v[i]/(1-.999**s.t)
            s.p[i] -= s.lr*mh/(np.sqrt(vh) + 1e-8)

pi = MLP([3, 64, 64, 1], out_gain=0.01)        # EMPTY model: random, output ~0
logstd = np.array([np.log(0.3)])
vf = MLP([3, 64, 64, 1], out_gain=1.0)
opt_pi = Adam(pi.P + [logstd], 3e-4)
opt_vf = Adam(vf.P, 1e-3)

def policy_det(o):                         # evaluation: deterministic (mean)
    return np.tanh(pi.forward(o)[:, 0])

def eval_test():
    x = v = 0.0; y_prev = 0.0; ys, us = [], []
    for r in r_test:
        o = np.array([[r, r - x, (x - y_prev)/DT]])
        u = U_MAX*float(np.clip(policy_det(o)[0], -1, 1))
        ys.append(x); us.append(u); y_prev = x
        x, v = plant_step(x, v, u)
    ys, us = np.array(ys), np.array(us)
    return ys, us, np.sum(np.abs(r_test - ys))*DT

# ---------------- PPO ----------------
N_ENV, ITERS, GAMMA, LAM, CLIP, EPOCHS, MB = 24, 240, 0.98, 0.95, 0.2, 10, 800
RSCALE = 0.1
env = TrackEnv(N_ENV)
val_env = TrackEnv(16); val_env.reset(); R_VAL = val_env.R.copy()   # fixed validation references
def eval_val():
    x = np.zeros(16); v = np.zeros(16); yp = x.copy(); iae = 0
    for r in R_VAL:
        o = np.stack([r, r - x, (x - yp)/DT], 1)
        u = U_MAX*np.clip(policy_det(o), -1, 1); yp = x.copy()
        x, v = plant_step(x, v, u); iae += np.abs(r - x).mean()*DT
    return iae
best = (np.inf, None, None)
log = dict(it=[], ret=[], iae=[], std=[], val=[]); snaps = {}
SNAP_AT = [0, 5, 20, 80, ITERS]
t0 = time.time()
for it in range(ITERS + 1):
    ys, us, iae = eval_test()
    if it in SNAP_AT: snaps[it] = (ys, us, iae)
    if it == ITERS: break
    val = eval_val()
    if val < best[0]: best = (val, [p.copy() for p in pi.P], it)
    frac = 1 - it/ITERS; opt_pi.lr = 3e-4*max(frac, 0.1); opt_vf.lr = 1e-3*max(frac, 0.1)
    # --- rollout ---
    o = env.reset(); O, A, LP, Rw, V = [], [], [], [], []
    std = np.exp(logstd[0])
    while True:
        mu = policy_det(o); a = mu + std*rng.normal(size=N_ENV)
        lp = -0.5*((a - mu)/std)**2 - logstd[0] - 0.5*np.log(2*np.pi)
        v = vf.forward(o)[:, 0]
        o2, rew, done = env.step(a)
        O.append(o); A.append(a); LP.append(lp); Rw.append(rew*RSCALE); V.append(v)
        if done: break
        o = o2
    last_obs = env.obs() if env.k < env.steps else np.stack(
        [env.R[-1], env.R[-1] - env.x, (env.x - env.y_prev)/DT], 1)
    last_v = vf.forward(last_obs)[:, 0]          # time limit -> bootstrap
    O, A, LP, Rw, V = map(np.array, (O, A, LP, Rw, V))
    adv = np.zeros_like(Rw); g = 0
    for t in reversed(range(len(Rw))):
        nv = last_v if t == len(Rw)-1 else V[t+1]
        d = Rw[t] + GAMMA*nv - V[t]; g = d + GAMMA*LAM*g; adv[t] = g
    ret = adv + V
    O = O.reshape(-1, 3); A, LP, adv, ret = A.ravel(), LP.ravel(), adv.ravel(), ret.ravel()
    adv = (adv - adv.mean())/(adv.std() + 1e-8)
    # --- update ---
    n = len(A)
    for ep in range(EPOCHS):
        perm = rng.permutation(n); kl_sum = 0
        for b in range(0, n, MB):
            idx = perm[b:b+MB]; o_, a_, lpo, ad = O[idx], A[idx], LP[idx], adv[idx]
            z = pi.forward(o_)[:, 0]; mu = np.tanh(z); s_ = np.exp(logstd[0])
            lp = -0.5*((a_ - mu)/s_)**2 - logstd[0] - 0.5*np.log(2*np.pi)
            ratio = np.exp(lp - lpo)
            active = ~(((ad > 0) & (ratio > 1+CLIP)) | ((ad < 0) & (ratio < 1-CLIP)))
            dlp = -(ratio*ad*active)/len(idx)
            dz = dlp*(a_ - mu)/s_**2*(1 - mu**2)
            gls = np.array([(dlp*((a_ - mu)**2/s_**2 - 1)).sum()])
            opt_pi.step(pi.backward(dz[:, None]) + [gls])
            vv = vf.forward(o_)[:, 0]
            opt_vf.step(vf.backward(((vv - ret[idx])/len(idx))[:, None]), max_norm=5.0)
            kl_sum += np.mean(lpo - lp)
        if kl_sum/((n + MB - 1)//MB) > 0.02: break
    logstd[0] = max(logstd[0], np.log(0.02))
    ep_ret = (Rw.sum(0)/RSCALE).mean()
    log['it'].append(it); log['ret'].append(ep_ret); log['iae'].append(iae); log['std'].append(np.exp(logstd[0])); log['val'].append(val)
    if it % 10 == 0:
        print(f"it {it:4d} | episode return {ep_ret:8.2f} | test IAE {iae:6.3f} | val {val:6.3f} | std {np.exp(logstd[0]):.3f} | {time.time()-t0:5.0f}s", flush=True)

for i, p in enumerate(best[1]): pi.P[i][...] = p
yb, ub, iaeb = eval_test(); snaps['best'] = (yb, ub, iaeb, best[2])
print('best (by validation) iteration', best[2], 'val', best[0], 'test IAE', iaeb)
pickle.dump(dict(log=log, snaps=snaps, pi=pi.P), open("data/results.pkl", "wb"))
print("done", time.time() - t0)
