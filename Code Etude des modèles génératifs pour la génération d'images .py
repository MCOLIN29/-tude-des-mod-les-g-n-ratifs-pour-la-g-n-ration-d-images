import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.linalg import sqrtm


device = "cuda" if torch.cuda.is_available() else "cpu"


def fig_path(name):
    return name


def save_current_figure(path):
    plt.tight_layout()
    plt.show()


D = 2


N_MODES  = 8
RADIUS   = 4.0
MODE_STD = 0.3


N_ARMS           = 2
SPIRAL_THETA_MAX = 4 * np.pi
SPIRAL_A         = 0.5
SPIRAL_NOISE_STD = 0.15


LAMBDA = 1.0


SIGMA_MIN = 0.01


N_STEPS_TRAIN = 4000
BATCH         = 512
N_TRAININGS   = 10
N_TRAININGS_RFM = 3


N_STEPS_TRAIN_SPIRAL = 10000
BATCH_SPIRAL          = 1024


N_GEN            = 2000
GEN_STEPS_FINAL  = 500
NFE_STEPS        = 100
NFE_SWEEP        = [5, 10, 20, 50, 100]
N_GEN_REPEATS    = 10


N_PAIRS_OT      = 8192
SINKHORN_EPS    = 0.05
SINKHORN_ITERS  = 50
N_PAIRS_REFLOW  = 8192
N_REFLOW_ROUNDS = 2


N_T_POINTS      = 60
N_SAMPLES_EVAL  = 1000
FLOOR_FRAC      = 0.05


SEED_TRAIN = 0
SEED_GEN   = 25


LOSS_YLIM      = (1e-4, 1e1)
ABS_ERROR_YLIM = (1e-4, 1e2)
REL_ERROR_YLIM = (1e-3, 1e3)

torch.manual_seed(SEED_TRAIN)
np.random.seed(SEED_TRAIN)


_angles = np.linspace(0, 2 * np.pi, N_MODES, endpoint=False)
GAUSSIAN_CENTERS = np.stack([RADIUS * np.cos(_angles), RADIUS * np.sin(_angles)], axis=1)


def sample_gaussians(n):
    idx = np.random.randint(0, N_MODES, size=n)
    return GAUSSIAN_CENTERS[idx] + np.random.randn(n, D) * MODE_STD


def sample_spiral(n):
    arm   = np.random.randint(0, N_ARMS, size=n)
    u     = np.random.rand(n)
    theta = u * SPIRAL_THETA_MAX
    r     = SPIRAL_A * theta
    phase = arm * (2 * np.pi / N_ARMS)
    pts = np.stack([r * np.cos(theta + phase), r * np.sin(theta + phase)], axis=1)
    pts += np.random.randn(n, D) * SPIRAL_NOISE_STD
    return pts


def sample_noise(n, lam=LAMBDA):
    return np.random.randn(n, D) * np.sqrt(lam)


def estimate_sigma_max(sample_fn, n_probe=2000):
    pts = sample_fn(n_probe)
    return float(np.linalg.norm(pts[:, None] - pts[None, :], axis=-1).max())


def make_geometric_schedule(sigma_min, sigma_max):
    def sigma_fn(t):
        return sigma_min * (sigma_max / sigma_min) ** t

    def dsigma2_dt_fn(t):
        ratio = sigma_max / sigma_min
        return 2.0 * np.log(ratio) * sigma_min ** 2 * ratio ** (2 * t)

    return sigma_fn, dsigma2_dt_fn


SIGMA_MAX = estimate_sigma_max(sample_gaussians)
sigma, dsigma2_dt = make_geometric_schedule(SIGMA_MIN, SIGMA_MAX)
print(f"sigma_min = {SIGMA_MIN},  sigma_max = {SIGMA_MAX:.4f}")


class FieldNet(nn.Module):

    def __init__(self, dim=D, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t.view(-1, 1)], dim=1))


def s_theta(model, x, t, sigma_fn, weighted):
    if weighted:
        sig_t = sigma_fn(t).view(-1, 1)
        return -model(x, t) / sig_t
    return model(x, t)


def train_score_model(seed, sample_data_fn, sigma_fn, weighted,
                       n_steps=N_STEPS_TRAIN, batch=BATCH, dim=D, label=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FieldNet(dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for step in range(n_steps):
        x0 = torch.tensor(sample_data_fn(batch), dtype=torch.float32, device=device)
        t = torch.rand(batch, device=device)
        sig = sigma_fn(t).view(-1, 1)
        z = torch.randn(batch, dim, device=device)
        xt = x0 + sig * z
        pred = model(xt, t)
        if weighted:
            loss = ((pred - z) ** 2).mean()
        else:
            target = -z / sig
            loss = 0.5 * ((pred - target) ** 2).sum(dim=1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"  [{label}] step {step + 1}/{n_steps}  loss={np.mean(losses[-1000:]):.4f}")
    return model, losses


def train_flow_model(seed, sample_x0_fn=None, sample_x1_fn=None,
                      x0_pool=None, x1_pool=None, init_state_dict=None,
                      n_steps=N_STEPS_TRAIN, batch=BATCH, dim=D, label=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FieldNet(dim=dim).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []

    fixed_pool = x0_pool is not None
    n_pairs = x0_pool.shape[0] if fixed_pool else None

    for step in range(n_steps):
        if fixed_pool:
            idx = np.random.randint(0, n_pairs, size=batch)
            x0 = torch.tensor(x0_pool[idx], dtype=torch.float32, device=device)
            x1 = torch.tensor(x1_pool[idx], dtype=torch.float32, device=device)
        else:
            x0 = torch.tensor(sample_x0_fn(batch), dtype=torch.float32, device=device)
            x1 = torch.tensor(sample_x1_fn(batch), dtype=torch.float32, device=device)

        t = torch.rand(batch, device=device)
        xt = (1 - t).view(-1, 1) * x0 + t.view(-1, 1) * x1
        target = x1 - x0

        pred = model(xt, t)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"  [{label}] step {step + 1}/{n_steps}  loss={np.mean(losses[-1000:]):.4f}")
    return model, losses


@torch.no_grad()
def generate_reverse_sde(model, sigma_fn, dsigma2_fn, sigma_max, weighted,
                          n_samples, n_steps=GEN_STEPS_FINAL, dim=D,
                          return_trajectories=False, x_init=None):
    x = x_init.clone() if x_init is not None else torch.randn(n_samples, dim, device=device) * sigma_max
    dt = 1.0 / n_steps
    traj = [x.cpu().numpy().copy()] if return_trajectories else None
    t = 1.0
    for _ in range(n_steps):
        t_batch = torch.full((n_samples,), t, device=device)
        g2 = dsigma2_fn(t)
        score = s_theta(model, x, t_batch, sigma_fn, weighted)
        drift = g2 * score
        diffusion = np.sqrt(max(g2, 0.0))
        x = x + drift * dt + diffusion * torch.randn_like(x) * np.sqrt(dt)
        t -= dt
        if return_trajectories:
            traj.append(x.cpu().numpy().copy())
    if return_trajectories:
        return x, np.stack(traj, axis=0)
    return x


@torch.no_grad()
def generate_probability_flow(model, sigma_fn, dsigma2_fn, sigma_max, weighted,
                               n_samples, n_steps=GEN_STEPS_FINAL, dim=D,
                               return_trajectories=False, x_init=None):
    x = x_init.clone() if x_init is not None else torch.randn(n_samples, dim, device=device) * sigma_max
    dt = 1.0 / n_steps
    traj = [x.cpu().numpy().copy()] if return_trajectories else None
    t = 1.0
    for _ in range(n_steps):
        t_batch = torch.full((n_samples,), t, device=device)
        g2 = dsigma2_fn(t)
        score = s_theta(model, x, t_batch, sigma_fn, weighted)
        x = x + 0.5 * g2 * score * dt
        t -= dt
        if return_trajectories:
            traj.append(x.cpu().numpy().copy())
    if return_trajectories:
        return x, np.stack(traj, axis=0)
    return x


@torch.no_grad()
def integrate_flow_ode(model, x1_init, n_steps=GEN_STEPS_FINAL, return_trajectories=False):
    x = x1_init.clone()
    dt = 1.0 / n_steps
    traj = [x.cpu().numpy().copy()] if return_trajectories else None
    t = 1.0
    for _ in range(n_steps):
        t_batch = torch.full((x.shape[0],), t, device=device)
        v = model(x, t_batch)
        x = x - v * dt
        t -= dt
        if return_trajectories:
            traj.append(x.cpu().numpy().copy())
    if return_trajectories:
        return x, np.stack(traj, axis=0)
    return x


@torch.no_grad()
def generate_flow_matching(model, n_samples, lam=LAMBDA, dim=D,
                            n_steps=GEN_STEPS_FINAL, return_trajectories=False, x1_init=None):
    if x1_init is None:
        x1_init = torch.randn(n_samples, dim, device=device) * np.sqrt(lam)
    return integrate_flow_ode(model, x1_init, n_steps=n_steps,
                               return_trajectories=return_trajectories)


def build_ot_pool_exact(n_pairs, sample_x0_fn, sample_x1_fn):
    x0 = sample_x0_fn(n_pairs)
    x1 = sample_x1_fn(n_pairs)
    C = ((x0[:, None, :] - x1[None, :, :]) ** 2).sum(-1)
    idx0, idx1 = linear_sum_assignment(C)
    return x0[idx0], x1[idx1]


def build_ot_pool_sinkhorn(n_pairs, sample_x0_fn, sample_x1_fn,
                            epsilon=SINKHORN_EPS, n_iters=SINKHORN_ITERS):
    n = n_pairs
    x0 = sample_x0_fn(n)
    x1 = sample_x1_fn(n)
    C = ((x0[:, None, :] - x1[None, :, :]) ** 2).sum(-1)
    Cn = C / C.max()

    K = np.exp(-Cn / epsilon)
    a = np.full(n, 1.0 / n)
    b = np.full(n, 1.0 / n)
    u, v = np.ones(n), np.ones(n)
    for _ in range(n_iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)

    order = np.argsort(C, axis=None)
    used0, used1 = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    idx0, idx1 = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    count = 0
    for flat in order:
        i, j = divmod(flat, n)
        if not used0[i] and not used1[j]:
            used0[i] = used1[j] = True
            idx0[count], idx1[count] = i, j
            count += 1
            if count == n:
                break
    return x0[idx0], x1[idx1]


def run_reflow(seed, n_rounds, sample_x0_fn, lam=LAMBDA, dim=D,
               n_steps=N_STEPS_TRAIN, batch=BATCH, n_pairs=N_PAIRS_REFLOW,
               gen_steps=GEN_STEPS_FINAL, n_gen_snapshot=N_GEN, warm_start=True,
               x1_init_snapshot=None):
    models, losses_all, snapshots = [], [], []

    model_k, losses_k = train_flow_model(
        seed, sample_x0_fn=sample_x0_fn, sample_x1_fn=lambda n: sample_noise(n, lam),
        n_steps=n_steps, batch=batch, dim=dim, label=f"seed {seed}, FM")
    models.append(model_k); losses_all.append(losses_k)
    gen_k, traj_k = generate_flow_matching(model_k, n_gen_snapshot, lam=lam, dim=dim,
                                            n_steps=gen_steps, return_trajectories=True,
                                            x1_init=x1_init_snapshot)
    snapshots.append((gen_k.cpu().numpy(), traj_k))

    g = torch.Generator().manual_seed(seed + 10_000)
    x1_pool = (torch.randn(n_pairs, dim, generator=g) * np.sqrt(lam)).numpy()

    for k in range(1, n_rounds + 1):
        x1_pool_t = torch.tensor(x1_pool, dtype=torch.float32, device=device)
        x0_pool = integrate_flow_ode(model_k, x1_pool_t, n_steps=gen_steps).cpu().numpy()
        init_sd = model_k.state_dict() if warm_start else None
        model_k, losses_k = train_flow_model(
            seed * 100 + k, x0_pool=x0_pool, x1_pool=x1_pool, init_state_dict=init_sd,
            n_steps=n_steps, batch=batch, dim=dim, label=f"seed {seed}, RFM étape {k}")
        models.append(model_k); losses_all.append(losses_k)
        gen_k, traj_k = generate_flow_matching(model_k, n_gen_snapshot, lam=lam, dim=dim,
                                                n_steps=gen_steps, return_trajectories=True,
                                                x1_init=x1_init_snapshot)
        snapshots.append((gen_k.cpu().numpy(), traj_k))
        print(f"  [seed {seed}] round {k}/{n_rounds} terminé")
    return models, losses_all, snapshots


def compute_fid(real, generated):
    mu_r, mu_g = real.mean(axis=0), generated.mean(axis=0)
    S_r, S_g = np.cov(real, rowvar=False), np.cov(generated, rowvar=False)
    diff = mu_r - mu_g
    covmean = sqrtm(S_r @ S_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(S_r + S_g - 2 * covmean))


def fid_curve(gen_fn, real_ref, nfe_sweep=NFE_SWEEP, n_repeats=N_GEN_REPEATS,
              seed_gen=SEED_GEN, n_gen=N_GEN, label=""):
    mean_curve, std_curve = [], []
    for nfe in nfe_sweep:
        fids = []
        for r in range(n_repeats):
            torch.manual_seed(seed_gen * 1000 + r)
            gen = gen_fn(n_gen, nfe).cpu().numpy()
            fids.append(compute_fid(real_ref, gen))
        m, s = float(np.mean(fids)), float(np.std(fids))
        mean_curve.append(m); std_curve.append(s)
        print(f"  [{label}] NFE={nfe:<4} FID={m:.4f} ± {s:.4f}")
    return mean_curve, std_curve


def divergence_exact(f, x):
    n, d = x.shape
    div = torch.zeros(n, device=x.device)
    for i in range(d):
        grad_outputs = torch.zeros_like(f)
        grad_outputs[:, i] = 1.0
        grad_x = torch.autograd.grad(f, x, grad_outputs=grad_outputs,
                                      retain_graph=(i < d - 1), create_graph=False)[0]
        div = div + grad_x[:, i]
    return div


def compute_log_px0(drift_fn, log_prior_fn, x0_batch, n_steps=NFE_STEPS):
    x = x0_batch.clone().detach()
    n = x.shape[0]
    dt = 1.0 / n_steps
    t = 0.0
    log_det = torch.zeros(n, device=x.device)
    for _ in range(n_steps):
        x = x.detach().requires_grad_(True)
        f = drift_fn(x, t)
        div = divergence_exact(f, x)
        log_det = log_det + div.detach() * dt
        x = (x + f.detach() * dt)
        t += dt
    return log_prior_fn(x) + log_det


def compute_mean_nll(drift_fn, log_prior_fn, real_data, n_steps=NFE_STEPS, label=""):
    x0 = torch.tensor(real_data, dtype=torch.float32, device=device)
    nll = -compute_log_px0(drift_fn, log_prior_fn, x0, n_steps=n_steps)
    m, s = nll.mean().item(), nll.std().item()
    print(f"  [{label}] NFE={n_steps:<4} NLL={m:.4f} ± {s:.4f}")
    return m, s


def nll_curve(drift_fn, log_prior_fn, real_data, label, nfe_sweep=NFE_SWEEP):
    mean_curve, std_curve = [], []
    for nfe in nfe_sweep:
        m, s = compute_mean_nll(drift_fn, log_prior_fn, real_data, n_steps=nfe, label=label)
        mean_curve.append(m); std_curve.append(s)
    return mean_curve, std_curve


def make_diffusion_drift(model, sigma_fn, dsigma2_fn, weighted):
    def drift_fn(x, t):
        t_batch = torch.full((x.shape[0],), t, device=x.device)
        g2 = dsigma2_fn(t)
        score = s_theta(model, x, t_batch, sigma_fn, weighted)
        return -0.5 * g2 * score
    return drift_fn


def log_prior_diffusion(x, sigma_max):
    d = x.shape[1]; var = sigma_max ** 2
    return -0.5 * d * np.log(2 * np.pi * var) - (x ** 2).sum(dim=1) / (2 * var)


def make_flow_drift(model):
    def drift_fn(x, t):
        t_batch = torch.full((x.shape[0],), t, device=x.device)
        return model(x, t_batch)
    return drift_fn


def log_prior_flow_matching(x, lam):
    d = x.shape[1]
    return -0.5 * d * np.log(2 * np.pi * lam) - (x ** 2).sum(dim=1) / (2 * lam)


def true_score_gmm(xt, t, sigma_fn, centers=GAUSSIAN_CENTERS, mode_std=MODE_STD):
    sig_t = float(sigma_fn(t))
    var_tot = mode_std ** 2 + sig_t ** 2
    diff = xt[:, None, :] - centers[None, :, :]
    sqdist = (diff ** 2).sum(-1)
    logw = -0.5 * sqdist / var_tot
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw); w /= w.sum(axis=1, keepdims=True)
    return -(w[:, :, None] * diff).sum(axis=1) / var_tot


def true_velocity_gmm(xt, t, centers=GAUSSIAN_CENTERS, mode_std=MODE_STD, lam=LAMBDA):
    var_t = (1 - t) ** 2 * mode_std ** 2 + t ** 2 * lam
    diff = xt[:, None, :] - (1 - t) * centers[None, :, :]
    sqdist = (diff ** 2).sum(-1)
    logw = -0.5 * sqdist / var_t
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw); w /= w.sum(axis=1, keepdims=True)
    coef = (t * lam - (1 - t) * mode_std ** 2) / var_t
    v_per_mode = coef * diff - centers[None, :, :]
    return (w[:, :, None] * v_per_mode).sum(axis=1)


def smooth_losses(losses_2d, window=50):
    kernel = np.ones(window) / window
    return np.stack([np.convolve(l, kernel, mode="valid") for l in losses_2d], axis=0)


def plot_scatter_data_vs_generated(real, generated, path, xlim=(-8, 8)):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(real[:, 0], real[:, 1], s=4, alpha=0.5, color="tab:blue")
    axes[0].set_title("Jeu de données")
    axes[0].set_xlim(*xlim); axes[0].set_ylim(*xlim); axes[0].set_aspect("equal")
    axes[1].scatter(generated[:, 0], generated[:, 1], s=4, alpha=0.5, color="tab:orange")
    axes[1].set_title("Échantillons générés")
    axes[1].set_xlim(*xlim); axes[1].set_ylim(*xlim); axes[1].set_aspect("equal")
    save_current_figure(path)


def plot_trajectories(traj, path, n_show=30, xlim=(-8, 8)):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for i in range(n_show):
        ax.plot(traj[:, i, 0], traj[:, i, 1], lw=0.8, alpha=0.7, color="tab:purple")
    ax.scatter(traj[0, :n_show, 0], traj[0, :n_show, 1], color="gray", s=15, label=r"$\widehat{x}_0$")
    ax.scatter(traj[-1, :n_show, 0], traj[-1, :n_show, 1], color="tab:orange", s=15, label=r"$\widehat{x}_1$")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(*xlim); ax.set_ylim(*xlim); ax.set_aspect("equal")
    save_current_figure(path)


def plot_trajectories_multi(traj_list, labels, path, n_show=30, xlim=(-8, 8)):
    n = len(traj_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for k, (traj, label) in enumerate(zip(traj_list, labels)):
        ax = axes[k]
        for i in range(n_show):
            ax.plot(traj[:, i, 0], traj[:, i, 1], lw=0.8, alpha=0.7, color="tab:purple")
        ax.scatter(traj[0, :n_show, 0], traj[0, :n_show, 1], color="gray", s=15, label=r"$\widehat{x}_0$")
        ax.scatter(traj[-1, :n_show, 0], traj[-1, :n_show, 1], color="tab:orange", s=15, label=r"$\widehat{x}_1$")
        ax.set_title(label)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(*xlim); ax.set_ylim(*xlim); ax.set_aspect("equal")
    save_current_figure(path)


def plot_loss_single(all_losses, path, ylabel=r"$\mathcal{L}(\theta)$", window=50,
                      ylim=LOSS_YLIM, log_scale=True):
    smoothed = smooth_losses(np.array(all_losses), window)
    mean, std = smoothed.mean(axis=0), smoothed.std(axis=0)
    x = np.arange(len(mean))
    plt.figure(figsize=(6, 4))
    plt.plot(x, mean, color="tab:blue", lw=1.8, label="moyenne")
    plt.fill_between(x, mean - std, mean + std, color="tab:blue", alpha=0.25, label="écart-type")
    plt.xlabel("itération"); plt.ylabel(ylabel)
    if log_scale:
        plt.yscale("log")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, which="both", ls="--", alpha=0.4); plt.legend()
    save_current_figure(path)


def plot_loss_multi(losses_by_group, labels, path, ylabel=r"$\mathcal{L}(\theta)$",
                     window=50, ylim=LOSS_YLIM):
    n = len(losses_by_group)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for k in range(n):
        smoothed = smooth_losses(np.array(losses_by_group[k]), window)
        mean, std = smoothed.mean(axis=0), smoothed.std(axis=0)
        x = np.arange(len(mean))
        ax = axes[k]
        ax.plot(x, mean, color="tab:blue", lw=1.8, label="moyenne")
        ax.fill_between(x, mean - std, mean + std, color="tab:blue", alpha=0.25, label="écart-type")
        ax.set_xlabel("itération"); ax.set_ylabel(ylabel)
        ax.set_yscale("log"); ax.set_ylim(*ylim)
        ax.set_title(labels[k])
        ax.grid(True, which="both", ls="--", alpha=0.4); ax.legend()
    save_current_figure(path)


def plot_error_two_panel(x_grid, mse_mean, mse_std, rel_mean, rel_std, xlabel, path,
                          xscale="log", abs_ylim=ABS_ERROR_YLIM, rel_ylim=REL_ERROR_YLIM,
                          abs_ylabel=r"$\mathbb{E}\,\|\cdot\|^2$", rel_log_y=True,
                          rel_ylabel="MSE / référence (%)"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    lower = np.maximum(mse_mean - mse_std, 1e-12)
    ax.plot(x_grid, mse_mean, color="tab:red", lw=2, label="moyenne")
    ax.fill_between(x_grid, lower, mse_mean + mse_std, color="tab:red", alpha=0.25, label="écart-type")
    if xscale == "log":
        ax.set_xscale("log")
    ax.set_yscale("log")
    if abs_ylim is not None:
        ax.set_ylim(*abs_ylim)
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(abs_ylabel, fontsize=11)
    ax.set_title("Erreur absolue MSE")
    ax.grid(True, which="both", ls="--", alpha=0.4); ax.legend(fontsize=9)

    ax = axes[1]
    floor = 1e-6 if rel_log_y else 0.0
    rel_lower = np.maximum((rel_mean - rel_std) * 100, floor)
    ax.plot(x_grid, rel_mean * 100, color="tab:blue", lw=2, label="moyenne")
    ax.fill_between(x_grid, rel_lower, (rel_mean + rel_std) * 100, color="tab:blue", alpha=0.25, label="écart-type")
    if xscale == "log":
        ax.set_xscale("log")
    if rel_log_y:
        ax.set_yscale("log")
    if rel_ylim is not None:
        ax.set_ylim(*rel_ylim)
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(rel_ylabel, fontsize=11)
    ax.set_title("Erreur relative (%)")
    ax.grid(True, which="both", ls="--", alpha=0.4); ax.legend(fontsize=9)
    save_current_figure(path)


def plot_error_relative_masked(x_grid, rel_mean, rel_std, low_norm_mask, xlabel, path,
                                xscale="linear", rel_ylim=REL_ERROR_YLIM,
                                rel_ylabel="MSE / référence (%)"):
    rel_mean_plot = np.where(low_norm_mask, np.nan, rel_mean)
    rel_std_plot = np.where(low_norm_mask, np.nan, rel_std)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    rel_lower = np.maximum((rel_mean_plot - rel_std_plot) * 100, 1e-6)
    ax.plot(x_grid, rel_mean_plot * 100, color="tab:blue", lw=2, label="moyenne")
    ax.fill_between(x_grid, rel_lower, (rel_mean_plot + rel_std_plot) * 100,
                     color="tab:blue", alpha=0.25, label="écart-type")
    if low_norm_mask.any():
        ax.axvspan(x_grid[low_norm_mask].min(), x_grid[low_norm_mask].max(),
                   color="gray", alpha=0.2, label="référence proche de 0\n(courbe non fiable)")
    if xscale == "log":
        ax.set_xscale("log")
    ax.set_yscale("log"); ax.set_ylim(*rel_ylim)
    ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(rel_ylabel, fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.4); ax.legend(fontsize=9)
    save_current_figure(path)


def plot_error_abs_multi(x_grid, mse_mean_list, mse_std_list, labels, path,
                          abs_ylim=ABS_ERROR_YLIM, xlabel=r"$t$",
                          ylabel=r"$\mathbb{E}\,\|v_\theta - v_t\|^2$"):
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for k in range(n):
        ax = axes[k]
        lower = np.maximum(mse_mean_list[k] - mse_std_list[k], 1e-12)
        ax.plot(x_grid, mse_mean_list[k], color="tab:red", lw=2, label="moyenne")
        ax.fill_between(x_grid, lower, mse_mean_list[k] + mse_std_list[k],
                         color="tab:red", alpha=0.25, label="écart-type")
        ax.set_yscale("log"); ax.set_ylim(*abs_ylim)
        ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(labels[k])
        ax.grid(True, which="both", ls="--", alpha=0.4); ax.legend(fontsize=9)
    save_current_figure(path)


def plot_error_rel_multi(x_grid, rel_mean_list, rel_std_list, low_norm_masks, labels, path,
                          rel_ylim=REL_ERROR_YLIM, xlabel=r"$t$",
                          rel_ylabel="MSE / référence (%)"):
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for k in range(n):
        ax = axes[k]
        mask = low_norm_masks[k]
        rel_mean_plot = np.where(mask, np.nan, rel_mean_list[k])
        rel_std_plot = np.where(mask, np.nan, rel_std_list[k])
        rel_lower = np.maximum((rel_mean_plot - rel_std_plot) * 100, 1e-6)
        ax.plot(x_grid, rel_mean_plot * 100, color="tab:blue", lw=2, label="moyenne")
        ax.fill_between(x_grid, rel_lower, (rel_mean_plot + rel_std_plot) * 100,
                         color="tab:blue", alpha=0.25, label="écart-type")
        if mask.any():
            ax.axvspan(x_grid[mask].min(), x_grid[mask].max(), color="gray", alpha=0.2,
                       label="référence proche de 0\n(courbe non fiable)")
        ax.set_yscale("log"); ax.set_ylim(*rel_ylim)
        ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(rel_ylabel, fontsize=11)
        ax.set_title(labels[k])
        ax.grid(True, which="both", ls="--", alpha=0.4); ax.legend(fontsize=9)
    save_current_figure(path)


METHOD_STYLES = {
    "SGM":               dict(color="tab:red",    marker="o", ls="-"),
    "Probability Flow":  dict(color="tab:orange", marker="s", ls="-"),
    "FM":                dict(color="tab:blue",   marker="^", ls="-"),
    "OFM":               dict(color="tab:green",  marker="v", ls="-"),
    "RFM, étape 2":      dict(color="tab:purple", marker="D", ls="-"),
}


def plot_metric_vs_nfe(nfe_sweep, curves, path, ylabel, log_y=True, ylim=None):
    plt.figure(figsize=(8, 5.5))
    for name, (mean_c, std_c) in curves.items():
        mean_c, std_c = np.array(mean_c), np.array(std_c)
        lower = np.maximum(mean_c - std_c, 1e-6) if log_y else mean_c - std_c
        plt.plot(nfe_sweep, mean_c, label=name, lw=1.8, markersize=5, **METHOD_STYLES[name])
        plt.fill_between(nfe_sweep, lower, mean_c + std_c, color=METHOD_STYLES[name]["color"], alpha=0.2)
    plt.xscale("log")
    if log_y:
        plt.yscale("log")
    if ylim:
        plt.ylim(*ylim)
    plt.xlabel("NFE"); plt.ylabel(ylabel)
    plt.grid(True, which="both", ls="--", alpha=0.4); plt.legend(fontsize=9)
    save_current_figure(path)


def main():
    t_grid = np.linspace(0.01, 0.99, N_T_POINTS)


    print("\n=== SGM non pondéré (l(θ)) ===")
    unweighted_models, unweighted_losses = [], []
    for run in range(N_TRAININGS):
        m, l = train_score_model(run, sample_gaussians, sigma, weighted=False,
                                  label=f"SGM non pondéré {run + 1}/{N_TRAININGS}")
        unweighted_models.append(m); unweighted_losses.append(l)
    score_model_uw = unweighted_models[0]


    gen_uw = generate_reverse_sde(score_model_uw, sigma, dsigma2_dt, SIGMA_MAX, weighted=False,
                                   n_samples=N_GEN, n_steps=GEN_STEPS_FINAL).cpu().numpy()
    real_gauss = sample_gaussians(N_GEN)
    plot_scatter_data_vs_generated(real_gauss, gen_uw, fig_path("fig01_sgm_non_pondere.png"))


    sigma_grid = sigma(t_grid)
    mse_runs, norm_runs = [], []
    for m in unweighted_models:
        m.eval()
        mse_list, norm_list = [], []
        with torch.no_grad():
            for t_val in t_grid:
                x0 = sample_gaussians(N_SAMPLES_EVAL)
                z = np.random.randn(N_SAMPLES_EVAL, D)
                xt = x0 + float(sigma(t_val)) * z
                s_true = true_score_gmm(xt, t_val, sigma)
                xt_th = torch.tensor(xt, dtype=torch.float32, device=device)
                t_th = torch.full((N_SAMPLES_EVAL,), float(t_val), device=device)
                s_pred = s_theta(m, xt_th, t_th, sigma, weighted=False).cpu().numpy()
                mse_list.append(((s_pred - s_true) ** 2).sum(axis=1).mean())
                norm_list.append((s_true ** 2).sum(axis=1).mean())
        mse_runs.append(mse_list); norm_runs.append(norm_list)
    mse_runs, norm_runs = np.array(mse_runs), np.array(norm_runs)
    rel_runs = mse_runs / np.maximum(norm_runs, 1e-8)
    plot_error_two_panel(sigma_grid, mse_runs.mean(0), mse_runs.std(0),
                          rel_runs.mean(0), rel_runs.std(0), r"$\sigma(t)$",
                          fig_path("fig02_sgm_erreur_score_non_pondere.png"),
                          abs_ylabel=r"$\mathbb{E}\,\|s_\theta - \nabla_x \log p_t\|^2$",
                          abs_ylim=None, rel_ylim=None, rel_log_y=False,
                          rel_ylabel=r"MSE / $\nabla_x \log p_t$ (%)")


    plot_loss_single(unweighted_losses, fig_path("fig03_sgm_loss_non_pondere.png"),
                      ylabel=r"$l(\theta)$", ylim=None, log_scale=False)


    print("\n=== SGM pondéré / Probability Flow (L(θ)) ===")
    weighted_models, weighted_losses = [], []
    for run in range(N_TRAININGS):
        m, l = train_score_model(run, sample_gaussians, sigma, weighted=True,
                                  label=f"SGM pondéré {run + 1}/{N_TRAININGS}")
        weighted_models.append(m); weighted_losses.append(l)
    score_model = weighted_models[0]


    plot_loss_single(weighted_losses, fig_path("fig04_sgm_loss_pondere.png"), ylabel=r"$L(\theta)$")


    mse_runs, norm_runs = [], []
    for m in weighted_models:
        m.eval()
        mse_list, norm_list = [], []
        with torch.no_grad():
            for t_val in t_grid:
                x0 = sample_gaussians(N_SAMPLES_EVAL)
                z = np.random.randn(N_SAMPLES_EVAL, D)
                xt = x0 + float(sigma(t_val)) * z
                s_true = true_score_gmm(xt, t_val, sigma)
                xt_th = torch.tensor(xt, dtype=torch.float32, device=device)
                t_th = torch.full((N_SAMPLES_EVAL,), float(t_val), device=device)
                s_pred = s_theta(m, xt_th, t_th, sigma, weighted=True).cpu().numpy()
                mse_list.append(((s_pred - s_true) ** 2).sum(axis=1).mean())
                norm_list.append((s_true ** 2).sum(axis=1).mean())
        mse_runs.append(mse_list); norm_runs.append(norm_list)
    mse_runs, norm_runs = np.array(mse_runs), np.array(norm_runs)
    rel_runs = mse_runs / np.maximum(norm_runs, 1e-8)
    plot_error_two_panel(sigma_grid, mse_runs.mean(0), mse_runs.std(0),
                          rel_runs.mean(0), rel_runs.std(0), r"$\sigma(t)$",
                          fig_path("fig05_sgm_erreur_score_pondere.png"),
                          abs_ylabel=r"$\mathbb{E}\,\|s_\theta - \nabla_x \log p_t\|^2$",
                          rel_ylabel=r"MSE / $\nabla_x \log p_t$ (%)")


    x_init_sgm = torch.randn(N_GEN, D, device=device) * SIGMA_MAX

    gen_w, traj_sde = generate_reverse_sde(score_model, sigma, dsigma2_dt, SIGMA_MAX, weighted=True,
                                            n_samples=N_GEN, n_steps=GEN_STEPS_FINAL,
                                            return_trajectories=True, x_init=x_init_sgm)
    gen_w = gen_w.cpu().numpy()
    plot_scatter_data_vs_generated(real_gauss, gen_w, fig_path("fig06_sgm_generation_gaussiennes.png"))
    plot_trajectories(traj_sde, fig_path("fig08_sgm_trajectoires_eds_reciproque.png"))


    _, traj_pf = generate_probability_flow(score_model, sigma, dsigma2_dt, SIGMA_MAX, weighted=True,
                                            n_samples=N_GEN, n_steps=GEN_STEPS_FINAL,
                                            return_trajectories=True, x_init=x_init_sgm)
    plot_trajectories(traj_pf, fig_path("fig09_probability_flow_trajectoires.png"))


    print("\n=== SGM pondéré sur une spirale ===")
    spiral_model, _ = train_score_model(SEED_TRAIN, sample_spiral, sigma, weighted=True,
                                         n_steps=N_STEPS_TRAIN_SPIRAL, batch=BATCH_SPIRAL,
                                         label="SGM pondéré, spirale")
    gen_spiral = generate_reverse_sde(spiral_model, sigma, dsigma2_dt, SIGMA_MAX, weighted=True,
                                       n_samples=N_GEN, n_steps=GEN_STEPS_FINAL).cpu().numpy()
    real_spiral = sample_spiral(N_GEN)
    plot_scatter_data_vs_generated(real_spiral, gen_spiral, fig_path("fig07_sgm_generation_spirale.png"))


    print("\n=== Flow Matching, couplage décorrélé ===")
    fm_models, fm_losses = [], []
    for run in range(N_TRAININGS):
        m, l = train_flow_model(run, sample_x0_fn=sample_gaussians, sample_x1_fn=sample_noise,
                                 label=f"FM {run + 1}/{N_TRAININGS}")
        fm_models.append(m); fm_losses.append(l)
    fm_model = fm_models[0]


    mse_runs, norm_runs = [], []
    for m in fm_models:
        m.eval()
        mse_list, norm_list = [], []
        with torch.no_grad():
            for t_val in t_grid:
                x0 = sample_gaussians(N_SAMPLES_EVAL)
                x1 = sample_noise(N_SAMPLES_EVAL)
                xt = (1 - t_val) * x0 + t_val * x1
                v_true = true_velocity_gmm(xt, t_val)
                xt_th = torch.tensor(xt, dtype=torch.float32, device=device)
                t_th = torch.full((N_SAMPLES_EVAL,), float(t_val), device=device)
                v_pred = m(xt_th, t_th).cpu().numpy()
                mse_list.append(((v_pred - v_true) ** 2).sum(axis=1).mean())
                norm_list.append((v_true ** 2).sum(axis=1).mean())
        mse_runs.append(mse_list); norm_runs.append(norm_list)
    mse_runs, norm_runs = np.array(mse_runs), np.array(norm_runs)
    rel_runs = mse_runs / np.maximum(norm_runs, 1e-8)
    mse_mean, mse_std = mse_runs.mean(0), mse_runs.std(0)
    norm_mean = norm_runs.mean(0)
    rel_mean, rel_std = rel_runs.mean(0), rel_runs.std(0)
    low_norm_mask_fm = norm_mean < FLOOR_FRAC * norm_mean.max()

    plot_error_two_panel(t_grid, mse_mean, mse_std, rel_mean, rel_std, r"$t$",
                          fig_path("fig10_fm_erreur_vitesse.png"), xscale="linear",
                          abs_ylabel=r"$\mathbb{E}\,\|v_\theta - v_t\|^2$",
                          rel_ylabel=r"MSE / $v_t$ (%)")
    plot_error_relative_masked(t_grid, rel_mean, rel_std, low_norm_mask_fm, r"$t$",
                                fig_path("fig11_fm_erreur_vitesse_ajustee.png"),
                                rel_ylabel=r"MSE / $v_t$ (%)")


    plot_loss_single(fm_losses, fig_path("fig12_fm_loss.png"))


    x1_init_traj = torch.randn(N_GEN, D, device=device) * np.sqrt(LAMBDA)

    _, traj_fm = integrate_flow_ode(fm_model, x1_init_traj, n_steps=GEN_STEPS_FINAL, return_trajectories=True)
    plot_trajectories(traj_fm, fig_path("fig13_fm_trajectoires.png"))


    print("\n=== OFM, couplage approché ===")
    x0_pool_sk, x1_pool_sk = build_ot_pool_sinkhorn(N_PAIRS_OT, sample_gaussians, sample_noise)
    ofm_sinkhorn_model, _ = train_flow_model(SEED_TRAIN, x0_pool=x0_pool_sk, x1_pool=x1_pool_sk,
                                              label="OFM (Sinkhorn)")
    _, traj_ofm_sk = integrate_flow_ode(ofm_sinkhorn_model, x1_init_traj, n_steps=GEN_STEPS_FINAL,
                                         return_trajectories=True)
    plot_trajectories(traj_ofm_sk, fig_path("fig14_ofm_trajectoires_sinkhorn.png"))


    print("\n=== OFM, couplage exact ===")
    ofm_models, ofm_losses, ofm_pools = [], [], []
    for run in range(N_TRAININGS):
        x0_pool, x1_pool = build_ot_pool_exact(N_PAIRS_OT, sample_gaussians, sample_noise)
        m, l = train_flow_model(run, x0_pool=x0_pool, x1_pool=x1_pool,
                                 label=f"OFM exact {run + 1}/{N_TRAININGS}")
        ofm_models.append(m); ofm_losses.append(l); ofm_pools.append((x0_pool, x1_pool))
    ofm_model = ofm_models[0]


    _, traj_ofm = integrate_flow_ode(ofm_model, x1_init_traj, n_steps=GEN_STEPS_FINAL, return_trajectories=True)
    plot_trajectories(traj_ofm, fig_path("fig15_ofm_trajectoires_exact.png"))


    plot_loss_single(ofm_losses, fig_path("fig16_ofm_loss_exact.png"))


    mse_runs, norm_runs = [], []
    for m, (x0_pool, x1_pool) in zip(ofm_models, ofm_pools):
        m.eval()
        idx = np.random.randint(0, x0_pool.shape[0], size=N_SAMPLES_EVAL)
        x0_eval, x1_eval = x0_pool[idx], x1_pool[idx]
        v_true = x1_eval - x0_eval
        mse_list, norm_list = [], []
        with torch.no_grad():
            for t_val in t_grid:
                xt = (1 - t_val) * x0_eval + t_val * x1_eval
                xt_th = torch.tensor(xt, dtype=torch.float32, device=device)
                t_th = torch.full((N_SAMPLES_EVAL,), float(t_val), device=device)
                v_pred = m(xt_th, t_th).cpu().numpy()
                mse_list.append(((v_pred - v_true) ** 2).sum(axis=1).mean())
                norm_list.append((v_true ** 2).sum(axis=1).mean())
        mse_runs.append(mse_list); norm_runs.append(norm_list)
    mse_runs, norm_runs = np.array(mse_runs), np.array(norm_runs)
    rel_runs = mse_runs / np.maximum(norm_runs, 1e-8)
    plot_error_two_panel(t_grid, mse_runs.mean(0), mse_runs.std(0), rel_runs.mean(0), rel_runs.std(0),
                          r"$t$", fig_path("fig17_ofm_erreur_vitesse_exact.png"), xscale="linear",
                          abs_ylabel=r"$\mathbb{E}\,\|v_\theta - v_t\|^2$",
                          rel_ylabel=r"MSE / $v_t$ (%)")


    print("\n=== Rectified Flow Matching ===")
    method_labels = ["FM"] + [f"RFM, étape {k}" for k in range(1, N_REFLOW_ROUNDS + 1)]
    models_by_round = [[] for _ in range(N_REFLOW_ROUNDS + 1)]
    losses_by_round = [[] for _ in range(N_REFLOW_ROUNDS + 1)]
    first_run_snapshots = None
    for run in range(N_TRAININGS_RFM):
        models_run, losses_run, snapshots_run = run_reflow(run, N_REFLOW_ROUNDS, sample_gaussians,
                                                             x1_init_snapshot=x1_init_traj)
        for k in range(N_REFLOW_ROUNDS + 1):
            models_by_round[k].append(models_run[k])
            losses_by_round[k].append(losses_run[k])
        if run == 0:
            first_run_snapshots = snapshots_run
    rfm_model_final = models_by_round[N_REFLOW_ROUNDS][0]


    plot_trajectories_multi([snap[1] for snap in first_run_snapshots], method_labels,
                             fig_path("fig18_rfm_trajectoires_comparaison.png"))


    plot_loss_multi(losses_by_round, method_labels, fig_path("fig19_rfm_loss_comparaison.png"))


    rng_eval = np.random.RandomState(20260724)
    x0_eval_fm = sample_gaussians(N_SAMPLES_EVAL)
    x1_eval_fm = sample_noise(N_SAMPLES_EVAL)
    x1_eval_reflow = rng_eval.randn(N_SAMPLES_EVAL, D) * np.sqrt(LAMBDA)
    x1_eval_reflow_th = torch.tensor(x1_eval_reflow, dtype=torch.float32, device=device)

    mse_mean_k, mse_std_k, rel_mean_k, rel_std_k, low_norm_mask_k = [], [], [], [], []
    for k in range(N_REFLOW_ROUNDS + 1):
        mse_runs, norm_runs = [], []
        for run in range(N_TRAININGS_RFM):
            model_k = models_by_round[k][run]
            model_k.eval()
            if k == 0:
                x0_k, x1_k = x0_eval_fm, x1_eval_fm
            else:
                model_prev = models_by_round[k - 1][run]
                model_prev.eval()
                with torch.no_grad():
                    x0_k = integrate_flow_ode(model_prev, x1_eval_reflow_th,
                                               n_steps=GEN_STEPS_FINAL).cpu().numpy()
                x1_k = x1_eval_reflow

            mse_list, norm_list = [], []
            with torch.no_grad():
                for t_val in t_grid:
                    xt = (1 - t_val) * x0_k + t_val * x1_k
                    v_true = true_velocity_gmm(xt, t_val) if k == 0 else (x1_k - x0_k)
                    xt_th = torch.tensor(xt, dtype=torch.float32, device=device)
                    t_th = torch.full((N_SAMPLES_EVAL,), float(t_val), device=device)
                    v_pred = model_k(xt_th, t_th).cpu().numpy()
                    mse_list.append(((v_pred - v_true) ** 2).sum(axis=1).mean())
                    norm_list.append((v_true ** 2).sum(axis=1).mean())
            mse_runs.append(mse_list); norm_runs.append(norm_list)
        mse_runs, norm_runs = np.array(mse_runs), np.array(norm_runs)
        rel_runs = mse_runs / np.maximum(norm_runs, 1e-8)
        mse_mean_k.append(mse_runs.mean(0)); mse_std_k.append(mse_runs.std(0))
        rel_mean_k.append(rel_runs.mean(0)); rel_std_k.append(rel_runs.std(0))
        low_norm_mask_k.append(norm_runs.mean(0) < FLOOR_FRAC * norm_runs.mean(0).max())


    plot_error_abs_multi(t_grid, mse_mean_k, mse_std_k, method_labels,
                          fig_path("fig20_rfm_erreur_absolue_comparaison.png"))

    plot_error_rel_multi(t_grid, rel_mean_k, rel_std_k, low_norm_mask_k, method_labels,
                          fig_path("fig21_rfm_erreur_relative_comparaison.png"),
                          rel_ylabel=r"MSE / $v_t$ (%)")


    print("\n=== Comparaison des modèles : NLL et FID en fonction du NFE ===")
    real_ref = sample_gaussians(N_GEN)

    gen_fns = {
        "SGM": lambda n, nfe: generate_reverse_sde(score_model, sigma, dsigma2_dt, SIGMA_MAX,
                                                     True, n, n_steps=nfe),
        "Probability Flow": lambda n, nfe: generate_probability_flow(score_model, sigma, dsigma2_dt,
                                                                        SIGMA_MAX, True, n, n_steps=nfe),
        "FM":  lambda n, nfe: generate_flow_matching(fm_model, n, n_steps=nfe),
        "OFM": lambda n, nfe: generate_flow_matching(ofm_model, n, n_steps=nfe),
        "RFM, étape 2": lambda n, nfe: generate_flow_matching(rfm_model_final, n, n_steps=nfe),
    }
    fid_sweep = {name: fid_curve(fn, real_ref, label=name) for name, fn in gen_fns.items()}
    plot_metric_vs_nfe(NFE_SWEEP, fid_sweep, fig_path("fig23_fid_vs_nfe.png"), "FID")

    drift_diff = make_diffusion_drift(score_model, sigma, dsigma2_dt, weighted=True)
    drift_fm = make_flow_drift(fm_model)
    drift_ofm = make_flow_drift(ofm_model)
    drift_rfm = make_flow_drift(rfm_model_final)

    nll_sweep = {
        "SGM": nll_curve(drift_diff, lambda x: log_prior_diffusion(x, SIGMA_MAX), real_ref, "SGM"),
        "Probability Flow": nll_curve(drift_diff, lambda x: log_prior_diffusion(x, SIGMA_MAX),
                                       real_ref, "Probability Flow"),
        "FM":  nll_curve(drift_fm, lambda x: log_prior_flow_matching(x, LAMBDA), real_ref, "FM"),
        "OFM": nll_curve(drift_ofm, lambda x: log_prior_flow_matching(x, LAMBDA), real_ref, "OFM"),
        "RFM, étape 2": nll_curve(drift_rfm, lambda x: log_prior_flow_matching(x, LAMBDA),
                                   real_ref, "RFM"),
    }
    plot_metric_vs_nfe(NFE_SWEEP, nll_sweep, fig_path("fig22_nll_vs_nfe.png"), "NLL", log_y=False)


    print(f"\n{'Modèle':<20}{'NLL':>10}{'FID':>12}")
    print("-" * 42)
    for name in gen_fns:
        print(f"{name:<20}{nll_sweep[name][0][-1]:>10.4f}{fid_sweep[name][0][-1]:>12.4f}")


if __name__ == "__main__":
    main()