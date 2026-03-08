"""
train.py
--------
Training loop for the Navier-Stokes PINN.

Features
--------
  * Adam optimiser with cosine-annealing learning-rate schedule
  * Early stopping (monitors total loss, configurable patience)
  * Checkpoint saved every `save_every` epochs  →  models/
  * Best model saved whenever validation loss improves
  * Live console output: progress %, ETA, and every loss component
"""

import os
import time
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
from model import PINN, pinn_loss, ns_residuals
from sampling import (sample_collocation, sample_all_boundaries,
                      _outside_cylinder, X_MIN, X_MAX, Y_MIN, Y_MAX)

# ---------------------------------------------------------------------------
# Configuration — edit these to change the experiment
# ---------------------------------------------------------------------------

CFG = {
    # Network architecture  [input, hidden..., output]  (10 hidden layers × 50 neurons)
    "layers"      : [2, 50, 50, 50, 50, 50, 50, 3],

    # Physics
    "Re"          : 1.0,
    "w_pde"       : 1.0,    # weight for PDE residual terms
    "w_bc"        : 10.0,   # weight for BC terms (larger → enforce BCs tighter)

    # Sampling counts
    "n_uniform"   : 8000,  # uniform baseline (replaces old near-cyl + wake buckets)
    "n_inflow"    : 200,
    "n_outflow"   : 200,
    "n_top"       : 150,
    "n_bot"       : 150,
    "n_cyl"       : 300,

    # Training
    "epochs"      : 10000,
    "lr"          : 1e-3,
    "lr_min"      : 4e-6,   # cosine annealing lower bound

    # Early stopping
    "patience"    : 500,   # epochs without improvement before stopping
    "min_delta"   : 1e-7,   # minimum decrease to count as improvement

    # Gradient clipping (set to None to disable)
    "grad_clip"   : 1.0,    # max L2 norm of gradients

    # Residual-based adaptive collocation resampling
    "resample_every"  : 500,    # add new pts every N epochs (0 = disabled)
    "n_residual_add"  : 2000,   # candidate pts to add per resample event
    "n_col_max"       : 15000,  # hard cap on total collocation pts

    # I/O
    "save_every"  : 1000,   # save checkpoint every N epochs
    "print_every" : 100,    # print progress every N epochs
    "models_dir"  : "models",

    # Device  ("cuda" | "cpu" | "mps")
    "device"      : "cuda" if torch.cuda.is_available() else "cpu",

    # Reproducibility
    "seed"        : 42,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.2f}h"


def build_progress_bar(fraction: float, width: int = 25) -> str:
    filled = int(fraction * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {fraction*100:5.1f}%"


def save_checkpoint(model, optimiser, scheduler, epoch, loss_val, path: str):
    torch.save(
        {
            "epoch"        : epoch,
            "loss"         : loss_val,
            "model_state"  : model.state_dict(),
            "optim_state"  : optimiser.state_dict(),
            "sched_state"  : scheduler.state_dict(),
            "cfg"          : CFG,
        },
        path,
    )


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, loss: float) -> bool:
        """Returns True if training should stop."""
        if loss < self.best - self.min_delta:
            self.best    = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# Residual-based adaptive collocation resampling
# ---------------------------------------------------------------------------

def adaptive_resample(model, xy_col, n_add, n_max, Re, device):
    """
    Add new collocation points near high-residual regions.
    Existing points are importance-sampled by total PDE residual magnitude
    and then jittered slightly to produce new candidate points.
    The total point count is capped at n_max.
    """
    model.eval()
    xy_eval = xy_col.detach().requires_grad_(True)
    r_cont, r_xmom, r_ymom = ns_residuals(model, xy_eval, Re)
    residuals = (
        r_cont.detach().abs() + r_xmom.detach().abs() + r_ymom.detach().abs()
    ).squeeze()
    model.train()

    probs = residuals.cpu().numpy().astype(np.float64)
    probs /= probs.sum()
    n_sample = min(n_add, xy_col.shape[0])
    rng = np.random.default_rng()
    idx = rng.choice(len(probs), size=n_sample, p=probs, replace=True)
    anchors = xy_col.detach().cpu().numpy()[idx].astype(np.float32)
    jitter  = rng.normal(0.0, 0.02, anchors.shape).astype(np.float32)
    new_pts = anchors + jitter
    new_pts[:, 0] = np.clip(new_pts[:, 0], X_MIN, X_MAX)
    new_pts[:, 1] = np.clip(new_pts[:, 1], Y_MIN, Y_MAX)
    new_pts = new_pts[_outside_cylinder(new_pts)]

    if len(new_pts) == 0:
        return xy_col

    new_t    = torch.tensor(new_pts, dtype=torch.float32, device=device)
    combined = torch.cat([xy_col.detach(), new_t], dim=0)
    if combined.shape[0] > n_max:
        combined = combined[-n_max:]   # keep freshest (most residual-guided) pts
    return combined.requires_grad_(True)


# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------

def train(cfg: dict = CFG):
    torch.manual_seed(cfg["seed"])
    device = cfg["device"]
    os.makedirs(cfg["models_dir"], exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Navier-Stokes PINN  |  Re={cfg['Re']}  |  device={device}")
    print(f"{'='*65}\n")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = PINN(cfg["layers"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture : {cfg['layers']}")
    print(f"  Parameters   : {total_params:,}")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    print("\n  Sampling collocation and boundary points …")
    xy_col = sample_collocation(
        n_uniform = cfg["n_uniform"],
        seed      = cfg["seed"],
        device    = device,
    )
    bc = sample_all_boundaries(
        n_inflow  = cfg["n_inflow"],
        n_outflow = cfg["n_outflow"],
        n_top     = cfg["n_top"],
        n_bot     = cfg["n_bot"],
        n_cyl     = cfg["n_cyl"],
        device    = device,
    )
    print(f"  Interior pts : {xy_col.shape[0]:,}")
    for k, v in bc.items():
        print(f"  BC {k:10s}: {v.shape[0]:,} pts")

    xy_in  = bc["inflow"]
    xy_out = bc["outflow"]
    xy_top = bc["top"]
    xy_bot = bc["bottom"]
    xy_cyl = bc["cylinder"]

    # ------------------------------------------------------------------
    # Optimiser + LR scheduler
    # ------------------------------------------------------------------
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max  = cfg["epochs"],
        eta_min= cfg["lr_min"],
    )
    early_stop = EarlyStopping(cfg["patience"], cfg["min_delta"])

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_loss   = float("inf")
    epochs_done = 0
    t_start     = time.time()
    t_last_print= t_start
    history     = []          # total loss per epoch
    pde_history = []          # PDE residual loss per epoch
    bc_history  = []          # boundary condition loss per epoch

    print(f"\n  Starting training for up to {cfg['epochs']:,} epochs …\n")
    header = (
        f"{'Epoch':>8}  {'Progress':27}  {'ETA':>8}  "
        f"{'LR':>9}  {'Total':>10}  "
        f"{'PDE cont':>10}  {'PDE xmom':>10}  {'PDE ymom':>10}  "
        f"{'BC inflow':>10}  {'BC outlet':>10}  {'BC walls':>10}  {'BC cyl':>10}"
    )
    print(header)
    print("-" * len(header))

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimiser.zero_grad()

        loss, loss_dict = pinn_loss(
            model,
            xy_col, xy_in, xy_out, xy_top, xy_bot, xy_cyl,
            Re    = cfg["Re"],
            w_pde = cfg["w_pde"],
            w_bc  = cfg["w_bc"],
        )

        loss.backward()
        if cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimiser.step()
        scheduler.step()

        # --- residual-based adaptive resampling ---
        resample_every = cfg.get("resample_every", 0)
        if resample_every and epoch % resample_every == 0:
            xy_col = adaptive_resample(
                model, xy_col,
                n_add  = cfg.get("n_residual_add", 2000),
                n_max  = cfg.get("n_col_max", 15000),
                Re     = cfg["Re"],
                device = device,
            )

        epochs_done = epoch
        total_loss  = loss_dict["total"]

        history.append(total_loss)
        pde_history.append(
            loss_dict["pde/cont"] + loss_dict["pde/xmom"] + loss_dict["pde/ymom"]
        )
        bc_history.append(
            loss_dict["bc/inflow_u_avg"] + loss_dict["bc/inflow_du_dx"] +
            loss_dict["bc/inflow_dv_dx"] +
            loss_dict["bc/outlet_p"]     + loss_dict["bc/outlet_du_dx"] +
            loss_dict["bc/outlet_dv_dx"] +
            loss_dict["bc/top_u"]        + loss_dict["bc/top_v"] +
            loss_dict["bc/bot_u"]        + loss_dict["bc/bot_v"] +
            loss_dict["bc/cyl_u"]        + loss_dict["bc/cyl_v"]
        )

        # --- periodic checkpoint ---
        if epoch % cfg["save_every"] == 0:
            ckpt_path = os.path.join(
                cfg["models_dir"], f"checkpoint_epoch{epoch:06d}.pt"
            )
            save_checkpoint(model, optimiser, scheduler, epoch, total_loss, ckpt_path)

        # --- best-model checkpoint ---
        if total_loss < best_loss:
            best_loss = total_loss
            save_checkpoint(
                model, optimiser, scheduler, epoch, total_loss,
                os.path.join(cfg["models_dir"], "best_model.pt"),
            )

        # --- console output ---
        if epoch % cfg["print_every"] == 0 or epoch == 1:
            now         = time.time()
            elapsed     = now - t_start
            rate        = elapsed / epoch                    # seconds per epoch
            remaining   = rate * (cfg["epochs"] - epoch)
            fraction    = epoch / cfg["epochs"]
            lr_now      = scheduler.get_last_lr()[0]

            bc_walls = (
                loss_dict["bc/top_u"] + loss_dict["bc/top_v"] +
                loss_dict["bc/bot_u"] + loss_dict["bc/bot_v"]
            )
            bc_outlet = (
                loss_dict["bc/outlet_p"] +
                loss_dict["bc/outlet_du_dx"] +
                loss_dict["bc/outlet_dv_dx"]
            )
            bc_inflow = (loss_dict["bc/inflow_u_avg"] +
                         loss_dict["bc/inflow_du_dx"] +
                         loss_dict["bc/inflow_dv_dx"])
            bc_cyl    = loss_dict["bc/cyl_u"] + loss_dict["bc/cyl_v"]

            print(
                f"{epoch:>8d}  {build_progress_bar(fraction)}  "
                f"{format_time(remaining):>8s}  "
                f"{lr_now:>9.2e}  {total_loss:>10.4e}  "
                f"{loss_dict['pde/cont']:>10.4e}  "
                f"{loss_dict['pde/xmom']:>10.4e}  "
                f"{loss_dict['pde/ymom']:>10.4e}  "
                f"{bc_inflow:>10.4e}  "
                f"{bc_outlet:>10.4e}  "
                f"{bc_walls:>10.4e}  "
                f"{bc_cyl:>10.4e}"
            )

        # --- early stopping ---
        if early_stop.step(total_loss):
            print(
                f"\n  [Early stopping]  No improvement for {cfg['patience']} epochs.  "
                f"Best loss = {best_loss:.6e}"
            )
            break

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    t_total = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  Training finished after {epochs_done:,} epochs  ({format_time(t_total)})")
    print(f"  Best total loss   : {best_loss:.6e}")
    print(f"  Best model saved  : {os.path.join(cfg['models_dir'], 'best_model.pt')}")
    print(f"{'='*65}\n")

    # ------------------------------------------------------------------
    # Loss curves
    # ------------------------------------------------------------------
    # Total loss
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(history, linewidth=0.8, color='steelblue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total loss  (log scale)')
    ax.set_title('PINN training loss')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    total_curve_path = os.path.join(cfg['models_dir'], 'loss_curve.png')
    plt.savefig(total_curve_path, dpi=150)
    plt.close(fig)
    print(f'  Loss curve saved : {total_curve_path}')

    # PDE vs BC component curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].semilogy(pde_history, linewidth=0.8, color='steelblue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('PDE loss  (log scale)')
    axes[0].set_title('PDE residual loss')
    axes[0].grid(True, which='both', alpha=0.3)
    axes[1].semilogy(bc_history, linewidth=0.8, color='darkorange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('BC loss  (log scale)')
    axes[1].set_title('Boundary condition loss')
    axes[1].grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    comp_curve_path = os.path.join(cfg['models_dir'], 'loss_curve_components.png')
    plt.savefig(comp_curve_path, dpi=150)
    plt.close(fig)
    print(f'  Component curves saved : {comp_curve_path}')

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    trained_model = train(CFG)
