"""
sampling.py
-----------
Collocation and boundary-point generation for the 2-D steady Navier-Stokes
PINN on a [0,1.2] x [-0.5, 0.5] channel with a circular cylinder of radius 0.08
centred at (0.3, 0).

Interior sampling strategy:
- Uniform random sampling across the entire domain, excluding the cylinder.
- Adaptive resampling based on PDE residuals: periodically add points in regions where the current model has high errors.
All points inside the solid cylinder (r <= R_cyl) are rejected.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle

# ---------------------------------------------------------------------------
# Problem geometry constants
# ---------------------------------------------------------------------------
X_MIN, X_MAX = 0.0, 1.2
Y_MIN, Y_MAX = -0.5, 0.5
CYL_X, CYL_Y, CYL_R = 0.3, 0.0, 0.08   # cylinder centre and radius


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _outside_cylinder(pts: np.ndarray, safety: float = 1e-3) -> np.ndarray:
    """Boolean mask: True where point is strictly outside the cylinder."""
    r = np.sqrt((pts[:, 0] - CYL_X) ** 2 + (pts[:, 1] - CYL_Y) ** 2)
    return r > CYL_R + safety


def _sample_uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n uniform random points from the bounding box; reject inside cyl."""
    pts = []
    while sum(len(p) for p in pts) < n:
        batch = rng.uniform(
            [X_MIN, Y_MIN], [X_MAX, Y_MAX],
            size=(n * 3, 2),          # oversample to compensate for rejections
        )
        batch = batch[_outside_cylinder(batch)]
        pts.append(batch)
    return np.concatenate(pts, axis=0)[:n]


# ---------------------------------------------------------------------------
# Main sampling function
# ---------------------------------------------------------------------------

def sample_collocation(
    n_uniform: int = 30000,
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate uniform interior collocation points (cylinder excluded).

    Parameters
    ----------
    n_uniform : number of uniform random points
    seed      : random seed for reproducibility
    device    : torch device string

    Returns
    -------
    xy_col : Tensor[n_uniform, 2] with requires_grad=True, on `device`
    """
    rng = np.random.default_rng(seed)
    pts = _sample_uniform(n_uniform, rng).astype(np.float32)
    return torch.tensor(pts, dtype=torch.float32, device=device, requires_grad=True)


def adaptive_resample(
    model: nn.Module,
    xy_col: torch.Tensor,
    cfg: dict,
    device: str,
    epoch: int,
) -> torch.Tensor:
    """
    Residual-guided adaptive resampling.

    Evaluates the PDE residuals on a fresh candidate pool and appends the
    ``cfg['resample_add']`` highest-error points to the current collocation
    set, capping the total at ``cfg['resample_max']``.

    Parameters
    ----------
    model  : current PINN (any nn.Module that outputs [u, v, p])
    xy_col : current collocation tensor  [N, 2]  (requires_grad=True)
    cfg    : config dict with keys: resample_add, resample_max, seed, rho, mu
    device : torch device string
    epoch  : current training epoch (used to vary the candidate pool)

    Returns
    -------
    xy_col : updated collocation tensor  [N', 2]  (requires_grad=True)
             If already at cap, the original tensor is returned unchanged.
    """
    # Lazy import to avoid circular dependency (model imports sampling)
    from model import ns_residuals

    n_current = xy_col.shape[0]
    if n_current >= cfg['resample_max']:
        return xy_col

    n_add = min(cfg['resample_add'], cfg['resample_max'] - n_current)

    # Draw a fresh candidate pool (5× oversampled for coverage)
    rng      = np.random.default_rng(cfg['seed'] + epoch)
    cands_np = _sample_uniform(n_add * 5, rng).astype(np.float32)
    xy_cand  = torch.tensor(cands_np, dtype=torch.float32,
                             device=device, requires_grad=True)

    # Evaluate PDE residuals at candidates
    was_training = model.training
    model.eval()
    r_cont, r_xmom, r_ymom = ns_residuals(model, xy_cand,
                                           rho=cfg['rho'], mu=cfg['mu'])
    res_mag = (
        r_cont.detach() ** 2 +
        r_xmom.detach() ** 2 +
        r_ymom.detach() ** 2
    ).squeeze()
    if was_training:
        model.train()

    # Select the top-n_add highest-residual candidates
    _, idx   = torch.topk(res_mag, n_add)
    selected = cands_np[idx.cpu().numpy()]

    # Merge with existing points and return a fresh requires_grad tensor
    merged = np.concatenate(
        [xy_col.detach().cpu().numpy(), selected], axis=0
    )
    return torch.tensor(merged, dtype=torch.float32,
                         device=device, requires_grad=True)


# ---------------------------------------------------------------------------
# Boundary-condition point generators
# ---------------------------------------------------------------------------

def sample_inflow(n: int = 200, seed: int = 0, device: str = "cpu") -> torch.Tensor:
    """
    Inflow boundary:  x = 0,  y ~ uniform in [Y_MIN, Y_MAX].
    BC: u = 1, v = 0.
    """
    rng = np.random.default_rng(seed)
    y  = rng.uniform(Y_MIN, Y_MAX, size=(n, 1)).astype(np.float32)
    x  = np.zeros_like(y)
    return torch.tensor(np.hstack([x, y]), dtype=torch.float32, device=device)


def sample_outflow(n: int = 200, seed: int = 1, device: str = "cpu") -> torch.Tensor:
    """
    Outflow boundary: x = X_MAX (1.2),  y ~ uniform in [Y_MIN, Y_MAX].
    BC enforced in loss: p_avg = 0,  v = 0  (u·t = 0).
    """
    rng = np.random.default_rng(seed)
    y  = rng.uniform(Y_MIN, Y_MAX, size=(n, 1)).astype(np.float32)
    x  = np.full_like(y, X_MAX)
    return torch.tensor(np.hstack([x, y]), dtype=torch.float32, device=device)


def sample_top_wall(n: int = 150, seed: int = 2, device: str = "cpu") -> torch.Tensor:
    """Top wall: y = 0.5, x ~ uniform. BC: u = 0, v = 0."""
    rng = np.random.default_rng(seed)
    x  = rng.uniform(X_MIN, X_MAX, size=(n, 1)).astype(np.float32)
    y  = np.full_like(x, Y_MAX)
    return torch.tensor(np.hstack([x, y]), dtype=torch.float32, device=device)


def sample_bottom_wall(n: int = 150, seed: int = 3, device: str = "cpu") -> torch.Tensor:
    """Bottom wall: y = -0.5, x ~ uniform. BC: u = 0, v = 0."""
    rng = np.random.default_rng(seed)
    x  = rng.uniform(X_MIN, X_MAX, size=(n, 1)).astype(np.float32)
    y  = np.full_like(x, Y_MIN)
    return torch.tensor(np.hstack([x, y]), dtype=torch.float32, device=device)


def sample_cylinder(n: int = 300, seed: int = 4, device: str = "cpu") -> torch.Tensor:
    """
    Cylinder surface: uniform in angle θ ~ [0, 2π].
    BC: u = 0, v = 0  (no-slip).
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=n).astype(np.float32)
    x = (CYL_X + CYL_R * np.cos(theta)).reshape(-1, 1)
    y = (CYL_Y + CYL_R * np.sin(theta)).reshape(-1, 1)
    return torch.tensor(np.hstack([x, y]), dtype=torch.float32, device=device)


def sample_all_boundaries(
    n_inflow: int = 200,
    n_outflow: int = 200,
    n_top: int = 150,
    n_bot: int = 150,
    n_cyl: int = 300,
    seed_offset: int = 0,
    device: str = "cpu",
) -> dict:
    """
    Convenience wrapper — returns a dict with all boundary tensors.

    Recommended counts (defaults):
        inflow / outflow  : 200  — full velocity profile matters here
        top / bottom walls: 150  — simple no-slip, gradients are milder
        cylinder surface  : 300  — most critical BC, high curvature

    Returns
    -------
    dict with keys: 'inflow', 'outflow', 'top', 'bottom', 'cylinder'
    """
    return {
        "inflow"  : sample_inflow(n_inflow,  seed=seed_offset + 0, device=device),
        "outflow" : sample_outflow(n_outflow, seed=seed_offset + 1, device=device),
        "top"     : sample_top_wall(n_top,   seed=seed_offset + 2, device=device),
        "bottom"  : sample_bottom_wall(n_bot, seed=seed_offset + 3, device=device),
        "cylinder": sample_cylinder(n_cyl,   seed=seed_offset + 4, device=device),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_points(
    xy_col: torch.Tensor,
    bc_dict: dict,
    figsize: tuple = (12, 6),
    alpha_col: float = 0.25,
    point_size: float = 3.0,
    save_path: str = None,
) -> None:
    """
    Plot all collocation and boundary points.

    Parameters
    ----------
    xy_col   : interior collocation tensor  (output of sample_collocation)
    bc_dict  : dict from sample_all_boundaries
    figsize  : matplotlib figure size
    alpha_col: transparency for interior collocation points
    point_size: marker size
    save_path : if given, save the figure to this path instead of showing it
    """
    col_np = xy_col.detach().cpu().numpy()
    n_total = len(xy_col)

    # Split interior points back into their populations for colouring
    # We reconstruct approximate sizes from the tensor length using
    # the label array stored below — if you changed the counts, pass
    # them explicitly or just colour everything the same.
    color_map = {
        "Collocation (uniform)"    : ("steelblue",  None),
        "Collocation (near cyl.)"  : ("darkorange", None),
        "Collocation (wake)"       : ("green",      None),
        "Inflow  (u=1, v=0)"       : ("red",        bc_dict.get("inflow")),
        "Outflow (p=0, ∂u/∂x=0)"  : ("purple",     bc_dict.get("outflow")),
        "Top wall (no-slip)"       : ("brown",      bc_dict.get("top")),
        "Bottom wall (no-slip)"    : ("saddlebrown",bc_dict.get("bottom")),
        "Cylinder (no-slip)"       : ("black",      bc_dict.get("cylinder")),
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Draw domain boundary
    rect = plt.Rectangle(
        (X_MIN, Y_MIN), X_MAX - X_MIN, Y_MAX - Y_MIN,
        linewidth=1.5, edgecolor="black", facecolor="whitesmoke", zorder=0,
    )
    ax.add_patch(rect)

    # Draw cylinder
    cyl_patch = Circle(
        (CYL_X, CYL_Y), CYL_R,
        linewidth=1.5, edgecolor="black", facecolor="lightgrey", zorder=5,
    )
    ax.add_patch(cyl_patch)

    # --- Interior collocation points (three populations) ---
    # We only have the concatenated tensor; we recompute the split
    # from the default counts.  If you used custom counts pass them here.
    ax.scatter(
        col_np[:, 0], col_np[:, 1],
        s=point_size, c="steelblue", alpha=alpha_col, zorder=2,
        label=f"Collocation interior  (N={n_total})",
    )

    # --- Boundary points ---
    bc_styles = {
        "inflow"  : dict(color="red",         marker="|", s=20, label="Inflow"),
        "outflow" : dict(color="purple",       marker="|", s=20, label="Outflow"),
        "top"     : dict(color="brown",        marker="_", s=20, label="Top wall"),
        "bottom"  : dict(color="saddlebrown",  marker="_", s=20, label="Bottom wall"),
        "cylinder": dict(color="black",        marker="o", s=8,  label="Cylinder surface"),
    }
    for key, style in bc_styles.items():
        pts = bc_dict.get(key)
        if pts is None:
            continue
        pts_np = pts.detach().cpu().numpy()
        ax.scatter(
            pts_np[:, 0], pts_np[:, 1],
            s=style["s"], c=style["color"], marker=style["marker"],
            zorder=6, label=f"{style['label']}  (N={len(pts_np)})",
        )

    ax.set_xlim(X_MIN - 0.05, X_MAX + 0.05)
    ax.set_ylim(Y_MIN - 0.07, Y_MAX + 0.07)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("PINN Sampling — Collocation & Boundary Points")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Small demo when run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating collocation points …")
    xy_col = sample_collocation(n_uniform=30000)
    print(f"  Interior collocation: {xy_col.shape[0]} pts")

    print("Generating boundary points …")
    bc = sample_all_boundaries(
        n_inflow=200,
        n_outflow=200,
        n_top=150,
        n_bot=150,
        n_cyl=300,
    )
    for k, v in bc.items():
        print(f"  {k:10s}: {v.shape[0]} pts")

    print("\nadaptive_resample is available — import and call it from the")
    print("training loop with a trained model to add high-residual points.")

    print("\nPlotting …")
    visualize_points(xy_col, bc)
