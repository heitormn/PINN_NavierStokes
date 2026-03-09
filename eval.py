"""
eval.py
-------
Load the best saved PINN checkpoint and plot the predicted flow fields
(u, v, p) over the full 2-D domain.  Also loads the COMSOL reference
solution (SolutionCOMSOL.txt) and plots the relative error maps.

Domain  : [0, 1.2] x [-0.5, 0.5]
Cylinder: centre (0.3, 0), radius 0.08  (masked out in plots)

Usage
-----
    python eval.py                          # uses default models/best_model.pt
    python eval.py --ckpt path/to/model.pt  # custom checkpoint
    python eval.py --res 600                # grid resolution (default 400)
    python eval.py --save                   # save figures instead of showing
    python eval.py --ref  path/to/SolutionCOMSOL.txt
"""

import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.interpolate import griddata

from model import PINN

# ---------------------------------------------------------------------------
# Domain / geometry constants (must match training)
# ---------------------------------------------------------------------------
X_MIN, X_MAX = 0.0, 1.2
Y_MIN, Y_MAX = -0.5, 0.5
CYL_X, CYL_Y, CYL_R = 0.3, 0.0, 0.08


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device) -> PINN:
    """Load a PINN from a checkpoint file."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    layers = ckpt["cfg"]["layers"]
    model = PINN(layers).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("loss",  float("nan"))
    print(f"  Loaded  : {ckpt_path}")
    print(f"  Epoch   : {epoch}")
    print(f"  Loss    : {loss:.6e}")
    print(f"  Layers  : {layers}")
    return model


def build_grid(res: int, device: torch.device):
    """
    Build a regular grid of (x, y) points over the domain and return:
        xy_grid  — Tensor[M, 2]  (only points outside the cylinder)
        xx, yy   — full meshgrid arrays (shape [res, res])
        mask     — bool array [res, res], True where outside cylinder
    """
    xs = np.linspace(X_MIN, X_MAX, res, dtype=np.float32)
    ys = np.linspace(Y_MIN, Y_MAX, res, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    dist = np.sqrt((xx - CYL_X)**2 + (yy - CYL_Y)**2)
    mask = dist > CYL_R                 

    xy_flat = np.stack([xx[mask], yy[mask]], axis=1)
    xy_grid = torch.tensor(xy_flat, dtype=torch.float32, device=device)
    return xy_grid, xx, yy, mask


@torch.no_grad()
def predict(model: PINN, xy: torch.Tensor, batch: int = 65536) -> np.ndarray:
    """Run the model in batches; returns numpy array [N, 3] (u, v, p)."""
    parts = []
    for i in range(0, len(xy), batch):
        out = model(xy[i : i + batch])
        parts.append(out.cpu().numpy())
    return np.concatenate(parts, axis=0)


def _add_cylinder(ax):
    cyl = Circle((CYL_X, CYL_Y), CYL_R, color="lightgrey", ec="black", lw=1.2, zorder=5)
    ax.add_patch(cyl)


def _style_ax(ax, title):
    _add_cylinder(ax)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=11)


# ---------------------------------------------------------------------------
# COMSOL reference loader
# ---------------------------------------------------------------------------

def load_comsol(ref_path: str) -> dict:
    """
    Parse the COMSOL export file (SolutionCOMSOL.txt).
    Lines beginning with '%' are comments/headers.
    Columns: x  y  u  v  p  speed
    Pressure is used as-is (Pa), no normalisation.
    """
    data = np.loadtxt(ref_path, comments="%")
    out = {
        "x"    : data[:, 0].astype(np.float32),
        "y"    : data[:, 1].astype(np.float32),
        "u"    : data[:, 2].astype(np.float32),
        "v"    : data[:, 3].astype(np.float32),
        "p"    : data[:, 4].astype(np.float32),
        "speed": data[:, 5].astype(np.float32),
    }
    print(f"  COMSOL  : {len(out['x']):,} nodes loaded from {ref_path}")
    return out


def interpolate_comsol(comsol: dict, xx: np.ndarray, yy: np.ndarray,
                       mask: np.ndarray) -> dict:
    """
    Interpolate the scattered COMSOL FEM nodes onto the regular PINN grid
    using linear barycentric interpolation (scipy griddata).
    Returns field arrays of shape (res, res) with NaN inside the cylinder.
    """
    pts = np.stack([comsol["x"], comsol["y"]], axis=1)
    grid_pts = np.stack([xx[mask], yy[mask]], axis=1)

    result = {}
    for key in ("u", "v", "p", "speed"):
        nan_field = np.full(xx.shape, float("nan"), dtype=np.float32)
        nan_field[mask] = griddata(pts, comsol[key], grid_pts,
                                   method="linear").astype(np.float32)
        result[key] = nan_field
    return result


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

# CFD-standard colormaps (COMSOL Rainbow / jet for all fields)
_CMAP_FIELD = "jet"   # identical to COMSOL default "Rainbow" colormap
_CMAP_ERR   = "hot_r" # high-contrast sequential for error maps


def _contourf_panel(ax, fig, xx, yy, field, cmap, vmin, vmax, title, unit=""):
    cf = ax.contourf(xx, yy, field, levels=80,
                     cmap=cmap, vmin=vmin, vmax=vmax, extend="both")
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    if unit:
        cb.set_label(unit, fontsize=8)
    _style_ax(ax, title)


def _plot_comparison(var_label, unit,
                     pinn_field, cfd_field, abs_err,
                     xx, yy, save_dir, filename):
    """
    One figure with three side-by-side panels:
        PINN  |  COMSOL  |  |PINN − CFD|
    PINN and COMSOL share the same colour scale (driven by COMSOL range).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Colour scale driven by COMSOL (reference) range
    vmin = float(np.nanmin(cfd_field))
    vmax = float(np.nanmax(cfd_field))

    _contourf_panel(axes[0], fig, xx, yy, pinn_field,
                    _CMAP_FIELD, vmin, vmax, f"PINN — {var_label}", unit)
    _contourf_panel(axes[1], fig, xx, yy, cfd_field,
                    _CMAP_FIELD, vmin, vmax, f"COMSOL — {var_label}", unit)
    err_max = float(np.nanmax(abs_err)) or 1.0
    _contourf_panel(axes[2], fig, xx, yy, abs_err,
                    _CMAP_ERR, 0, err_max,
                    f"|PINN − CFD| — {var_label}", unit)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved   : {path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def run_eval(model: PINN, res: int, device: torch.device,
             ref_path: str | None = None,
             save_dir: str | None = None):

    print(f"\n  Building {res}×{res} evaluation grid …")
    xy_grid, xx, yy, mask = build_grid(res, device)

    # ------------------------------------------------------------------
    # PINN predictions
    # ------------------------------------------------------------------
    print(f"  Running PINN inference on {mask.sum():,} fluid points …")
    preds = predict(model, xy_grid)

    nan = float("nan")
    pinn = {}
    for i, key in enumerate(("u", "v", "p")):
        f = np.full(xx.shape, nan, dtype=np.float32)
        f[mask] = preds[:, i]
        pinn[key] = f
    pinn["speed"] = np.sqrt(pinn["u"]**2 + pinn["v"]**2)
    
    # ------------------------------------------------------------------
    # PINN streamlines
    # ------------------------------------------------------------------
    xs = np.linspace(X_MIN, X_MAX, res)
    ys = np.linspace(Y_MIN, Y_MAX, res)

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    cf2 = ax2.contourf(xx, yy, pinn["speed"], levels=60, cmap="viridis", extend="both")
    fig2.colorbar(cf2, ax=ax2, label="speed  ||u||")
    ax2.streamplot(xs, ys,
                   np.where(mask, pinn["u"], 0.0),
                   np.where(mask, pinn["v"], 0.0),
                   color="white", linewidth=0.6, density=1.8, arrowsize=0.8, zorder=4)
    _style_ax(ax2, "PINN — Streamlines coloured by speed")
    plt.tight_layout()
    if save_dir:
        p2 = os.path.join(save_dir, "eval_streamlines.png")
        plt.savefig(p2, dpi=150, bbox_inches="tight")
        print(f"  Saved   : {p2}")
    else:
        plt.show()
    plt.close(fig2)

    # ------------------------------------------------------------------
    # COMSOL reference + error (only when ref file is available)
    # ------------------------------------------------------------------
    if ref_path is None or not os.path.exists(ref_path):
        if ref_path is not None:
            print(f"  WARNING: reference file not found: {ref_path}")
        print("  Skipping COMSOL comparison (no reference file).")
        print("  Done.")
        return

    print(f"\n  Loading COMSOL reference …")
    comsol_raw = load_comsol(ref_path)
    print(f"  Interpolating COMSOL data onto {res}×{res} grid …")
    comsol = interpolate_comsol(comsol_raw, xx, yy, mask)

    # ------------------------------------------------------------------
    # L2 error metrics (fluid domain only)
    # ------------------------------------------------------------------
    def _l2(a, b):
        diff  = (a - b)[mask]
        ref   = b[mask]
        l2    = np.sqrt(np.nanmean(diff**2))
        rel   = l2 / (np.sqrt(np.nanmean(ref**2)) + 1e-12)
        return l2, rel

    l2_u,     rel_u     = _l2(pinn["u"],     comsol["u"])
    l2_v,     rel_v     = _l2(pinn["v"],     comsol["v"])
    l2_p,     rel_p     = _l2(pinn["p"],     comsol["p"])
    l2_speed, rel_speed = _l2(pinn["speed"], comsol["speed"])

    print(f"\n  {'Field':<12}  {'L2 error':>12}  {'Relative L2':>12}")
    print(f"  {'-'*40}")
    for name, l2, rel in [("u",     l2_u,     rel_u),
                           ("v",     l2_v,     rel_v),
                           ("p",     l2_p,     rel_p),
                           ("speed", l2_speed, rel_speed)]:
        print(f"  {name:<12}  {l2:>12.4e}  {rel:>12.4e}")

    # ------------------------------------------------------------------
    # Pointwise absolute error  |Q_PINN − Q_CFD|
    # ------------------------------------------------------------------
    def _abs_err(a, b):
        err = np.full(xx.shape, float("nan"), dtype=np.float32)
        err[mask] = np.abs(a - b)[mask]
        return err

    abs_err_u = _abs_err(pinn["u"], comsol["u"])
    abs_err_v = _abs_err(pinn["v"], comsol["v"])
    abs_err_p = _abs_err(pinn["p"], comsol["p"])

    # ------------------------------------------------------------------
    # Comparison figures: one per variable  (PINN | COMSOL | |PINN−CFD|)
    # ------------------------------------------------------------------
    print("  Plotting comparison figures …")

    _plot_comparison(
        "u  [m/s]", "m/s",
        pinn["u"], comsol["u"], abs_err_u,
        xx, yy, save_dir, "comparison_u.png",
    )
    _plot_comparison(
        "v  [m/s]", "m/s",
        pinn["v"], comsol["v"], abs_err_v,
        xx, yy, save_dir, "comparison_v.png",
    )
    _plot_comparison(
        "p  [Pa]", "Pa",
        pinn["p"], comsol["p"], abs_err_p,
        xx, yy, save_dir, "comparison_p.png",
    )

    print("  Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a saved PINN checkpoint.")
    p.add_argument("--ckpt",  default=os.path.join("models", "best_model.pt"),
                   help="Path to .pt checkpoint  (default: models/best_model.pt)")
    p.add_argument("--ref",   default="SolutionCOMSOL.txt",
                   help="Path to COMSOL reference file  (default: SolutionCOMSOL.txt)")
    p.add_argument("--res",   type=int, default=400,
                   help="Grid resolution along each axis  (default: 400)")
    p.add_argument("--save",  action="store_true",
                   help="Save figures to the same folder as the checkpoint "
                        "instead of displaying them interactively.")
    p.add_argument("--device", default=None,
                   help="'cuda' | 'cpu'  (auto-detected if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\n{'='*55}")
    print(f"  PINN  Evaluation")
    print(f"{'='*55}")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  Reference  : {args.ref}")
    print(f"  Resolution : {args.res} × {args.res}")
    print(f"  Device     : {device}")

    model   = load_model(args.ckpt, device)
    save_dir = os.path.dirname(os.path.abspath(args.ckpt)) if args.save else None
    run_eval(model, args.res, device, ref_path=args.ref, save_dir=save_dir)
