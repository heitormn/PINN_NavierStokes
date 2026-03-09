import torch.nn as nn
import torch

# ---------------------------------------------------------------------------
# PINN Architecture
# ---------------------------------------------------------------------------
# Classic feedforward PINN, fully connected.
# Activation: tanh, commonly used in PINNs due to its smoothness, which is
# required for the automatic-differentiation-based physics residuals.
# Weight initialisation: Xavier uniform, which is a common choice for tanh activations.
#
# For this problem, we have:
# Input: (x, y) — 2 features.
# Output: (u, v, p) — 3 outputs.
# ---------------------------------------------------------------------------


class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            
        # Input normalisation: maps x∈[0,1.2]→[-1,1], y∈[-0.5,0.5]→[-1,1]
        self.register_buffer('x_mean',  torch.tensor([0.6, 0.0], dtype=torch.float32))
        self.register_buffer('x_scale', torch.tensor([0.6, 0.5], dtype=torch.float32))
    def forward(self, x):
        # Normalize inputs to roughly [-1, 1] per dimension
        x = (x - self.x_mean) / self.x_scale
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)


# ---------------------------------------------------------------------------
# Helper: first- and second-order derivatives via autograd
# ---------------------------------------------------------------------------

def grad(outputs, inputs):
    """∂outputs/∂inputs, keeping the graph for higher-order derivatives."""
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


# ---------------------------------------------------------------------------
# Physics residuals  (steady, incompressible Navier-Stokes, 2-D)
# ---------------------------------------------------------------------------
# Equations solved:
#
#   continuity :  u_x + v_y = 0
#   x-momentum :  u·u_x + v·u_y + p_x − (μ/ρ)·(u_xx + u_yy) = 0
#   y-momentum :  u·v_x + v·v_y + p_y − (μ/ρ)·(v_xx + v_yy) = 0
#
# ---------------------------------------------------------------------------

def ns_residuals(model, xy_col, rho=1.0, mu=0.01):
    """
    Compute the three Navier-Stokes residuals at interior collocation points.

    Parameters
    ----------
    model   : PINN  — network that maps (x,y) → (u, v, p)
    xy_col  : Tensor[N, 2]  — collocation points with requires_grad=True
    rho     : float  — density [kg/m³] (default 1.0)
    mu      : float  — dynamic viscosity [Pa·s] (default 0.01)

    Returns
    -------
    r_cont, r_xmom, r_ymom : each Tensor[N, 1]
    """
    out = model(xy_col)                 # (N, 3)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    # --- first derivatives ---
    u_x = grad(u, xy_col)[:, 0:1]
    u_y = grad(u, xy_col)[:, 1:2]
    v_x = grad(v, xy_col)[:, 0:1]
    v_y = grad(v, xy_col)[:, 1:2]
    p_x = grad(p, xy_col)[:, 0:1]
    p_y = grad(p, xy_col)[:, 1:2]

    # --- second derivatives ---
    u_xx = grad(u_x, xy_col)[:, 0:1]
    u_yy = grad(u_y, xy_col)[:, 1:2]
    v_xx = grad(v_x, xy_col)[:, 0:1]
    v_yy = grad(v_y, xy_col)[:, 1:2]

    nu = mu / rho 

    r_cont  = u_x + v_y
    r_xmom  = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    r_ymom  = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return r_cont, r_xmom, r_ymom


# ---------------------------------------------------------------------------
# Total PINN loss
# ---------------------------------------------------------------------------
# Domain   : [0, 1.2] × [−0.5, 0.5]
# Geometry : solid circle centred at (0.3, 0) with radius 0.08
#
# Boundary conditions
# -------------------
#   Inflow  (x = 0)               : u = 0.3·(1 − 4y²),  v = 0  (parabolic, U_max=0.3)
#   Outflow (x = 1.2)             : p_avg = 0,  v = 0
#   Top wall    (y =  0.5)        : u = 0,  v = 0  (no-slip)
#   Bottom wall (y = −0.5)        : u = 0,  v = 0  (no-slip)
#   Cylinder wall (r = 0.08)      : u = 0,  v = 0  (no-slip)
# ---------------------------------------------------------------------------

def pinn_loss(model, xy_col, xy_in, xy_out, xy_top, xy_bot, xy_cyl, rho=1.0, mu=0.01,
              w_pde=1.0, w_bc=1.0):
    """
    Compute the total PINN loss for 2-D steady Navier-Stokes flow past a
    circular cylinder.

    Parameters
    ----------
    model   : PINN
    xy_col  : Tensor[Nc, 2]  interior collocation pts, requires_grad=True
    xy_in   : Tensor[Ni, 2]  inflow  boundary pts    (x = 0)
    xy_out  : Tensor[No, 2]  outflow boundary pts    (x = 1), requires_grad=True
    xy_top  : Tensor[Nt, 2]  top wall pts            (y =  0.5)
    xy_bot  : Tensor[Nb, 2]  bottom wall pts         (y = −0.5)
    xy_cyl  : Tensor[Nk, 2]  cylinder surface pts
    rho     : float           density [kg/m³]
    mu      : float           dynamic viscosity [Pa·s]
    w_pde   : float           weight for PDE residual terms
    w_bc    : float           weight for boundary-condition terms

    Returns
    -------
    loss        : scalar Tensor  — total weighted loss
    loss_dict   : dict           — individual loss components for logging
    """
    mse = nn.MSELoss()
    zeros = lambda n: torch.zeros(n, 1, device=xy_col.device)

    # ------------------------------------------------------------------
    # 1. PDE / physics residuals  (interior)
    # ------------------------------------------------------------------
    r_cont, r_xmom, r_ymom = ns_residuals(model, xy_col, rho, mu)
    loss_cont  = mse(r_cont,  zeros(r_cont.shape[0]))
    loss_xmom  = mse(r_xmom, zeros(r_xmom.shape[0]))
    loss_ymom  = mse(r_ymom, zeros(r_ymom.shape[0]))

    # ------------------------------------------------------------------
    # 2. Inflow  (x = 0): u = 0.3·(1 − 4y²),  v = 0  (parabolic, U_max=0.3)
    # ------------------------------------------------------------------
    out_in        = model(xy_in)
    y_in          = xy_in[:, 1:2]
    u_in_target   = 0.3 * (1.0 - 4.0 * y_in ** 2)
    loss_in_u     = mse(out_in[:, 0:1], u_in_target)
    loss_in_v     = mse(out_in[:, 1:2], zeros(xy_in.shape[0]))

    # ------------------------------------------------------------------
    # 3. Outflow (x = 1): p_avg = 0,  v = 0  (u·t = 0)
    # ------------------------------------------------------------------
    out_out      = model(xy_out)
    p_out_mean   = out_out[:, 2:3].mean()  # average pressure at outlet
    loss_out_p   = p_out_mean ** 2          # enforce average pressure = 0

    v_out = out_out[:, 1:2]
    loss_out_v   = mse(v_out, zeros(xy_out.shape[0]))  # v = 0 (tangential velocity)

    # ------------------------------------------------------------------
    # 4. Top / bottom walls (y = ±0.5): u = 0, v = 0   (no-slip)
    # ------------------------------------------------------------------
    out_top      = model(xy_top)
    loss_top_u   = mse(out_top[:, 0:1], zeros(xy_top.shape[0]))
    loss_top_v   = mse(out_top[:, 1:2], zeros(xy_top.shape[0]))

    out_bot      = model(xy_bot)
    loss_bot_u   = mse(out_bot[:, 0:1], zeros(xy_bot.shape[0]))
    loss_bot_v   = mse(out_bot[:, 1:2], zeros(xy_bot.shape[0]))

    # ------------------------------------------------------------------
    # 5. Cylinder surface: u = 0, v = 0   (no-slip)
    # ------------------------------------------------------------------
    out_cyl      = model(xy_cyl)
    loss_cyl_u   = mse(out_cyl[:, 0:1], zeros(xy_cyl.shape[0]))
    loss_cyl_v   = mse(out_cyl[:, 1:2], zeros(xy_cyl.shape[0]))

    # ------------------------------------------------------------------
    # Weighted total
    # ------------------------------------------------------------------
    loss_pde = loss_cont + loss_xmom + loss_ymom

    loss_bc  = (loss_in_u  + loss_in_v  +
                loss_out_p + loss_out_v +
                loss_top_u + loss_top_v  +
                loss_bot_u + loss_bot_v  +
                loss_cyl_u + loss_cyl_v)

    loss = w_pde * loss_pde + w_bc * loss_bc

    loss_dict = {
        "total"     : loss.item(),
        "pde/cont"  : loss_cont.item(),
        "pde/xmom"  : loss_xmom.item(),
        "pde/ymom"  : loss_ymom.item(),
        "bc/inflow_u"    : loss_in_u.item(),
        "bc/inflow_v"    : loss_in_v.item(),
        "bc/outlet_p"    : loss_out_p.item(),
        "bc/outlet_v"    : loss_out_v.item(),
        "bc/top_u"  : loss_top_u.item(),
        "bc/top_v"  : loss_top_v.item(),
        "bc/bot_u"  : loss_bot_u.item(),
        "bc/bot_v"  : loss_bot_v.item(),
        "bc/cyl_u"  : loss_cyl_u.item(),
        "bc/cyl_v"  : loss_cyl_v.item(),
    }

    return loss, loss_dict

