# main_debug.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Optional: improve numerical headroom
try:
    jax.config.update("jax_enable_x64", True)
except Exception:
    pass
# Uncomment to debug without JIT:
# jax.config.update("jax_disable_jit", True)

from momentRTjax.momentRT import (
    M1Config,  # your config dataclass (keep frozen if used as static)
    advance,   # we'll NOT use this in debug; we step manually to catch first NaN
)

# --------------------------
# Import hardened helpers (add these to your momentRT module if not there yet)
# --------------------------
def enforce_positivity_and_causality(E, F, c, epsE=1e-12):
    Ep = jnp.maximum(E, epsE)
    Fmag = jnp.linalg.norm(F, axis=-1)
    limit = jnp.minimum(1.0, (c * Ep) / (Fmag + 1e-30))
    Fp = F * limit[..., None]
    return Ep, Fp

# we’ll call your compiled step, but wrap with a post-check:
from functools import partial
from momentRTjax.momentRT import step_m1 as _step_m1  # jitted

def step_m1_safe(U, dt, cfg, kappa_a, kappa_t, E_eq):
    U = _step_m1(U, dt, cfg, kappa_a, kappa_t, E_eq)
    # Final safety clamp (in case sources drive weird transients)
    E = jnp.maximum(U[..., 0], cfg.epsE)
    F = U[..., 1:4]
    E, F = enforce_positivity_and_causality(E, F, cfg.c, epsE=cfg.epsE)
    return jnp.concatenate([E[..., None], F], axis=-1)

# --------------------------
# Problem setup (very safe)
# --------------------------
Nx, Ny, Nz = 64, 64, 64
dx = dy = dz = 1.0
c = 1.0
cfg = M1Config(c=c, dx=dx, dy=dy, dz=dz, cfl=0.2, epsE=1e-12)

# Opacities (keep >0 for absorber/emitter; small scattering damps early transients)
kappa_a_val = 0.05    # MUST be > 0
kappa_t_val = 1e-3

assert kappa_a_val > 0.0, "kappa_a_val must be > 0 when using E_eq as source."

kappa_a = jnp.ones((Nx, Ny, Nz)) * kappa_a_val
kappa_t = jnp.ones((Nx, Ny, Nz)) * kappa_t_val

# Source power and equilibrium energy in the source cell
L = 1e-1  # keep modest; you can scale up later
cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
cell_vol = dx * dy * dz
E_eq = jnp.zeros((Nx, Ny, Nz))
E_eq_center = L / (c * kappa_a_val * cell_vol)  # finite if kappa_a_val>0
E_eq = E_eq.at[cx, cy, cz].set(E_eq_center)

# Initial condition: strictly positive E, zero F
E0 = jnp.ones((Nx, Ny, Nz)) * 1e-6
F0 = jnp.zeros((Nx, Ny, Nz, 3))
U = jnp.concatenate([E0[..., None], F0], axis=-1)

# --------------------------
# Stable time step (more conservative than usual 3D CFL)
# --------------------------
dt = 0.12 * min(dx, dy, dz) / (c * np.sqrt(3))  # very safe CFL

# --------------------------
# Manual stepping with NaN guard
# --------------------------
nsteps = 400
for n in range(nsteps):
    U = step_m1_safe(U, dt, cfg, kappa_a, kappa_t, E_eq)
    # Check for NaNs/Inf as soon as they appear
    if bool(jnp.isnan(U).any() | jnp.isinf(U).any()):
        Ecur = U[..., 0]; Fcur = U[..., 1:4]
        print(f"[NaN] at step {n}")
        print("  E range:", float(jnp.nanmin(Ecur)), float(jnp.nanmax(Ecur)))
        print("  |F| max:", float(jnp.nanmax(jnp.linalg.norm(Fcur, axis=-1))))
        print("  E_eq_center:", float(E_eq_center))
        print("  kappa_a_val:", kappa_a_val, "kappa_t_val:", kappa_t_val, "dt:", dt)
        raise SystemExit

# If we get here, it’s clean
print("OK: no NaNs after", nsteps, "steps.")

# --------------------------
# Build analytic reference for plots
# --------------------------
x = (jnp.arange(Nx) - cx) * dx
y = (jnp.arange(Ny) - cy) * dy
z = (jnp.arange(Nz) - cz) * dz
X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
r = jnp.sqrt(X*X + Y*Y + Z*Z)
eps_r = 1e-6

E_num, F_num = U[..., 0], U[..., 1:4]

E_an = (L / (4.0 * jnp.pi * c)) * jnp.exp(-kappa_a_val * r) / jnp.maximum(r*r, eps_r)
ux = X / jnp.maximum(r, eps_r)
uy = Y / jnp.maximum(r, eps_r)
uz = Z / jnp.maximum(r, eps_r)
F_an = (L / (4.0 * jnp.pi)) * jnp.exp(-kappa_a_val * r) / jnp.maximum(r*r, eps_r)
F_an_vec = jnp.stack([F_an * ux, F_an * uy, F_an * uz], axis=-1)

# Mask borders & center for metrics
mask = jnp.ones_like(E_num, dtype=bool)
mask = mask.at[0, :, :].set(False).at[-1, :, :].set(False)
mask = mask.at[:, 0, :].set(False).at[:, -1, :].set(False)
mask = mask.at[:, :, 0].set(False).at[:, :, -1].set(False)
mask = mask.at[cx, cy, cz].set(False)

rel_err_E = jnp.abs((E_num - E_an) / (E_an + 1e-12))
mean_rel_err_E = float(jnp.mean(rel_err_E[mask]))
Fmag = jnp.linalg.norm(F_num, axis=-1)
Fhat_num = jnp.where((Fmag > 0)[..., None], F_num / (Fmag[..., None] + 1e-30), 0.0)
Fhat_an  = jnp.stack([ux, uy, uz], axis=-1)
cos_sim = jnp.clip(jnp.sum(Fhat_num * Fhat_an, axis=-1), -1.0, 1.0)
mean_cos = float(jnp.mean(cos_sim[mask]))
print(f"[M1] mean relative error E (masked): {mean_rel_err_E:.3e}")
print(f"[M1] mean cosine similarity(F, r̂) (masked): {mean_cos:.3f}")

# --------------------------
# Plots (XY mid-slice)
# --------------------------
outdir = "plots_m1_debug"
os.makedirs(outdir, exist_ok=True)
mid = cz

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(np.log10(np.asarray(E_num[:, :, mid]) + 1e-12), origin='lower', cmap='inferno')
plt.title("log10 E (numeric)"); plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(np.log10(np.asarray(E_an[:, :, mid]) + 1e-12), origin='lower', cmap='inferno')
plt.title("log10 E (analytic)"); plt.colorbar()
plt.subplot(1, 3, 3)
res = np.log10(np.asarray(rel_err_E[:, :, mid]) + 1e-12)
plt.imshow(res, origin='lower', cmap='magma'); plt.title("log10 |rel. err E|"); plt.colorbar()
plt.tight_layout(); plt.savefig(os.path.join(outdir, "E_xy.png"), dpi=150); plt.close()

Fmag_num = jnp.linalg.norm(F_num, axis=-1)
Fmag_an  = jnp.linalg.norm(F_an_vec, axis=-1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(np.log10(np.asarray(Fmag_num[:, :, mid]) + 1e-12), origin='lower', cmap='viridis')
plt.title("log10 |F| (numeric)"); plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(np.log10(np.asarray(Fmag_an[:, :, mid]) + 1e-12), origin='lower', cmap='viridis')
plt.title("log10 |F| (analytic)"); plt.colorbar()
plt.subplot(1, 3, 3)
resF = np.log10(np.asarray(jnp.abs((Fmag_num - Fmag_an) / (Fmag_an + 1e-12))[:, :, mid]) + 1e-12)
plt.imshow(resF, origin='lower', cmap='magma'); plt.title("log10 |rel. err |F||"); plt.colorbar()
plt.tight_layout(); plt.savefig(os.path.join(outdir, "F_xy.png"), dpi=150); plt.close()

print(f"Saved plots to: {outdir}")

