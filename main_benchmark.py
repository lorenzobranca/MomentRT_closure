# main_m1_benchmark.py
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # optional

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from momentRTjax.momentRT import M1Config, advance, cfl_dt



def main():
    # --------------------------
    # Grid / physical parameters
    # --------------------------
    Nx, Ny, Nz = 64, 64, 64
    dx = dy = dz = 1.0
    c = 1.0

    cfg = M1Config(c=c, dx=dx, dy=dy, dz=dz, cfl=0.4, epsE=1e-12)

    # --------------------------
    # Medium and source settings
    # --------------------------
    # Absorption (set to 0.0 for pure transport; keep small >0 to include attenuation)
    kappa_a_val = 0.05
    # Transport/scattering opacity (damps flux; small positive stabilizes start-up)
    kappa_t_val = 1e-3

    # Total luminosity (power injected by the central cell)
    L = 1.0

    # --------------------------
    # Initial condition
    # --------------------------
    E0 = jnp.ones((Nx, Ny, Nz)) * 1e-6
    F0 = jnp.zeros((Nx, Ny, Nz, 3))
    U0 = jnp.concatenate([E0[..., None], F0], axis=-1)

    # --------------------------
    # Material fields (+ LTE, unused unless kappa_a>0 and you want E_eq forcing)
    # --------------------------
    kappa_a = jnp.ones((Nx, Ny, Nz)) * kappa_a_val
    kappa_t = jnp.ones((Nx, Ny, Nz)) * kappa_t_val
    E_eq    = jnp.zeros((Nx, Ny, Nz))

    # --------------------------
    # Explicit volumetric source S_E (central cell), S_F = 0
    # --------------------------
    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    cell_vol = dx * dy * dz
    S_E = jnp.zeros((Nx, Ny, Nz)).at[cx, cy, cz].set(L / cell_vol)
    S_F = jnp.zeros((Nx, Ny, Nz, 3))

    # --------------------------
    # Time step & integration
    # --------------------------
    # Use CFL-based dt (you can set dt=None and let advance() choose it).
    dt = 0.12 * min(dx, dy, dz) / (c * np.sqrt(3.0))#cfl_dt(dx, dy, dz, c, cfg.cfl)
    nsteps = 400

    U = advance(U0, nsteps=nsteps, cfg=cfg,
                kappa_a=kappa_a, kappa_t=kappa_t, E_eq=E_eq,
                dt=dt, S_E=S_E, S_F=S_F)

    # Safety check
    if bool(jnp.isnan(U).any() | jnp.isinf(U).any()):
        raise RuntimeError("Solution contains NaN/Inf. Try reducing dt, lowering L, "
                           "or increasing kappa_t_val slightly.")

    E_num, F_num = U[..., 0], U[..., 1:4]

    # --------------------------
    # Analytic reference
    # --------------------------
    # E_an(r) = L / (4π c r^2) * exp(-kappa_a * r)
    # F_an(r) = L / (4π r^2) * exp(-kappa_a * r) * r_hat
    x = (jnp.arange(Nx) - cx) * dx
    y = (jnp.arange(Ny) - cy) * dy
    z = (jnp.arange(Nz) - cz) * dz
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    r = jnp.sqrt(X*X + Y*Y + Z*Z)
    eps_r = 1e-6

    E_an = (L / (4.0 * jnp.pi * c)) * jnp.exp(-kappa_a_val * r) / jnp.maximum(r*r, eps_r)
    ux = X / jnp.maximum(r, eps_r)
    uy = Y / jnp.maximum(r, eps_r)
    uz = Z / jnp.maximum(r, eps_r)
    F_an = (L / (4.0 * jnp.pi)) * jnp.exp(-kappa_a_val * r) / jnp.maximum(r*r, eps_r)
    F_an_vec = jnp.stack([F_an * ux, F_an * uy, F_an * uz], axis=-1)

    # --------------------------
    # Metrics
    # --------------------------
    # Mask borders and the singular center cell
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
    print(f"[M1] mean cosine similarity(F, r̂)   : {mean_cos:.3f}")

    # --------------------------
    # Plots
    # --------------------------
    outdir = "plots_m1_benchmark"
    os.makedirs(outdir, exist_ok=True)
    mid = cz
    
    E_an =  E_an.at[mid, mid, mid].set(E_num)

    # Energy slices: numeric | analytic | log10 residual
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
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "E_slices_xy.png"), dpi=150); plt.close()

    # Flux magnitude: numeric vs analytic + rel error
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
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "Fmag_slices_xy.png"), dpi=150); plt.close()

    # Direction metrics: cosine similarity + angle error
    ang_deg = np.degrees(np.arccos(np.clip(np.asarray(cos_sim[:, :, mid]), -1.0, 1.0)))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.asarray(cos_sim[:, :, mid]), origin='lower', vmin=0, vmax=1, cmap='plasma')
    plt.title("cosine(F_num, r̂)"); plt.colorbar()
    plt.subplot(1, 2, 2)
    vmax_ang = min(60.0, float(np.nanpercentile(ang_deg, 99))) if np.isfinite(ang_deg).any() else 30.0
    plt.imshow(ang_deg, origin='lower', vmin=0, vmax=vmax_ang, cmap='magma')
    plt.title("angle error (deg)"); plt.colorbar()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "Fdir_metrics_xy.png"), dpi=150); plt.close()

    # Radial profile of E (x-axis through center)
    yy, zz = cy, cz
    r_line = (np.arange(cx+1, Nx) - cx) * dx
    E_line_num = np.asarray(E_num[cx+1:, yy, zz])
    E_line_an  = np.asarray(E_an [cx+1:, yy, zz])

    plt.figure(figsize=(6, 4))
    plt.loglog(r_line, E_line_num, 'o', ms=3, label='numeric')
    plt.loglog(r_line, E_line_an,  '-', lw=2, label='analytic')
    plt.xlabel("r (cells)"); plt.ylabel("E")
    plt.title("Radial profile of E (x-axis)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "E_radial_profile.png"), dpi=150); plt.close()

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    # Optional: enable 64-bit for a bit more numerical headroom
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass
    main()

