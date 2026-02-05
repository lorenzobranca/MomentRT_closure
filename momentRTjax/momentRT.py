# momentRT.py
# Minimal 3D gray radiative transfer (two-moment M1 closure) with HLLE,
# implicit absorption/scattering, and explicit volumetric sources.
# JAX implementation (CPU/GPU/TPU). First-order FV; MUSCL can be added later.

from __future__ import annotations
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp

Array = jnp.ndarray

# ---------------------------
# M1 closure (Levermore/Minerbo)
# ---------------------------

def m1_closure(E: Array, F: Array, c: float, eps: float = 1e-12):
    """
    Compute Eddington tensor components for the M1 closure.

    Inputs
      E : (...,)     energy density
      F : (...,3)    flux vector
      c : float      radiation speed
      eps : float    small floor for numerical safety

    Returns
      (P_xx, P_yy, P_zz, P_xy, P_xz, P_yz) each shape (...,)
    """
    Fmag = jnp.linalg.norm(F, axis=-1)
    cE   = c * jnp.maximum(E, eps)
    f    = jnp.clip(Fmag / cE, 0.0, 1.0 - 1e-9)

    # Eddington factor chi(f)
    chi = (3.0 + 4.0 * f * f) / (5.0 + 2.0 * jnp.sqrt(jnp.maximum(0.0, 4.0 - 3.0 * f * f)))
    a = 0.5 * (1.0 - chi)
    b = 0.5 * (3.0 * chi - 1.0)

    # Unit direction n
    n = jnp.where((Fmag > 0)[..., None], F / (Fmag[..., None] + eps), jnp.zeros_like(F))
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]

    Efac_a = E * a
    Efac_b = E * b

    P_xx = Efac_a + Efac_b * nx * nx
    P_yy = Efac_a + Efac_b * ny * ny
    P_zz = Efac_a + Efac_b * nz * nz
    P_xy = Efac_b * nx * ny
    P_xz = Efac_b * nx * nz
    P_yz = Efac_b * ny * nz
    return P_xx, P_yy, P_zz, P_xy, P_xz, P_yz


# ---------------------------
# Physical fluxes per direction
# ---------------------------

def flux_x(E: Array, F: Array, c: float):
    """
    x-direction flux:
      Fx(U) = [ c F_x,  c^2 P_xx,  c^2 P_xy,  c^2 P_xz ].
    """
    P_xx, _, _, P_xy, P_xz, _ = m1_closure(E, F, c)
    Fx0 = c * F[..., 0]
    Fx1 = c * c * P_xx
    Fx2 = c * c * P_xy
    Fx3 = c * c * P_xz
    return jnp.stack([Fx0, Fx1, Fx2, Fx3], axis=-1)

def flux_y(E: Array, F: Array, c: float):
    """
    y-direction flux:
      Fy(U) = [ c F_y,  c^2 P_xy,  c^2 P_yy,  c^2 P_yz ].
    """
    P_xx, P_yy, _, P_xy, _, P_yz = m1_closure(E, F, c)
    Fy0 = c * F[..., 1]
    Fy1 = c * c * P_xy
    Fy2 = c * c * P_yy
    Fy3 = c * c * P_yz
    return jnp.stack([Fy0, Fy1, Fy2, Fy3], axis=-1)

def flux_z(E: Array, F: Array, c: float):
    """
    z-direction flux:
      Fz(U) = [ c F_z,  c^2 P_xz,  c^2 P_yz,  c^2 P_zz ].
    """
    _, _, P_zz, _, P_xz, P_yz = m1_closure(E, F, c)
    Fz0 = c * F[..., 2]
    Fz1 = c * c * P_xz
    Fz2 = c * c * P_yz
    Fz3 = c * c * P_zz
    return jnp.stack([Fz0, Fz1, Fz2, Fz3], axis=-1)


# ---------------------------
# HLLE numerical flux (symmetric speeds ±c)
# ---------------------------

def hlle(F_L: Array, F_R: Array, U_L: Array, U_R: Array, smax: float):
    """HLLE flux with sL=-smax, sR=+smax."""
    return 0.5 * (F_L + F_R) - 0.5 * smax * (U_R - U_L)


# ---------------------------
# Finite-volume utilities
# ---------------------------

def periodic_roll(a: Array, shift: int, axis: int):
    return jnp.roll(a, shift=shift, axis=axis)

def divergence_flux(U: Array, dx: float, dy: float, dz: float, c: float, epsE: float = 1e-12):
    """
    Compute ∇·F with HLLE in x, y, z. Periodic BC.
    U shape: (Nx, Ny, Nz, 4)  with [E, Fx, Fy, Fz].
    Returns: same shape as U (divergence of the conserved fluxes).
    """
    # Positive energy proxies for closure stability
    E  = jnp.maximum(U[..., 0], epsE)
    F  = U[..., 1:4]

    # x-interfaces
    U_R  = periodic_roll(U, -1, axis=0)
    E_R  = jnp.maximum(U_R[..., 0], epsE)
    F_R  = U_R[..., 1:4]
    Fx_L = flux_x(E,  F,  c)
    Fx_R = flux_x(E_R, F_R, c)
    Fx_face = hlle(Fx_L, Fx_R, U, U_R, smax=c)
    div_x = (Fx_face - periodic_roll(Fx_face, 1, axis=0)) / dx

    # y-interfaces
    U_R  = periodic_roll(U, -1, axis=1)
    E_R  = jnp.maximum(U_R[..., 0], epsE)
    F_R  = U_R[..., 1:4]
    Fy_L = flux_y(E,  F,  c)
    Fy_R = flux_y(E_R, F_R, c)
    Fy_face = hlle(Fy_L, Fy_R, U, U_R, smax=c)
    div_y = (Fy_face - periodic_roll(Fy_face, 1, axis=1)) / dy

    # z-interfaces
    U_R  = periodic_roll(U, -1, axis=2)
    E_R  = jnp.maximum(U_R[..., 0], epsE)
    F_R  = U_R[..., 1:4]
    Fz_L = flux_z(E,  F,  c)
    Fz_R = flux_z(E_R, F_R, c)
    Fz_face = hlle(Fz_L, Fz_R, U, U_R, smax=c)
    div_z = (Fz_face - periodic_roll(Fz_face, 1, axis=2)) / dz

    return div_x + div_y + div_z


# ---------------------------
# Source terms
# ---------------------------

def source_implicit(E: Array, F: Array, dt: float, c: float,
                    kappa_a: Array, kappa_t: Array, E_eq: Array):
    """
    Local implicit update for absorption/emission & transport damping:
      dE/dt =  c kappa_a (E_eq - E)
      dF/dt = -c kappa_t F

    Implicit Euler (algebraic, local):
      E^{n+1} = (E* + dt c kappa_a E_eq) / (1 + dt c kappa_a)
      F^{n+1} = F* / (1 + dt c kappa_t)
    """
    fac_E = 1.0 / (1.0 + dt * c * kappa_a)
    E_new = (E + dt * c * kappa_a * E_eq) * fac_E

    fac_F = 1.0 / (1.0 + dt * c * kappa_t)
    F_new = F * fac_F[..., None]
    return E_new, F_new


def enforce_positivity_and_causality(E: Array, F: Array, c: float, epsE: float = 1e-12):
    """
    Clamp E >= epsE and |F| <= cE. Keeps the M1 closure well-defined.
    """
    Ep = jnp.maximum(E, epsE)
    Fmag = jnp.linalg.norm(F, axis=-1)
    limit = jnp.minimum(1.0, (c * Ep) / (Fmag + 1e-30))
    Fp = F * limit[..., None]
    return Ep, Fp


# ---------------------------
# Timestep control
# ---------------------------

def cfl_dt(dx: float, dy: float, dz: float, c: float, cfl: float = 0.4):
    """Basic CFL step for wave speed ~ c."""
    return cfl * min(dx, dy, dz) / c


# ---------------------------
# Main integrator
# ---------------------------

@dataclass(frozen=True)  # hashable/static for jit caching
class M1Config:
    c: float = 1.0
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0
    cfl: float = 0.4
    epsE: float = 1e-12


@partial(jax.jit, static_argnums=(2,))  # cfg is the 3rd positional arg
def step_m1(U: Array, dt: float, cfg: M1Config,
            kappa_a: Array, kappa_t: Array, E_eq: Array,
            S_E: Array, S_F: Array):
    """
    One FV step:
      1) Hyperbolic update (HLLE, periodic BC)
      2) Positivity/causality limiter
      3) Implicit absorption/scattering
      4) Explicit volumetric sources (S_E, S_F)
      5) Positivity/causality limiter
    """
    # Hyperbolic update
    divU   = divergence_flux(U, cfg.dx, cfg.dy, cfg.dz, cfg.c, epsE=cfg.epsE)
    U_star = U - dt * divU

    # Pre-source limiter
    E_star = jnp.maximum(U_star[..., 0], cfg.epsE)
    F_star = U_star[..., 1:4]
    E_star, F_star = enforce_positivity_and_causality(E_star, F_star, cfg.c, epsE=cfg.epsE)

    # Implicit local sources (absorption/emission & flux damping)
    E_np1, F_np1 = source_implicit(E_star, F_star, dt, cfg.c, kappa_a, kappa_t, E_eq)

    # Explicit volumetric sources
    E_np1 = E_np1 + dt * S_E
    F_np1 = F_np1 + dt * S_F

    # Post-source limiter
    E_np1, F_np1 = enforce_positivity_and_causality(E_np1, F_np1, cfg.c, epsE=cfg.epsE)

    return jnp.concatenate([E_np1[..., None], F_np1], axis=-1)


def advance(U0: Array, nsteps: int, cfg: M1Config,
            kappa_a: Array, kappa_t: Array, E_eq: Array,
            dt: float | None = None,
            S_E: Array | None = None, S_F: Array | None = None):
    """
    Advance nsteps. If dt is None, uses CFL dt (fixed).
    Optional explicit sources:
      S_E: (Nx,Ny,Nz)    energy source density [power / volume]
      S_F: (Nx,Ny,Nz,3)  momentum source       [flux / time]
    """
    if dt is None:
        dt = cfl_dt(cfg.dx, cfg.dy, cfg.dz, cfg.c, cfg.cfl)

    # Default explicit sources to zeros (avoid None under jit)
    if S_E is None:
        S_E = jnp.zeros_like(U0[..., 0])
    if S_F is None:
        S_F = jnp.zeros_like(U0[..., 1:4])

    def body(U, _):
        U = step_m1(U, dt, cfg, kappa_a, kappa_t, E_eq, S_E, S_F)
        return U, None

    Uf, _ = jax.lax.scan(body, U0, None, length=nsteps)
    return Uf

