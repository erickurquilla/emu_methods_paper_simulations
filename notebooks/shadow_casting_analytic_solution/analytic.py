"""Analytic shadow-casting solution moved out of the notebook.

The functions use the geometry, opacities, and angular mesh of the caller.
Call configure(namespace) with the notebook globals before evaluating them.
Physical constants come from phys_const.
"""
import sys
from pathlib import Path

import numpy as np

_phys_const_dir = Path(__file__).resolve().parents[2] / "phys_const"
if str(_phys_const_dir) not in sys.path:
    sys.path.insert(0, str(_phys_const_dir))
from phys_const import PhysConst

_REQUIRED = ['kappa_a', 'kappa_e', 'S_e', 'center_e', 'center_a', 'R_e', 'R_a', 'e_sep', 'x_external', 'x_internal', 'beta_u', 'beta_p', 'd', 'dx', 'dy', 'dz', 'ncell', 'phat', 'vphase_a', 'n_directions', 'full_f_bytes']


def configure(namespace):
    """Copy solver inputs from a notebook namespace into this module."""
    missing = [name for name in _REQUIRED if name not in namespace]
    if missing:
        raise KeyError("shadow solution is missing: " + ", ".join(missing))
    module = globals()
    for name in _REQUIRED:
        module[name] = namespace[name]


def clipped_chord(position, direction, center, radius):
    """Length of the backward half-ray inside a ball.

    Backward ray: y(s) = position - s * direction, s >= 0.
    `position` and `direction` broadcast on every axis except the last,
    which holds Cartesian components. A miss, a tangent ray (Delta = 0),
    and a sphere that lies only downstream (s_+ <= 0) all return 0.
    An interior point returns s_+.
    """
    relative = np.asarray(position, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    impact_projection = np.sum(relative * direction, axis=-1)
    distance_squared = np.sum(relative * relative, axis=-1)
    delta = radius * radius - distance_squared + impact_projection * impact_projection
    root = np.sqrt(np.maximum(delta, 0.0))
    s_plus = impact_projection + root
    s_minus = impact_projection - root
    length = s_plus - np.maximum(s_minus, 0.0)
    hits = (delta > 0.0) & (s_plus > 0.0)
    return np.where(hits, length, 0.0)


def f_on_points(points, directions, kappa_absorber=None, max_bytes=256 * 1024**2):
    """Occupation f at `points` (N, 3) for `directions` (A, 3).

    Returns an array of shape (N, A). Large point sets are evaluated in
    batches so the temporary (N, A) arrays stay near `max_bytes`.
    `kappa_absorber=0` is the isolated emitter.
    """
    if kappa_absorber is None:
        kappa_absorber = kappa_a
    points = np.asarray(points, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    if directions.ndim == 1:
        directions = directions.reshape(1, 3)
    n_points = points.shape[0]
    n_dir = directions.shape[0]
    if n_points > 1 and n_points * n_dir * 8 > max_bytes:
        batch = max(1, int(max_bytes // (n_dir * 8)))
        values = np.empty((n_points, n_dir), dtype=np.float64)
        for start in range(0, n_points, batch):
            stop = min(start + batch, n_points)
            values[start:stop] = f_on_points(
                points[start:stop],
                directions,
                kappa_absorber=kappa_absorber,
                max_bytes=np.inf,
            )
        return values

    ell_emitter = clipped_chord(points[:, None, :], directions[None, :, :], center_e, R_e)
    ell_absorber = clipped_chord(points[:, None, :], directions[None, :, :], center_a, R_a)
    # Foreground only when the absorber center lies upstream of the emitter
    # center along the backward ray. Theta(0) = 0.
    foreground = (directions @ e_sep) > 0.0
    transmission = np.exp(-kappa_absorber * ell_absorber * foreground[None, :])
    emergent = 1.0 - np.exp(-kappa_e * ell_emitter)
    return S_e * emergent * transmission


def ray_quantities(position, direction, kappa_absorber=None):
    """Chords, transmission, and f for one point and one direction."""
    if kappa_absorber is None:
        kappa_absorber = kappa_a
    position = np.asarray(position, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    ell_emitter = float(clipped_chord(position, direction, center_e, R_e))
    ell_absorber = float(clipped_chord(position, direction, center_a, R_a))
    foreground = bool(np.dot(direction, e_sep) > 0.0)
    optical_absorber = kappa_absorber * ell_absorber if foreground else 0.0
    transmission = float(np.exp(-optical_absorber))
    emergent = float(1.0 - np.exp(-kappa_e * ell_emitter))
    return {
        "ell_e": ell_emitter,
        "ell_a": ell_absorber,
        "foreground": foreground,
        "transmission": transmission,
        "emergent": emergent,
        "f": S_e * emergent * transmission,
    }


def separation_coordinates(position):
    """Axial coordinate from the emitter center, and cylindrical radius."""
    position = np.asarray(position, dtype=np.float64)
    relative = position - center_e
    axial = np.sum(relative * e_sep, axis=-1)
    transverse = relative - axial[..., None] * e_sep
    cylindrical_radius = np.linalg.norm(transverse, axis=-1)
    return axial, cylindrical_radius


def shadow_surfaces(axial):
    """Umbra, antumbra, and penumbra cylindrical radii at axial coordinate `axial`."""
    axial = np.asarray(axial, dtype=np.float64)
    varpi_umbra = (x_external - axial) * np.tan(beta_u)
    varpi_antumbra = (axial - x_external) * np.tan(beta_u)
    varpi_penumbra = (axial - x_internal) * np.tan(beta_p)
    return varpi_umbra, varpi_antumbra, varpi_penumbra


def shadow_region(position):
    """Opaque-limit region label. Finite-kappa f is still the chord formula."""
    position = np.asarray(position, dtype=np.float64)
    if np.linalg.norm(position - center_a) <= R_a:
        return "inside absorber"
    if np.linalg.norm(position - center_e) <= R_e:
        return "inside emitter"
    axial, cylindrical_radius = separation_coordinates(position)
    varpi_umbra, varpi_antumbra, varpi_penumbra = shadow_surfaces(float(axial))
    axial_umbra_start = d + R_a * np.sin(beta_u)
    axial_shadow_start = d - R_a * np.sin(beta_p)
    # The common-tangent surface is the boundary. A float residual of
    # ~1e-10 cm must not drop that surface into the interior label.
    on_umbra = abs(cylindrical_radius - varpi_umbra) <= 1e-6 * max(varpi_umbra, 1.0)
    on_antumbra = abs(cylindrical_radius - varpi_antumbra) <= 1e-6 * max(abs(varpi_antumbra), 1.0)
    on_penumbra = abs(cylindrical_radius - varpi_penumbra) <= 1e-6 * max(abs(varpi_penumbra), 1.0)
    if axial_umbra_start <= axial <= x_external and on_umbra:
        return "umbra boundary"
    if axial > x_external and on_antumbra:
        return "antumbra boundary"
    if axial >= axial_shadow_start and on_penumbra:
        return "penumbra boundary"
    if axial_umbra_start <= axial <= x_external and cylindrical_radius < varpi_umbra:
        return "umbra"
    if axial > x_external and cylindrical_radius < varpi_antumbra:
        return "antumbra"
    if axial >= axial_shadow_start and cylindrical_radius < varpi_penumbra:
        return "penumbra"
    return "unobstructed"


def cell_center(i, j, k):
    """Center of cell (i, j, k). Indices are zero-based."""
    return np.array(
        [(i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz],
        dtype=np.float64,
    )


def cell_index(position):
    """Zero-based index of the cell containing `position`."""
    position = np.asarray(position, dtype=np.float64)
    return (
        int(np.floor(position[0] / dx)),
        int(np.floor(position[1] / dy)),
        int(np.floor(position[2] / dz)),
    )


def f_at_cell(i, j, k):
    """f[i, j, k, a] at the cell center, for every quadrature direction."""
    values = f_on_points(cell_center(i, j, k), phat)
    return values[0]


def moments_from_f(distribution, directions=None, phase_volume=None):
    """n and F from occupation samples on a common direction set.

    `distribution` has the direction axis last. `phase_volume` is the
    unscaled per-direction Vphase. The sums are

        n = sum_a f_a * Vphase_a / (2*pi*hbar*c)^3
        F = sum_a f_a * n_hat_a * Vphase_a / (2*pi*hbar*c)^3

    Units are 1/cm^3. There is no factor of c.
    """
    if directions is None:
        directions = phat
    if phase_volume is None:
        phase_volume = vphase_a
    distribution = np.asarray(distribution, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    phase_volume = np.asarray(phase_volume, dtype=np.float64)
    prefactor = phase_volume / (2.0 * np.pi * PhysConst.hbar * PhysConst.c) ** 3
    number_density = np.sum(distribution * prefactor, axis=-1)
    number_flux = np.einsum(
        "...a,aj,a->...j",
        distribution,
        directions,
        prefactor,
        optimize=True,
    )
    return number_density, number_flux


def reference_moments_on_points(points, max_bytes=128 * 1024**2):
    """n, F, and isolated-emitter n_0 at points of shape (N, 3).

    Both densities share one pair of chord lengths. n_0 drops the absorber.
    Units are cm^-3. F is the number flux divided by c.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    n_dir = phat.shape[0]
    batch = max(1, int(max_bytes // (n_dir * 8)))
    foreground = (phat @ e_sep) > 0.0
    number = np.empty(points.shape[0], dtype=np.float64)
    isolated = np.empty(points.shape[0], dtype=np.float64)
    flux = np.empty((points.shape[0], 3), dtype=np.float64)
    for start in range(0, points.shape[0], batch):
        stop = min(start + batch, points.shape[0])
        chunk = points[start:stop]
        ell_emitter = clipped_chord(chunk[:, None, :], phat[None, :, :], center_e, R_e)
        ell_absorber = clipped_chord(chunk[:, None, :], phat[None, :, :], center_a, R_a)
        transmission = np.exp(-kappa_a * ell_absorber * foreground[None, :])
        emergent = 1.0 - np.exp(-kappa_e * ell_emitter)
        occupation_0 = S_e * emergent
        occupation = occupation_0 * transmission
        number[start:stop], flux[start:stop] = moments_from_f(
            occupation, directions=phat, phase_volume=vphase_a
        )
        isolated[start:stop], _ = moments_from_f(
            occupation_0, directions=phat, phase_volume=vphase_a
        )
    return number, flux, isolated


def occupation_number(x, y, z, phat):
    """Occupation number f at (x, y, z) for a unit momentum direction.

    x, y, z are in cm. phat is a unit vector, shape (3,), or several
    directions of shape (A, 3). A single direction returns a float.
    """
    position = np.array([x, y, z], dtype=np.float64)
    direction = np.asarray(phat, dtype=np.float64)
    values = f_on_points(position, direction)[0]
    if direction.ndim == 1:
        return float(values[0])
    return values


def number_density_and_flux(x, y, z):
    """Number density n and number flux F at (x, y, z).

    Both are in cm^-3. F is the EMU flux, the number flux divided by c.
    The sum uses the configured theta-phi mesh and its solid-angle weights.
    """
    position = np.array([x, y, z], dtype=np.float64)
    distribution = f_on_points(position, phat)[0]
    number_density, number_flux = moments_from_f(distribution)
    return float(number_density), np.asarray(number_flux, dtype=np.float64)


def moments_at_cell(i, j, k):
    """n[i, j, k] and F[i, j, k, :] from f at that cell center."""
    number_density, number_flux = moments_from_f(f_at_cell(i, j, k))
    return float(number_density), number_flux


def evaluate_chunk(i0, i1, j0=0, j1=None, k0=0, k1=None, store_f=True, max_f_bytes=2 * 1024**3):
    """Evaluate a rectangular block of cells, on cell centers.

    Index ranges are half-open. Returns
        f_chunk with shape (i1-i0, j1-j0, k1-k0, n_directions) or None,
        n_chunk  with shape (i1-i0, j1-j0, k1-k0),
        F_chunk  with shape (i1-i0, j1-j0, k1-k0, 3).
    Refuses to allocate `f` above `max_f_bytes`. The full grid at this
    resolution is about 8.82 GiB, so request slabs, or one cell via
    f_at_cell.
    """
    if j1 is None:
        j1 = ncell[1]
    if k1 is None:
        k1 = ncell[2]
    if not (
        0 <= i0 < i1 <= ncell[0]
        and 0 <= j0 < j1 <= ncell[1]
        and 0 <= k0 < k1 <= ncell[2]
    ):
        raise IndexError(
            f"chunk [{i0}:{i1}, {j0}:{j1}, {k0}:{k1}] is outside ncell={ncell}"
        )
    n_i = i1 - i0
    n_j = j1 - j0
    n_k = k1 - k0
    f_bytes = n_i * n_j * n_k * n_directions * np.dtype(np.float64).itemsize
    if store_f and f_bytes > max_f_bytes:
        raise MemoryError(
            f"f chunk would be {f_bytes / 1024**3:.2f} GiB "
            f"(full grid {full_f_bytes / 1024**3:.2f} GiB). "
            "Use a thinner slab, store_f=False, or f_at_cell(i, j, k)."
        )
    x_centers = (np.arange(i0, i1) + 0.5) * dx
    y_centers = (np.arange(j0, j1) + 0.5) * dy
    z_centers = (np.arange(k0, k1) + 0.5) * dz
    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    distribution = f_on_points(points, phat)
    number_density, number_flux = moments_from_f(distribution)
    number_density = number_density.reshape(n_i, n_j, n_k)
    number_flux = number_flux.reshape(n_i, n_j, n_k, 3)
    if not store_f:
        return None, number_density, number_flux
    return distribution.reshape(n_i, n_j, n_k, n_directions), number_density, number_flux


def reference_moments_x_slab(i0, i1):
    """n, F, and isolated n_0 on the half-open x-slab [i0, i1)."""
    x_centers = (np.arange(i0, i1) + 0.5) * dx
    y_centers = (np.arange(ncell[1]) + 0.5) * dy
    z_centers = (np.arange(ncell[2]) + 0.5) * dz
    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    points = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    number, flux, isolated = reference_moments_on_points(points)
    shape = (i1 - i0, int(ncell[1]), int(ncell[2]))
    return number.reshape(shape), flux.reshape(shape + (3,)), isolated.reshape(shape)


def reference_moments_x_slab_task(bounds):
    """Pool entry: bounds is (i0, i1). Returns the slab index with the moments."""
    i0, i1 = bounds
    number, flux, isolated = reference_moments_x_slab(i0, i1)
    return i0, number, flux, isolated
