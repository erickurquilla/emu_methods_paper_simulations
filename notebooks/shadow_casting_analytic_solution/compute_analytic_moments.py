"""Write the reference moments into the directory where this script is launched.

The mesh, opacities, energy bin, and angular resolution are the inputs below.
Change them here, then submit the script from the directory that should
receive the HDF5 file:

    python compute_analytic_moments.py
    python compute_analytic_moments.py --workers 32

The output name is shadow_casting_analytic_moments_ncell_{Nx}_{Ny}_{Nz}.h5.
A file already in that directory is resumed when its signature matches these
inputs. A different signature is left untouched.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[1]
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

import shadow_casting_analytic_solution as analytic_solution
from phys_const import CGSUnitsConst
from shadow_casting_analytic_solution.analytic import (
    reference_moments_on_points,
    reference_moments_x_slab_task,
)

# ---------------------------------------------------------------------------
# Inputs. These are the quantities to edit before submitting a job.
# ---------------------------------------------------------------------------

ncell = (192, 64, 64)
Lx = 8424000.0  # cm, (R + 0.2 R) * 6, R = 1.17e6 cm
Ly = 2808000.0  # cm, (R + 0.2 R) * 2
Lz = 2808000.0  # cm

R = 1170000.0  # cm
R_e = R
R_a = 1.5 * R / 4.5  # R/3
r_emitter_sphere_center_cm = (1425937.5, 1425937.5, 1425937.5)
r_shadowing_sphere_center_cm = (4935937.5, 1425937.5, 1425937.5)

mu_e = -3.130432908491162  # MeV
T_e = 5.1189  # MeV
kappa_e = 0.00000854544806011998  # 1/cm, NuLib absorption, used as stored

mu_a = -19.95627340541274  # MeV
T_a = 0.0566631  # MeV
kappa_a = 0.00012391882339512724  # 1/cm

mu_vacuum = -25.859107462353197  # MeV
T_vacuum = 0.05666306  # MeV
kappa_vacuum = 1.4877457720665111e-9  # 1/cm; recorded, not used in f

# NuLib group containing the run's monochromatic energy.
E_MeV = 144.74
E_lo_MeV = 128.65
E_hi_MeV = 160.83

# Angular mesh.
n_theta = 64 * 2  # bins in polar angle, theta in [0, pi]
n_phi = 128 * 2  # bins in azimuth, phi in [0, 2 pi]

# Worker count when --workers is omitted and SLURM_CPUS_PER_TASK is unset.
WORKERS = 12


def fermi_dirac_occupation(energy_MeV, mu_MeV, temperature_MeV):
    """Occupation [exp((E-mu)/T)+1]^(-1), stable for a cold absorber."""
    exponent = (energy_MeV - mu_MeV) / temperature_MeV
    if exponent >= 700.0:
        return 0.0
    if exponent > 60.0:
        return float(np.exp(-exponent))
    return float(1.0 / (np.exp(exponent) + 1.0))


def build_solver_inputs():
    """Cell sizes, source occupation, shadow angles, and the direction mesh."""
    global dx, dy, dz, cell_volume
    global center_e, center_a, S_e, d, e_sep, beta_u, beta_p
    global x_internal, x_external
    global phat, vphase_a, n_directions, full_f_bytes

    dx = Lx / ncell[0]
    dy = Ly / ncell[1]
    dz = Lz / ncell[2]
    cell_volume = dx * dy * dz
    center_e = np.array(r_emitter_sphere_center_cm, dtype=np.float64)
    center_a = np.array(r_shadowing_sphere_center_cm, dtype=np.float64)
    S_e = fermi_dirac_occupation(E_MeV, mu_e, T_e)

    separation = center_a - center_e
    d = float(np.linalg.norm(separation))
    e_sep = separation / d
    if not (R_e > R_a > 0.0 and d > R_e + R_a):
        raise ValueError("emitter and absorber are not a disjoint shadow pair")
    beta_u = float(np.arcsin((R_e - R_a) / d))
    beta_p = float(np.arcsin((R_e + R_a) / d))
    x_internal = d * R_e / (R_e + R_a)
    x_external = d * R_e / (R_e - R_a)

    theta_edges = np.linspace(0.0, np.pi, n_theta + 1)
    phi_edges = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    delta_phi = np.diff(phi_edges)
    delta_cos = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
    theta_grid, phi_grid = np.meshgrid(theta_centers, phi_centers, indexing="ij")
    dcos_grid, dphi_grid = np.meshgrid(delta_cos, delta_phi, indexing="ij")
    theta_a = theta_grid.ravel()
    phi_a = phi_grid.ravel()
    solid_angle = (dcos_grid * dphi_grid).ravel()
    phat = np.column_stack(
        (
            np.sin(theta_a) * np.cos(phi_a),
            np.sin(theta_a) * np.sin(phi_a),
            np.cos(theta_a),
        )
    )
    n_directions = int(phat.shape[0])
    energy_lo = E_lo_MeV * 1e6 * CGSUnitsConst.eV
    energy_hi = E_hi_MeV * 1e6 * CGSUnitsConst.eV
    vphase_a = solid_angle * (energy_hi**3 - energy_lo**3) / 3.0
    full_f_bytes = int(np.prod(ncell)) * n_directions * np.dtype(np.float64).itemsize

    if n_directions != n_theta * n_phi:
        raise RuntimeError("direction count does not equal n_theta * n_phi")
    if abs(float(np.sum(solid_angle)) - 4.0 * np.pi) > 1e-10:
        raise RuntimeError("bin solid angles do not sum to 4*pi")
    if np.max(np.abs(np.linalg.norm(phat, axis=1) - 1.0)) > 1e-12:
        raise RuntimeError("mesh directions are not unit vectors")


def output_path():
    """HDF5 path in the directory from which this script was launched."""
    name = f"shadow_casting_analytic_moments_ncell_{ncell[0]}_{ncell[1]}_{ncell[2]}.h5"
    return Path.cwd() / name


def _scientific(value):
    return f"{float(value):.16e}"


def reference_signature():
    """Identity of the stored grid. A mismatch means these inputs would build a different file."""
    payload = {
        "formula": "chord-f-and-isolated-v1",
        "ncell": [int(n) for n in ncell],
        "dx": _scientific(dx),
        "dy": _scientific(dy),
        "dz": _scientific(dz),
        "center_e": [_scientific(value) for value in center_e],
        "center_a": [_scientific(value) for value in center_a],
        "R_e": _scientific(R_e),
        "R_a": _scientific(R_a),
        "kappa_e": _scientific(kappa_e),
        "kappa_a": _scientific(kappa_a),
        "S_e": _scientific(S_e),
        "n_theta": int(n_theta),
        "n_phi": int(n_phi),
        "E_lo_MeV": _scientific(E_lo_MeV),
        "E_hi_MeV": _scientific(E_hi_MeV),
    }
    return json.dumps(payload, sort_keys=True)


def describe_moment_file(path):
    with h5py.File(path, "r") as handle:
        print(f"file {path.resolve()}")
        print(f"  complete = {bool(handle.attrs.get('complete', False))}")
        print(f"  units = {handle.attrs.get('units', '')}")
        print(f"  seconds = {handle.attrs.get('seconds', 'not recorded')}")
        for name in ("n", "F", "n0", "done"):
            dataset = handle[name]
            print(f"  {name}: shape {dataset.shape}, dtype {dataset.dtype}")
        done = np.array(handle["done"])
        density = np.array(handle["n"])
        isolated = np.array(handle["n0"])
        flux = np.array(handle["F"])
        print(f"  done slabs {int(done.sum())}/{done.size}")
        print(f"  n  min/max = {density.min():.6e} / {density.max():.6e} cm^-3")
        print(f"  n0 min/max = {isolated.min():.6e} / {isolated.max():.6e} cm^-3")
        print(f"  |F| max = {np.linalg.norm(flux, axis=-1).max():.6e} cm^-3")
        for group_name in ("grid", "spheres", "radiation", "shadow"):
            if group_name not in handle:
                continue
            print(f"  {group_name}:")
            for name, dataset in handle[group_name].items():
                print(f"    {name}: {dataset.attrs.get('description', '')}")


def open_moment_file(path, signature):
    if path.exists():
        with h5py.File(path, "r") as handle:
            stored = handle.attrs.get("signature")
            complete = bool(handle.attrs.get("complete", False))
        if stored != signature:
            raise RuntimeError(
                f"{path.resolve()} was built with a different geometry or angular mesh. "
                "This script does not replace it."
            )
        return complete
    with h5py.File(path, "w") as handle:
        handle.attrs["signature"] = signature
        handle.attrs["complete"] = False
        handle.attrs["units"] = "cm^-3"
        handle.create_dataset("n", shape=tuple(ncell), dtype="f8")
        handle.create_dataset("F", shape=tuple(ncell) + (3,), dtype="f8")
        handle.create_dataset("n0", shape=tuple(ncell), dtype="f8")
        handle.create_dataset("done", shape=(ncell[0],), dtype="u1")
    return False


def _put(group, name, data, unit, description):
    data = np.asarray(data)
    if name in group:
        del group[name]
    dataset = group.create_dataset(name, data=data)
    dataset.attrs["unit"] = unit
    dataset.attrs["description"] = description


def write_file_record(handle):
    """Store the mesh and the inputs next to n, F, and n0."""
    handle.attrs["readme"] = (
        "Reference moments of the chord solution at cell centers. "
        "n and n0 are number densities in cm^-3. "
        "F is the number flux divided by c, with components (Fx, Fy, Fz), in cm^-3. "
        "n0 is the same sum with the absorber removed. "
        "Index order is (x, y, z). "
        "Cell centers are grid/x_cm, grid/y_cm, and grid/z_cm. "
        "The domain is [0, Lx] x [0, Ly] x [0, Lz]."
    )
    handle.attrs["indexing"] = "ij, axis 0 = x, axis 1 = y, axis 2 = z"
    for name, description in (
        ("n", "number density at cell centers"),
        ("F", "number flux divided by c; last axis is (Fx, Fy, Fz)"),
        ("n0", "number density of the isolated emitter; the absorber is removed"),
    ):
        handle[name].attrs["unit"] = "cm^-3"
        handle[name].attrs["description"] = description
        handle[name].attrs["indexing"] = "ij, axis 0 = x, axis 1 = y, axis 2 = z"
    handle["done"].attrs["description"] = "1 when that x-slab has been written"

    grid = handle.require_group("grid")
    _put(grid, "ncell", np.asarray(ncell, dtype=np.int64), "count", "number of cells (Nx, Ny, Nz)")
    _put(grid, "cell_size_cm", (dx, dy, dz), "cm", "cell widths (dx, dy, dz)")
    _put(grid, "domain_size_cm", (Lx, Ly, Lz), "cm", "domain widths (Lx, Ly, Lz)")
    _put(grid, "domain_origin_cm", (0.0, 0.0, 0.0), "cm", "lower corner of the domain")
    _put(grid, "x_cm", (np.arange(ncell[0]) + 0.5) * dx, "cm", "x of cell centers, (i + 0.5) * dx")
    _put(grid, "y_cm", (np.arange(ncell[1]) + 0.5) * dy, "cm", "y of cell centers, (j + 0.5) * dy")
    _put(grid, "z_cm", (np.arange(ncell[2]) + 0.5) * dz, "cm", "z of cell centers, (k + 0.5) * dz")
    _put(grid, "cell_volume_cm3", cell_volume, "cm^3", "volume of one cell")

    spheres = handle.require_group("spheres")
    _put(spheres, "center_emitter_cm", center_e, "cm", "emitter center (x, y, z)")
    _put(spheres, "center_absorber_cm", center_a, "cm", "absorber center (x, y, z)")
    _put(spheres, "radius_emitter_cm", R_e, "cm", "emitter radius")
    _put(spheres, "radius_absorber_cm", R_a, "cm", "absorber radius")
    _put(spheres, "separation_cm", d, "cm", "distance between the sphere centers")
    _put(spheres, "separation_axis", e_sep, "1", "unit vector from the emitter center to the absorber center")

    radiation = handle.require_group("radiation")
    _put(radiation, "kappa_emitter_per_cm", kappa_e, "cm^-1", "emitter absorption opacity")
    _put(radiation, "kappa_absorber_per_cm", kappa_a, "cm^-1", "absorber absorption opacity")
    _put(radiation, "kappa_vacuum_per_cm", kappa_vacuum, "cm^-1", "vacuum opacity recorded from the run; not used in f")
    _put(radiation, "mu_emitter_MeV", mu_e, "MeV", "emitter chemical potential used for the source occupation")
    _put(radiation, "temperature_emitter_MeV", T_e, "MeV", "emitter temperature used for the source occupation")
    _put(radiation, "mu_absorber_MeV", mu_a, "MeV", "absorber chemical potential; the absorber source is zero")
    _put(radiation, "temperature_absorber_MeV", T_a, "MeV", "absorber temperature; the absorber source is zero")
    _put(radiation, "mu_vacuum_MeV", mu_vacuum, "MeV", "vacuum chemical potential; not used in f")
    _put(radiation, "temperature_vacuum_MeV", T_vacuum, "MeV", "vacuum temperature; not used in f")
    _put(radiation, "source_occupation", S_e, "1", "Fermi-Dirac occupation of the emitter at energy_MeV")
    _put(radiation, "energy_MeV", E_MeV, "MeV", "energy used for the emitter occupation")
    _put(radiation, "energy_bin_lo_MeV", E_lo_MeV, "MeV", "lower edge of the energy bin")
    _put(radiation, "energy_bin_hi_MeV", E_hi_MeV, "MeV", "upper edge of the energy bin")
    _put(radiation, "n_theta", int(n_theta), "count", "polar bins, theta in [0, pi]")
    _put(radiation, "n_phi", int(n_phi), "count", "azimuthal bins, phi in [0, 2 pi]")
    _put(radiation, "n_directions", int(n_directions), "count", "n_theta * n_phi")

    shadow = handle.require_group("shadow")
    _put(shadow, "beta_umbra_rad", beta_u, "rad", "umbra cone half-angle")
    _put(shadow, "beta_penumbra_rad", beta_p, "rad", "penumbra cone half-angle")
    _put(shadow, "tip_cm", x_external, "cm", "axial position of the umbra tip, measured from the emitter center")
    _put(
        shadow,
        "internal_similitude_cm",
        x_internal,
        "cm",
        "axial position of the internal center of similitude, measured from the emitter center",
    )


def store_x_slab(path, i0, number, flux, isolated):
    with h5py.File(path, "a") as handle:
        handle["n"][i0 : i0 + number.shape[0]] = number
        handle["F"][i0 : i0 + flux.shape[0]] = flux
        handle["n0"][i0 : i0 + isolated.shape[0]] = isolated
        handle["done"][i0 : i0 + number.shape[0]] = 1
        write_file_record(handle)
        handle.flush()


def fill_reference_moments(path, workers):
    """Write n, F, and n0 for every unfinished x-slab. One slab is one cell thick in x."""
    with h5py.File(path, "r") as handle:
        done = np.array(handle["done"])
    pending = [(i, i + 1) for i in range(ncell[0]) if done[i] == 0]
    if not pending:
        with h5py.File(path, "a") as handle:
            handle.attrs["complete"] = True
        return
    workers = max(1, min(int(workers), os.cpu_count() or 1, len(pending)))
    print(
        f"filling {len(pending)} x-slabs, {n_directions} directions, {workers} workers",
        flush=True,
    )
    started = time.perf_counter()
    finished = 0

    def keep(result):
        nonlocal finished
        i0, number, flux, isolated = result
        store_x_slab(path, i0, number, flux, isolated)
        finished += 1
        if finished == 1 or finished % 16 == 0 or finished == len(pending):
            print(f"  slabs {finished}/{len(pending)} in {time.perf_counter() - started:.1f}s", flush=True)

    if workers == 1:
        for bounds in pending:
            keep(reference_moments_x_slab_task(bounds))
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            for result in pool.imap_unordered(reference_moments_x_slab_task, pending, chunksize=1):
                keep(result)
    with h5py.File(path, "a") as handle:
        if int(np.count_nonzero(handle["done"][:])) != ncell[0]:
            raise RuntimeError("reference moment file is missing slabs")
        handle.attrs["complete"] = True
        handle.attrs["seconds"] = time.perf_counter() - started
    print(f"wrote {path.resolve()}")


def check_stored_cells(path):
    analytic_solution.configure(globals())
    check_cells = ((32, 32, 32), (10, 32, 32), (96, 32, 32))
    with h5py.File(path, "r") as handle:
        for i, j, k in check_cells:
            point = np.array([(i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz])
            number, flux, isolated = reference_moments_on_points(point)
            stored_n = float(handle["n"][i, j, k])
            stored_n0 = float(handle["n0"][i, j, k])
            stored_f = np.array(handle["F"][i, j, k])
            scale = max(abs(stored_n), float(np.max(np.abs(stored_f))), 1.0)
            if not np.allclose(number[0], stored_n, rtol=1e-12, atol=0.0):
                raise AssertionError(f"n at {(i, j, k)} does not match the file")
            if not np.allclose(isolated[0], stored_n0, rtol=1e-12, atol=0.0):
                raise AssertionError(f"n0 at {(i, j, k)} does not match the file")
            if not np.allclose(flux[0], stored_f, rtol=0.0, atol=1e-9 * scale):
                raise AssertionError(f"F at {(i, j, k)} does not match the file")
            print(f"cell {(i, j, k)} matches the file: n = {number[0]:.6e} cm^-3")


def worker_count(requested):
    if requested is not None:
        return int(requested)
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return int(WORKERS)


def main():
    parser = argparse.ArgumentParser(description="Write reference moments in the current directory.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="worker processes (default: SLURM_CPUS_PER_TASK, otherwise WORKERS in this file)",
    )
    args = parser.parse_args()
    build_solver_inputs()
    path = output_path()
    print(f"output {path.resolve()}", flush=True)
    print(
        f"ncell = {ncell}, n_theta = {n_theta}, n_phi = {n_phi}, n_directions = {n_directions}",
        flush=True,
    )
    print(f"workers = {worker_count(args.workers)}", flush=True)
    signature = reference_signature()
    already_complete = open_moment_file(path, signature)
    if not already_complete:
        analytic_solution.configure(globals())
        fill_reference_moments(path, worker_count(args.workers))
    else:
        print("file is complete and matches these inputs; the slabs are not recomputed")
    with h5py.File(path, "a") as handle:
        write_file_record(handle)
    describe_moment_file(path)
    check_stored_cells(path)


if __name__ == "__main__":
    main()
