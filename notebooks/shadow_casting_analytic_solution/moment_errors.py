"""Regional errors of deposited n and F against given reference moments.

The reference fields n, F, and n_0 are read from an existing file.
This module does not evaluate the distribution or the chord formula.
Absolute norms use only the two moment fields. Normalized norms divide
by the isolated-emitter density n_0.
"""
from pathlib import Path

import h5py
import numpy as np

from phys_const import PhysConst

LABEL_NAMES = {
    0: "unobstructed",
    1: "umbra",
    2: "penumbra",
    3: "antumbra",
    4: "boundary",
    5: "inside emitter",
    6: "inside absorber",
}
PRIMARY_ORDER = (
    "umbra",
    "penumbra",
    "antumbra",
    "unobstructed",
    "boundary",
    "inside emitter",
    "inside absorber",
)
CORE_ORDER = (
    "umbra core",
    "penumbra core",
    "antumbra core",
    "unobstructed core",
)

def geometry_record(center_e, center_a, R_e, R_a, dx, dy, dz):
    """Cone geometry used to assign cells to shadow regions."""
    center_e = np.asarray(center_e, dtype=np.float64)
    center_a = np.asarray(center_a, dtype=np.float64)
    separation = center_a - center_e
    distance = float(np.linalg.norm(separation))
    if not (R_e > R_a > 0.0 and distance > R_e + R_a):
        raise ValueError("emitter and absorber are not a disjoint shadow pair")
    e_sep = separation / distance
    beta_u = float(np.arcsin((R_e - R_a) / distance))
    beta_p = float(np.arcsin((R_e + R_a) / distance))
    return {
        "center_e": center_e,
        "center_a": center_a,
        "R_e": float(R_e),
        "R_a": float(R_a),
        "d": distance,
        "e_sep": e_sep,
        "dx": float(dx),
        "dy": float(dy),
        "dz": float(dz),
        "beta_u": beta_u,
        "beta_p": beta_p,
        "x_internal": distance * R_e / (R_e + R_a),
        "x_external": distance * R_e / (R_e - R_a),
    }


def _surface_hit(cylindrical, surface, absolute=False):
    scale = np.abs(surface) if absolute else surface
    return np.abs(cylindrical - surface) <= 1e-6 * np.maximum(scale, 1.0)


def classify_positions(points, geom):
    """Opaque-limit labels for points of shape (..., 3).

    Cone hits within the floating-point guard are label 4 and are excluded
    from the open umbra, penumbra, and antumbra. Sphere surfaces belong to
    the sphere. Absorber wins if a point is inside both balls.
    """
    points = np.asarray(points, dtype=np.float64)
    center_e = geom["center_e"]
    center_a = geom["center_a"]
    relative = points - center_e
    radial_e = np.linalg.norm(relative, axis=-1)
    radial_a = np.linalg.norm(points - center_a, axis=-1)
    inside_absorber = radial_a <= geom["R_a"]
    inside_emitter = radial_e <= geom["R_e"]
    axial = np.sum(relative * geom["e_sep"], axis=-1)
    transverse = relative - axial[..., None] * geom["e_sep"]
    varpi = np.linalg.norm(transverse, axis=-1)
    varpi_u = (geom["x_external"] - axial) * np.tan(geom["beta_u"])
    varpi_up = (axial - geom["x_external"]) * np.tan(geom["beta_u"])
    varpi_p = (axial - geom["x_internal"]) * np.tan(geom["beta_p"])
    axial_umbra_start = geom["d"] + geom["R_a"] * np.sin(geom["beta_u"])
    axial_shadow_start = geom["d"] - geom["R_a"] * np.sin(geom["beta_p"])
    exterior = ~inside_absorber & ~inside_emitter
    umbra_axial = (axial >= axial_umbra_start) & (axial <= geom["x_external"])
    antumbra_axial = axial > geom["x_external"]
    penumbra_axial = axial >= axial_shadow_start
    on_umbra = exterior & umbra_axial & _surface_hit(varpi, varpi_u, absolute=False)
    on_antumbra = (
        exterior
        & antumbra_axial
        & _surface_hit(varpi, varpi_up, absolute=True)
        & ~on_umbra
    )
    on_penumbra = (
        exterior
        & penumbra_axial
        & _surface_hit(varpi, varpi_p, absolute=True)
        & ~on_umbra
        & ~on_antumbra
    )
    boundary = on_umbra | on_antumbra | on_penumbra
    umbra = exterior & umbra_axial & (varpi < varpi_u) & ~on_umbra
    antumbra = exterior & antumbra_axial & (varpi < varpi_up) & ~on_antumbra & ~umbra
    penumbra = (
        exterior
        & penumbra_axial
        & (varpi < varpi_p)
        & ~umbra
        & ~antumbra
        & ~boundary
    )
    label = np.zeros(points.shape[:-1], dtype=np.int8)
    label[penumbra] = 2
    label[antumbra] = 3
    label[umbra] = 1
    label[boundary] = 4
    label[inside_emitter & ~inside_absorber] = 5
    label[inside_absorber] = 6
    fields = {
        "axial": axial,
        "varpi": varpi,
        "varpi_u": varpi_u,
        "varpi_up": varpi_up,
        "varpi_p": varpi_p,
        "r_e": radial_e,
        "r_a": radial_a,
        "axial_umbra_start": axial_umbra_start,
        "axial_shadow_start": axial_shadow_start,
    }
    return label, fields


def classify_mesh(xc, yc, zc, geom, slab=32):
    """Label every cell center. Returns label and the cone fields."""
    nx, ny, nz = len(xc), len(yc), len(zc)
    label = np.empty((nx, ny, nz), dtype=np.int8)
    store = {}
    for i0 in range(0, nx, slab):
        i1 = min(nx, i0 + slab)
        xx, yy, zz = np.meshgrid(xc[i0:i1], yc, zc, indexing="ij")
        points = np.stack((xx, yy, zz), axis=-1)
        slab_label, fields = classify_positions(points, geom)
        label[i0:i1] = slab_label
        for key, value in fields.items():
            if np.ndim(value) == 0:
                store[key] = value
                continue
            if key not in store:
                store[key] = np.empty((nx, ny, nz), dtype=np.float64)
            store[key][i0:i1] = value
    counts = [int(np.count_nonzero(label == code)) for code in range(7)]
    if sum(counts) != label.size:
        raise RuntimeError("shadow labels do not partition the mesh")
    return label, store


def buffer_masks(label, fields, geom, delta):
    """Open regions with a cylindrical delta-neighborhood of each surface removed."""
    axial = fields["axial"]
    varpi = fields["varpi"]
    s_x = geom["x_external"]
    s_u0 = fields["axial_umbra_start"]
    s_p0 = fields["axial_shadow_start"]
    inner = np.where(axial <= s_x, fields["varpi_u"], fields["varpi_up"])
    away_from_inner = (axial < s_u0) | (varpi > inner + delta)
    umbra_core = (
        (label == 1)
        & (axial >= s_u0 + delta)
        & (axial <= s_x - delta)
        & (varpi < fields["varpi_u"] - delta)
    )
    antumbra_core = (
        (label == 3) & (axial > s_x + delta) & (varpi < fields["varpi_up"] - delta)
    )
    penumbra_core = (
        (label == 2)
        & (axial > s_p0 + delta)
        & (varpi < fields["varpi_p"] - delta)
        & away_from_inner
        & (fields["r_a"] >= geom["R_a"] + delta)
    )
    outside_cone = (axial < s_p0 - delta) | (varpi > fields["varpi_p"] + delta)
    exterior_core = (
        (label == 0)
        & (fields["r_e"] >= geom["R_e"] + delta)
        & (fields["r_a"] >= geom["R_a"] + delta)
        & outside_cone
    )
    return {
        "umbra core": umbra_core,
        "penumbra core": penumbra_core,
        "antumbra core": antumbra_core,
        "unobstructed core": exterior_core,
    }


def moment_errors(n_num, n_ana, flux_num, flux_ana):
    """Absolute density error and Euclidean flux-vector error."""
    density = np.abs(np.asarray(n_num, dtype=np.float64) - np.asarray(n_ana, dtype=np.float64))
    delta_flux = np.asarray(flux_num, dtype=np.float64) - np.asarray(flux_ana, dtype=np.float64)
    flux = np.sqrt(np.sum(delta_flux * delta_flux, axis=-1))
    return density, flux


def absolute_norms(error, mask):
    """L_inf, volume-weighted mean, and volume-weighted RMS.

    On a uniform mesh the cell volume cancels. An empty mask returns NaN
    and does not reduce an empty selection.
    """
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {"count": 0, "linf": np.nan, "l1": np.nan, "l2": np.nan}
    values = np.asarray(error, dtype=np.float64)[mask]
    return {
        "count": count,
        "linf": float(np.max(values)),
        "l1": float(np.mean(values)),
        "l2": float(np.sqrt(np.mean(values * values))),
    }


def normalized_norms(error, mask, reference):
    """Manuscript ratios against a reference field that stays positive in the umbra.

    L_inf is the maximum of e/reference over cells with reference > 0.
    L_1 and L_2 divide the volume-weighted error by the same power of the
    reference. A non-positive denominator sum returns NaN. No epsilon is added.
    """
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {"count": 0, "linf": np.nan, "l1": np.nan, "l2": np.nan}
    values = np.asarray(error, dtype=np.float64)[mask]
    scale = np.asarray(reference, dtype=np.float64)[mask]
    usable = np.isfinite(scale) & (scale > 0.0)
    if np.any(usable):
        linf = float(np.max(values[usable] / scale[usable]))
    else:
        linf = np.nan
    denom1 = float(np.sum(scale))
    denom2 = float(np.sum(scale * scale))
    l1 = float(np.sum(values) / denom1) if np.isfinite(denom1) and denom1 > 0.0 else np.nan
    l2 = (
        float(np.sqrt(np.sum(values * values) / denom2))
        if np.isfinite(denom2) and denom2 > 0.0
        else np.nan
    )
    return {"count": count, "linf": linf, "l1": l1, "l2": l2}


def named_masks(label):
    return {name: label == code for code, name in LABEL_NAMES.items()}


def domain_interior_mask(ncell, margin):
    """True inside the domain, with `margin` cells removed from every face."""
    margin = int(margin)
    if margin < 0:
        raise ValueError("domain margin must be non-negative")
    interior = np.ones(tuple(ncell), dtype=bool)
    if margin == 0:
        return interior
    for axis, length in enumerate(ncell):
        if 2 * margin >= length:
            raise ValueError(
                f"domain margin {margin} removes axis {axis} of length {length}"
            )
        index = [slice(None)] * len(ncell)
        index[axis] = slice(0, margin)
        interior[tuple(index)] = False
        index[axis] = slice(length - margin, length)
        interior[tuple(index)] = False
    return interior


def run_synthetic_tests():
    """Identical fields, a constant error, a two-cell mix, and an empty mask."""
    base_n = np.ones((4, 4, 4))
    base_f = np.zeros((4, 4, 4, 3))
    full = np.ones((4, 4, 4), dtype=bool)
    density, flux = moment_errors(base_n, base_n, base_f, base_f)
    zero_n = absolute_norms(density, full)
    zero_f = absolute_norms(flux, full)
    if not (zero_n["linf"] == zero_n["l1"] == zero_n["l2"] == 0.0):
        raise AssertionError("identical densities do not have zero error")
    if not (zero_f["linf"] == zero_f["l1"] == zero_f["l2"] == 0.0):
        raise AssertionError("identical fluxes do not have zero error")

    shifted_f = base_f + np.array([0.0, 0.0, 4.0])
    density, flux = moment_errors(base_n + 3.0, base_n, shifted_f, base_f)
    constant_n = absolute_norms(density, full)
    constant_f = absolute_norms(flux, full)
    if not np.allclose([constant_n["linf"], constant_n["l1"], constant_n["l2"]], 3.0):
        raise AssertionError(f"constant density error returned {constant_n}")
    if not np.allclose([constant_f["linf"], constant_f["l1"], constant_f["l2"]], 4.0):
        raise AssertionError(f"constant flux error returned {constant_f}")

    mixed = np.array([0.0, 3.0])
    mixed_norms = absolute_norms(mixed, np.ones(2, dtype=bool))
    if mixed_norms["linf"] != 3.0 or not np.isclose(mixed_norms["l1"], 1.5):
        raise AssertionError(f"two-cell L1/Linf returned {mixed_norms}")
    if not np.isclose(mixed_norms["l2"], np.sqrt(4.5)):
        raise AssertionError(f"two-cell L2 returned {mixed_norms['l2']}")

    empty = absolute_norms(mixed, np.zeros(2, dtype=bool))
    if empty["count"] != 0 or not all(np.isnan(empty[key]) for key in ("linf", "l1", "l2")):
        raise AssertionError(f"empty region returned {empty}")

    ratio = normalized_norms(np.array([0.0, 2.0]), np.ones(2, dtype=bool), np.array([4.0, 4.0]))
    if not np.isclose(ratio["linf"], 0.5) or not np.isclose(ratio["l1"], 0.25):
        raise AssertionError(f"normalized two-cell result returned {ratio}")
    if not np.isclose(ratio["l2"], np.sqrt(0.125)):
        raise AssertionError(f"normalized L2 returned {ratio['l2']}")
    dark = normalized_norms(np.array([0.0, 2.0]), np.ones(2, dtype=bool), np.zeros(2))
    if not all(np.isnan(dark[key]) for key in ("linf", "l1", "l2")):
        raise AssertionError("a zero reference must produce NaN normalized norms")
    still = absolute_norms(np.array([0.0, 2.0]), np.ones(2, dtype=bool))
    if not np.isclose(still["l1"], 1.0):
        raise AssertionError("absolute norms changed when the reference vanished")



def load_numerical_moments(mesh_path, ncell):
    """Active-cell n and F from an EMU mesh plotfile. Units are cm^-3."""
    keys = {
        "n": "N00_Re(1|ccm)",
        "x": "Fx00_Re(1|ccm)",
        "y": "Fy00_Re(1|ccm)",
        "z": "Fz00_Re(1|ccm)",
    }
    with h5py.File(mesh_path, "r") as handle:
        arrays = {name: np.array(handle[key]) for name, key in keys.items()}
        time_s = float(handle["t(s)"][()])
        file_ncell = (
            int(handle["Nx"][()]),
            int(handle["Ny"][()]),
            int(handle["Nz"][()]),
        )
        spacing = (
            float(handle["dx(cm)"][()]),
            float(handle["dy(cm)"][()]),
            float(handle["dz(cm)"][()]),
        )
    if file_ncell != tuple(ncell):
        raise ValueError(f"plotfile ncell {file_ncell} != {tuple(ncell)}")
    ghost = tuple(n + 2 for n in ncell)

    def active(array, name):
        if array.shape == tuple(ncell):
            return array
        if array.shape == ghost:
            return array[(slice(1, -1),) * 3]
        raise ValueError(f"{name} has shape {array.shape}, expected {tuple(ncell)}")

    density = active(arrays["n"], "n")
    flux = np.stack(
        [active(arrays["x"], "Fx"), active(arrays["y"], "Fy"), active(arrays["z"], "Fz")],
        axis=-1,
    )
    if flux.shape != tuple(ncell) + (3,):
        raise ValueError(f"F has shape {flux.shape}")
    if not np.isfinite(density).all() or not np.isfinite(flux).all():
        raise ValueError("numerical moments contain a non-finite value")
    return density, flux, time_s, spacing


def domain_t_star(xc, yc, zc, center_e, radius_e, speed):
    """Latest direct-signal arrival, measured from the emitter center."""
    farthest = np.sqrt(
        np.max((xc - center_e[0]) ** 2)
        + np.max((yc - center_e[1]) ** 2)
        + np.max((zc - center_e[2]) ** 2)
    )
    return float((farthest + radius_e) / speed)


def _fmt(value, width=12):
    if not np.isfinite(value):
        return f"{'nan':>{width}}"
    return f"{value:{width}.4e}"


def format_table(title, rows, normalized=False):
    kind = "dimensionless, divided by n_0" if normalized else "cm^-3"
    header = (
        f"{'region':<20} {'N_cells':>10} {'volume_cm3':>12} "
        f"{'Linf_n':>12} {'L1_n':>12} {'L2_n':>12} "
        f"{'Linf_F':>12} {'L1_F':>12} {'L2_F':>12}"
    )
    lines = [title + f" ({kind})", header]
    for row in rows:
        lines.append(
            f"{row['region']:<20} {row['count']:10d} {_fmt(row['volume'])} "
            f"{_fmt(row['linf_n'])} {_fmt(row['l1_n'])} {_fmt(row['l2_n'])} "
            f"{_fmt(row['linf_F'])} {_fmt(row['l1_F'])} {_fmt(row['l2_F'])}"
        )
    return "\n".join(lines)


def _rows(order, masks, e_n, e_f, cell_volume, reference=None):
    rows = []
    for name in order:
        mask = masks[name]
        density = absolute_norms(e_n, mask) if reference is None else normalized_norms(e_n, mask, reference)
        flux = absolute_norms(e_f, mask) if reference is None else normalized_norms(e_f, mask, reference)
        if density["count"] != flux["count"]:
            raise RuntimeError(f"{name} density and flux masks differ")
        rows.append(
            {
                "region": name,
                "count": density["count"],
                "volume": density["count"] * cell_volume,
                "linf_n": density["linf"],
                "l1_n": density["l1"],
                "l2_n": density["l2"],
                "linf_F": flux["linf"],
                "l1_F": flux["l1"],
                "l2_F": flux["l2"],
            }
        )
    return rows



def save_error_figure(path, e_n, xc, yc, zc, geom, time_s, stationary):
    import sys
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    j_mid = int(np.argmin(np.abs(yc - geom["center_e"][1])))
    k_mid = int(np.argmin(np.abs(zc - geom["center_e"][2])))
    xy = np.log10(np.maximum(e_n[:, :, k_mid], 1e-30))
    xz = np.log10(np.maximum(e_n[:, j_mid, :], 1e-30))
    finite = np.concatenate([xy.ravel(), xz.ravel()])
    finite = finite[np.isfinite(finite)]
    vmin = float(np.percentile(finite, 2))
    vmax = float(np.percentile(finite, 98))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    panels = (
        (axes[0], xy.T, xc, yc, r"$y$", j_mid, yc[j_mid]),
        (axes[1], xz.T, xc, zc, r"$z$", k_mid, zc[k_mid]),
    )
    image = None
    for axis, values, horizontal, vertical, vertical_name, index, coord in panels:
        image = axis.imshow(
            values,
            origin="lower",
            extent=[
                (horizontal[0] - 0.5 * geom["dx"]) / 1e5,
                (horizontal[-1] + 0.5 * geom["dx"]) / 1e5,
                (vertical[0] - 0.5 * (geom["dy"] if vertical is yc else geom["dz"])) / 1e5,
                (vertical[-1] + 0.5 * (geom["dy"] if vertical is yc else geom["dz"])) / 1e5,
            ],
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            aspect="equal",
        )
        for center, radius in (
            (geom["center_e"], geom["R_e"]),
            (geom["center_a"], geom["R_a"]),
        ):
            circle = plt.Circle(
                (center[0] / 1e5, center[1] / 1e5 if vertical_name == r"$y$" else center[2] / 1e5),
                radius / 1e5,
                fill=False,
                color="cyan",
                lw=0.8,
            )
            axis.add_patch(circle)
        axis.set_xlabel(r"$x$ (km)")
        axis.set_ylabel(vertical_name + " (km)")
        axis.set_title(f"index {index}, coordinate {coord / 1e5:.2f} km")
    state = "stationary" if stationary else "before t*"
    fig.suptitle(
        rf"$\log_{{10}}|n_{{\mathrm{{num}}}}-n_{{\mathrm{{ana}}}}|$ at $t={time_s * 1e3:.4f}$ ms ({state})"
    )
    fig.colorbar(image, ax=axes, shrink=0.85, label=r"$\log_{10}(|\Delta n|\,/\,\mathrm{cm}^{-3})$")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    return fig


def _write_table(handle, name, rows, description, normalized):
    """One regional-norm table: one dataset per column, regions in a string column."""
    group = handle.create_group(name)
    group.attrs["description"] = description
    group.attrs["normalized_by_n0"] = bool(normalized)
    unit = "1" if normalized else "cm^-3"
    group.attrs["error_unit"] = unit
    regions = np.array([row["region"] for row in rows], dtype=object)
    group.create_dataset("region", data=regions, dtype=h5py.string_dtype(encoding="utf-8"))
    columns = (
        ("count", "N_cells", "1", np.int64),
        ("volume", "volume_cm3", "cm^3", np.float64),
        ("linf_n", "Linf_n", unit, np.float64),
        ("l1_n", "L1_n", unit, np.float64),
        ("l2_n", "L2_n", unit, np.float64),
        ("linf_F", "Linf_F", unit, np.float64),
        ("l1_F", "L1_F", unit, np.float64),
        ("l2_F", "L2_F", unit, np.float64),
    )
    for key, dataset_name, column_unit, dtype in columns:
        values = np.array([row[key] for row in rows], dtype=dtype)
        dataset = group.create_dataset(dataset_name, data=values)
        dataset.attrs["unit"] = column_unit
    group.attrs["columns"] = "region,N_cells,volume_cm3,Linf_n,L1_n,L2_n,Linf_F,L1_F,L2_F"


def write_moment_error_report(path, e_n, e_f, label, tables, metadata, report_text):
    """Write the regional error report beside the simulation, as one HDF5 file.

    `report` is the same text the notebook prints. The four table groups hold
    those numbers as columns. `e_n`, `e_F`, and `label` are the full mesh.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    legend = ", ".join(f"{code} {name}" for code, name in sorted(LABEL_NAMES.items()))
    with h5py.File(path, "w") as handle:
        handle.attrs["readme"] = (
            "Print the dataset report for the tables: "
            "print(handle['report'].asstr()[()]). "
            "Region names are handle['normalized/region'].asstr()[:]. "
            "Groups absolute, absolute_core, normalized, and normalized_core "
            "store the same rows as columns. "
            "e_n and e_F are full-mesh cell-center errors. "
            "Norms omit domain_margin_cells on every face. "
            "Core tables also omit cells within delta_cm of a shadow surface. "
            f"label: {legend}."
        )
        for key, value in metadata.items():
            if isinstance(value, str):
                handle.attrs[key] = value
            else:
                handle.attrs[key] = value
        handle.attrs["label_names"] = legend
        text = np.array(report_text, dtype=object)
        handle.create_dataset("report", data=text, dtype=h5py.string_dtype(encoding="utf-8"))
        for name, rows, description, normalized in tables:
            _write_table(handle, name, rows, description, normalized)
        density = handle.create_dataset("e_n", data=np.asarray(e_n, dtype=np.float64), compression="gzip")
        density.attrs["unit"] = "cm^-3"
        density.attrs["description"] = "absolute density error |n_num - n_ref| at cell centers"
        density.attrs["indexing"] = "ij, axis 0 = x, axis 1 = y, axis 2 = z"
        flux = handle.create_dataset("e_F", data=np.asarray(e_f, dtype=np.float64), compression="gzip")
        flux.attrs["unit"] = "cm^-3"
        flux.attrs["description"] = "Euclidean error of the number-flux vector divided by c"
        flux.attrs["indexing"] = "ij, axis 0 = x, axis 1 = y, axis 2 = z"
        labels = handle.create_dataset("label", data=np.asarray(label, dtype=np.int8), compression="gzip")
        labels.attrs["description"] = legend
        labels.attrs["indexing"] = "ij, axis 0 = x, axis 1 = y, axis 2 = z"
    return path


def load_reference_moments(path, ncell):
    """Read given n, F, and n_0. This module does not compute them."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"reference moments not found: {path}. This module does not compute them."
        )
    with h5py.File(path, "r") as handle:
        if "complete" in handle.attrs and not bool(handle.attrs["complete"]):
            raise RuntimeError(
                f"reference moment file is incomplete: {path}. This module does not compute them."
            )
        missing = [name for name in ("n", "F", "n0") if name not in handle]
        if missing:
            raise KeyError(f"{path} is missing {missing}")
        density = np.array(handle["n"])
        flux = np.array(handle["F"])
        isolated = np.array(handle["n0"])
    expected = tuple(ncell)
    if density.shape != expected or isolated.shape != expected or flux.shape != expected + (3,):
        raise ValueError(
            f"reference shapes n {density.shape}, F {flux.shape}, n0 {isolated.shape} "
            f"do not match {expected}"
        )
    if (
        not np.isfinite(density).all()
        or not np.isfinite(flux).all()
        or not np.isfinite(isolated).all()
    ):
        raise ValueError(f"reference moments in {path} contain a non-finite value")
    return density, flux, isolated


def compare_deposited_mesh(
    mesh_path,
    moments_path,
    dx,
    dy,
    dz,
    center_e,
    center_a,
    R_e,
    R_a,
    delta_cells=1,
    kernel_half_width_cm=0.0,
    make_figure=True,
    domain_margin=2,
    report_path=None,
):
    """Regional errors of one deposited mesh against given reference moments."""
    run_synthetic_tests()
    geom = geometry_record(center_e, center_a, float(R_e), float(R_a), dx, dy, dz)
    ncell = _ncell_from_mesh(mesh_path)
    with h5py.File(mesh_path, "r") as handle:
        spacing = (
            float(handle["dx(cm)"][()]),
            float(handle["dy(cm)"][()]),
            float(handle["dz(cm)"][()]),
        )
    requested = (geom["dx"], geom["dy"], geom["dz"])
    if spacing != requested:
        raise ValueError(f"plotfile spacing {spacing} != requested {requested}")
    xc = (np.arange(ncell[0]) + 0.5) * geom["dx"]
    yc = (np.arange(ncell[1]) + 0.5) * geom["dy"]
    zc = (np.arange(ncell[2]) + 0.5) * geom["dz"]
    n_ana, f_ana, n0 = load_reference_moments(moments_path, ncell)
    n_num, f_num, time_s, _ = load_numerical_moments(mesh_path, ncell)
    if n_num.shape != ncell or f_num.shape != ncell + (3,):
        raise AssertionError("numerical moment shapes do not match the mesh")

    label, fields = classify_mesh(xc, yc, zc, geom)
    if label.shape != ncell:
        raise AssertionError("label shape does not match the mesh")
    emitter_index = tuple(
        int(np.floor(geom["center_e"][axis] / step))
        for axis, step in enumerate((geom["dx"], geom["dy"], geom["dz"]))
    )
    absorber_index = tuple(
        int(np.floor(geom["center_a"][axis] / step))
        for axis, step in enumerate((geom["dx"], geom["dy"], geom["dz"]))
    )
    if int(label[emitter_index]) != 5 or int(label[absorber_index]) != 6:
        raise AssertionError("sphere centers are not labeled as sphere interiors")

    e_n, e_f = moment_errors(n_num, n_ana, f_num, f_ana)
    masks = named_masks(label)
    delta = delta_cells * max(geom["dx"], geom["dy"], geom["dz"]) + kernel_half_width_cm
    cores = buffer_masks(label, fields, geom, delta)
    for name, mask in cores.items():
        parent = {
            "umbra core": masks["umbra"],
            "penumbra core": masks["penumbra"],
            "antumbra core": masks["antumbra"],
            "unobstructed core": masks["unobstructed"],
        }[name]
        if np.any(mask & ~parent):
            raise AssertionError(f"{name} is not a subset of its open region")

    interior = domain_interior_mask(ncell, domain_margin)
    masks = {name: mask & interior for name, mask in masks.items()}
    cores = {name: mask & interior for name, mask in cores.items()}
    excluded = int(np.count_nonzero(~interior))

    cell_volume = geom["dx"] * geom["dy"] * geom["dz"]
    absolute_rows = _rows(PRIMARY_ORDER, masks, e_n, e_f, cell_volume)
    absolute_core_rows = _rows(CORE_ORDER, cores, e_n, e_f, cell_volume)
    normalized_rows = _rows(PRIMARY_ORDER, masks, e_n, e_f, cell_volume, reference=n0)
    normalized_core_rows = _rows(CORE_ORDER, cores, e_n, e_f, cell_volume, reference=n0)
    margin = int(domain_margin)
    t_star = domain_t_star(
        xc[margin : ncell[0] - margin],
        yc[margin : ncell[1] - margin],
        zc[margin : ncell[2] - margin],
        geom["center_e"],
        geom["R_e"],
        PhysConst.c,
    )
    stationary = time_s >= t_star
    negative = int(np.count_nonzero((n_num < 0.0) & interior))
    analyzed = n_num[interior]

    report_text = "\n".join(
        (
            f"mesh {ncell}, dx = {geom['dx']:.6g} cm, V_c = {cell_volume:.6e} cm^3",
            f"domain margin = {margin} cells on each face, "
            f"excluded {excluded} cells, norms use {int(interior.sum())} cells",
            f"reference moments read from {Path(moments_path).resolve()}",
            f"numerical n negative in {negative} analyzed cells, "
            f"minimum {float(np.min(analyzed)):.6e} cm^-3",
            f"t = {time_s:.6e} s, t* = {t_star:.6e} s, stationary = {stationary}",
            f"buffer delta = {delta:.6g} cm "
            f"({delta_cells} cell widths + kernel half-width {kernel_half_width_cm:.6g} cm)",
            format_table("Absolute errors", absolute_rows, normalized=False),
            format_table(f"Absolute errors, boundary buffer {delta:.3g} cm", absolute_core_rows),
            format_table("Normalized errors, denominator n_0", normalized_rows, normalized=True),
            format_table(
                "Normalized errors, buffered, denominator n_0",
                normalized_core_rows,
                normalized=True,
            ),
        )
    )
    print(report_text)
    figure = None
    figure_path = None
    if make_figure:
        figure_path = (
            Path(__file__).resolve().parents[1]
            / "plots"
            / f"shadow_moment_errors_ncell_{ncell[0]}_{ncell[1]}_{ncell[2]}.png"
        )
        sl = tuple(slice(margin, length - margin) for length in ncell)
        figure = save_error_figure(
            figure_path,
            e_n[sl],
            xc[sl[0]],
            yc[sl[1]],
            zc[sl[2]],
            geom,
            time_s,
            stationary,
        )
        print(f"wrote {figure_path}")
    if report_path is None:
        report_path = Path(mesh_path).resolve().parent / "moment_errors.h5"
    metadata = {
        "mesh_path": str(Path(mesh_path).resolve()),
        "reference_moments": str(Path(moments_path).resolve()),
        "figure_path": "" if figure_path is None else str(Path(figure_path).resolve()),
        "Nx": int(ncell[0]),
        "Ny": int(ncell[1]),
        "Nz": int(ncell[2]),
        "dx_cm": float(geom["dx"]),
        "dy_cm": float(geom["dy"]),
        "dz_cm": float(geom["dz"]),
        "cell_volume_cm3": float(cell_volume),
        "center_e_cm": np.asarray(geom["center_e"], dtype=np.float64),
        "center_a_cm": np.asarray(geom["center_a"], dtype=np.float64),
        "R_e_cm": float(geom["R_e"]),
        "R_a_cm": float(geom["R_a"]),
        "t_s": float(time_s),
        "t_star_s": float(t_star),
        "stationary": int(bool(stationary)),
        "domain_margin_cells": int(margin),
        "n_excluded": int(excluded),
        "n_analyzed": int(interior.sum()),
        "n_negative": int(negative),
        "n_minimum_cm3": float(np.min(analyzed)),
        "delta_cm": float(delta),
        "delta_cells": int(delta_cells),
        "kernel_half_width_cm": float(kernel_half_width_cm),
    }
    tables = (
        (
            "absolute",
            absolute_rows,
            "Open regions. Absolute errors in cm^-3. Domain faces of width domain_margin_cells are omitted.",
            False,
        ),
        (
            "absolute_core",
            absolute_core_rows,
            "Open regions with cells within delta_cm of a shadow surface removed. Absolute errors in cm^-3.",
            False,
        ),
        (
            "normalized",
            normalized_rows,
            "Open regions. Errors divided by the isolated-emitter density n_0. Domain faces are omitted.",
            True,
        ),
        (
            "normalized_core",
            normalized_core_rows,
            "Buffered regions. Errors divided by the isolated-emitter density n_0.",
            True,
        ),
    )
    report_path = write_moment_error_report(
        report_path, e_n, e_f, label, tables, metadata, report_text
    )
    print(f"wrote {report_path}")
    return {
        "e_n": e_n,
        "e_F": e_f,
        "label": label,
        "n_num": n_num,
        "n_ana": n_ana,
        "F_num": f_num,
        "F_ana": f_ana,
        "n0": n0,
        "t": time_s,
        "t_star": t_star,
        "stationary": stationary,
        "delta_cm": delta,
        "absolute": absolute_rows,
        "absolute_core": absolute_core_rows,
        "normalized": normalized_rows,
        "normalized_core": normalized_core_rows,
        "figure": figure,
        "figure_path": figure_path,
        "report_path": report_path,
        "cell_volume": cell_volume,
        "domain_margin": margin,
        "interior": interior,
    }


def _ncell_from_mesh(mesh_path):
    with h5py.File(mesh_path, "r") as handle:
        return (int(handle["Nx"][()]), int(handle["Ny"][()]), int(handle["Nz"][()]))


