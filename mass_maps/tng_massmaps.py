"""
Generate stellar mass surface density maps for IllustrisTNG subhaloes at user-specified
camera orientations (inclination + azimuth), matching the example arrays:

    cams  = np.array(['v0','v1','v2','v3'])
    incls = [109.5, 109.5, 109.5, 0.0]   # degrees
    azims = [0.0, 120.0, -120.0, 0.0]    # degrees

This script uses the `illustris_python` helpers to load local TNG data.
It will save both FITS and PNG maps for each view.

Dependencies:
    - illustris_python (pip install illustris_python)
    - numpy, matplotlib, astropy

Notes on units:
    - Particle coordinates from TNG snapshots are in comoving kpc / h.
    - Particle masses are in 1e10 Msun / h.
    - We convert to physical kpc and Msun using snapshot header (z, h).

Rotation convention:
    - We apply an azimuthal rotation about +Z by `azim` degrees (Rz),
      then an inclination/pitch about +X by `incl` degrees (Rx).
    - After rotation, we project along the +Z axis onto the XY plane.

Outputs:
    - FITS:  <outdir>/tng<SIM>_snap<SN>_sub<SH>_<cam>_massmap.fits
    - PNG:   <outdir>/tng<SIM>_snap<SN>_sub<SH>_<cam>_massmap.png
    - Pixel units: Msun / kpc^2. Header stores pixel scale (kpc/pix).
"""
from __future__ import annotations
import os
import math
from typing import Sequence, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

try:
    from illustris_python import groupcat as il_group
    from illustris_python import snapshot as il_snap
except Exception as e:
    raise ImportError(
        "illustris_python not found. Install with `pip install illustris_python` and ensure TNG data is locally available."
    ) from e


# ------------------------------
# Core utilities
# ------------------------------

def _load_stars(base_path: str, snapnum: int, subhalo_id: int):
    """Load stellar particle positions and masses for a subhalo.

    Returns
    -------
    coords_ckpch : (N,3) float array
        Coordinates in comoving kpc / h (TNG native units)
    mass_1e10Msunh : (N,) float array
        Masses in 1e10 Msun / h
    subpos_ckpch : (3,) float array
        Subhalo center position in comoving kpc / h
    z : float
        Snapshot redshift
    h : float
        Hubble parameter (h)
    """
    hdr = il_group.loadHeader(base_path, snapnum)
    z = hdr["Redshift"]
    h = hdr["HubbleParam"]

    subtab = il_group.loadSubhalos(base_path, snapnum, fields=["SubhaloPos"])  # ckpc/h
    subpos_ckpch = subtab["SubhaloPos"][subhalo_id]

    fields = ["Coordinates", "Masses", "GFM_StellarFormationTime"]
    star = il_snap.loadSubhalo(base_path, snapnum, subhalo_id, partType=4, fields=fields)
    coords = star["Coordinates"]  # ckpc/h
    mass = star["Masses"]         # 1e10 Msun/h

    # Filter out wind or non-stellar particles if necessary (formation time > 0 for stars)
    if "GFM_StellarFormationTime" in star:
        form = star["GFM_StellarFormationTime"]
        mask = form > 0
        coords = coords[mask]
        mass = mass[mask]

    return coords, mass, subpos_ckpch, z, h


def _ckpch_to_phys_kpc(x_ckpch: np.ndarray, z: float, h: float) -> np.ndarray:
    """Convert comoving kpc/h to physical kpc."""
    return x_ckpch / ((1.0 + z) * h)


def _mass_to_Msun(m_1e10Msunh: np.ndarray, h: float) -> np.ndarray:
    return m_1e10Msunh * 1.0e10 / h


def _rotation_matrices(azim_deg: float, incl_deg: float) -> np.ndarray:
    """Return rotation matrix R = Rx(incl) @ Rz(azim)."""
    az = np.deg2rad(azim_deg)
    inc = np.deg2rad(incl_deg)

    Rz = np.array([
        [ np.cos(az), -np.sin(az), 0.0],
        [ np.sin(az),  np.cos(az), 0.0],
        [        0.0,         0.0, 1.0],
    ])
    Rx = np.array([
        [1.0,       0.0,        0.0],
        [0.0,  np.cos(inc), -np.sin(inc)],
        [0.0,  np.sin(inc),  np.cos(inc)],
    ])
    return Rx @ Rz


def _hist2d_mass(xy_kpc: np.ndarray, m_Msun: np.ndarray, img_size_kpc: float, npix: int) -> Tuple[np.ndarray, float]:
    """Accumulate mass into a 2D histogram (surface density map).

    Parameters
    ----------
    xy_kpc : (N, 2)
        XY positions in kpc after rotation and centering.
    m_Msun : (N,)
        Stellar particle masses in Msun.
    img_size_kpc : float
        Total image width/height in kpc (square FOV).
    npix : int
        Image resolution in pixels (npix x npix).

    Returns
    -------
    sig_Msun_per_kpc2 : (npix, npix) array
        Surface density map in Msun/kpc^2.
    pixscale_kpc : float
        Pixel scale in kpc/pix.
    """
    half = img_size_kpc / 2.0
    pixscale = img_size_kpc / npix

    H, xedges, yedges = np.histogram2d(
        xy_kpc[:, 0], xy_kpc[:, 1],
        bins=[npix, npix],
        range=[[-half, half], [-half, half]],
        weights=m_Msun,
    )
    # H currently sums mass per bin (Msun). Divide by area per pixel (kpc^2) to get surface density.
    area = (pixscale * pixscale)  # kpc^2
    sig = H / area
    return sig.T, pixscale  # transpose so that [row, col] matches image convention


def stellar_mass_map(
    base_path: str,
    snapnum: int,
    subhalo_id: int,
    incl_deg: float,
    azim_deg: float,
    *,
    img_size_kpc: float = 50.0,
    npix: int = 512,
    recentre_on_com: bool = True,
) -> Tuple[np.ndarray, float]:
    """Compute a stellar mass surface density map for a given subhalo and view.

    Returns (map, pixscale_kpc).
    """
    coords_ckpch, mass_1e10Msunh, subpos_ckpch, z, h = _load_stars(base_path, snapnum, subhalo_id)

    # Convert to physical units
    coords_kpc = _ckpch_to_phys_kpc(coords_ckpch, z, h)
    subpos_kpc = _ckpch_to_phys_kpc(subpos_ckpch, z, h)
    m_Msun = _mass_to_Msun(mass_1e10Msunh, h)

    # Centering
    pts = coords_kpc - subpos_kpc

    if recentre_on_com and pts.size:
        com = np.average(pts, axis=0, weights=m_Msun)
        pts = pts - com

    # Rotate: R = Rx(incl) @ Rz(az)
    R = _rotation_matrices(azim_deg, incl_deg)
    pts_rot = pts @ R.T

    # Project along +Z -> use XY plane
    xy = pts_rot[:, :2]

    # Accumulate into 2D surface density map
    sig, pixscale = _hist2d_mass(xy, m_Msun, img_size_kpc=img_size_kpc, npix=npix)
    return sig, pixscale


def save_fits_and_png(
    sig: np.ndarray,
    pixscale_kpc: float,
    outbase: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "magma",
    add_log_stretch: bool = True,
):
    """Save FITS + PNG for a surface density map (Msun/kpc^2)."""
    # FITS
    hdu = fits.PrimaryHDU(sig.astype(np.float32))
    hdr = hdu.header
    hdr["BUNIT"] = "Msun/kpc^2"
    hdr["PIXSCALE"] = (pixscale_kpc, "kpc/pixel")
    fits_path = f"{outbase}.fits"
    hdu.writeto(fits_path, overwrite=True)

    # PNG (optionally log stretch)
    img = sig
    if add_log_stretch:
        # avoid log(0)
        img = np.log10(np.clip(sig, a_min=np.maximum(1e-2, sig[sig>0].min(initial=1e-2)), a_max=None))

    plt.figure(figsize=(5, 5), dpi=140)
    extent = np.array([-sig.shape[1]/2, sig.shape[1]/2, -sig.shape[0]/2, sig.shape[0]/2]) * pixscale_kpc
    plt.imshow(img, origin="lower", extent=extent, cmap=cmap)
    plt.xlabel("x [kpc]")
    plt.ylabel("y [kpc]")
    if add_log_stretch:
        plt.title("log10 Sigma*  [Msun/kpc^2]")
    else:
        plt.title("Sigma*  [Msun/kpc^2]")
    if vmin is not None or vmax is not None:
        plt.clim(vmin, vmax)
    cbar = plt.colorbar()
    if add_log_stretch:
        cbar.set_label("log10 Msun/kpc^2")
    else:
        cbar.set_label("Msun/kpc^2")
    png_path = f"{outbase}.png"
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    return fits_path, png_path


# ------------------------------
# Batch driver for the user's example camera set
# ------------------------------

def run_batch(
    base_path: str,
    snapnum: int,
    subhalo_id: int,
    cams: Sequence[str] = ("v0", "v1", "v2", "v3"),
    incls: Sequence[float] = (109.5, 109.5, 109.5, 0.0),
    azims: Sequence[float] = (0.0, 120.0, -120.0, 0.0),
    *,
    img_size_kpc: float = 50.0,
    npix: int = 512,
    outdir: str = "./massmaps",
    tag: str = "TNG",
    recentre_on_com: bool = True,
) -> Dict[str, Tuple[str, str]]:
    """Generate maps for all views and save to disk.

    Returns a dict cam -> (fits_path, png_path).
    """
    os.makedirs(outdir, exist_ok=True)
    results = {}
    for cam, inc, az in zip(cams, incls, azims):
        sig, pix = stellar_mass_map(
            base_path, snapnum, subhalo_id, inc, az,
            img_size_kpc=img_size_kpc, npix=npix, recentre_on_com=recentre_on_com,
        )
        outbase = os.path.join(outdir, f"{tag}_snap{snapnum:03d}_sub{subhalo_id:06d}_{cam}")
        fits_path, png_path = save_fits_and_png(sig, pix, outbase)
        results[cam] = (fits_path, png_path)
        print(f"Saved {cam}: {fits_path} | {png_path}")
    return results


# ------------------------------
# Example usage (edit these paths/IDs to your setup)
# ------------------------------
if __name__ == "__main__":
    # Example configuration — EDIT to your environment
    BASE = "/virgotng/universe/IllustrisTNG/TNG100-1/output"   # or TNG50-1, TNG300-1, etc.
    SNAP = 99                     # e.g. z ~ 0 snapshot, adjust as needed
    SUBID = 123456                # the SubhaloID of the galaxy of interest

    cams = np.array(['v0','v1','v2','v3'])
    incls = [109.5, 109.5, 109.5, 0.0]
    azims = [0.0, 120.0, -120.0, 0.0]

    run_batch(
        base_path=BASE,
        snapnum=SNAP,
        subhalo_id=SUBID,
        cams=cams,
        incls=incls,
        azims=azims,
        img_size_kpc=60.0,   # widen if galaxy is large
        npix=512,
        outdir="/u/mhuertas/data/hsc",
        tag="TNG100-1",
        recentre_on_com=True,
    )
