import os
import numpy as np
import bigfish.detection as detection
import dask.array as da
from tqdm import tqdm
import tifffile as tf
from fire import Fire
from pathlib import Path


def loc_recursive(
    crops_zarr: Path,
    prefix: Path,
    results_folder="locs",
    scale: tuple = (5, 0.65, 0.65),
    psf: tuple = (7.5, 1.3, 1.3),
):
    out = os.path.join(prefix, results_folder)
    os.makedirs(out, exist_ok=True)
    crops_dask = da.from_zarr(crops_zarr)
    _loc_recursive(
        crops_dask,
        scale=scale,
        pfs=psf,
        axes=[],
        prefix=out
    )


def loc3d(
    stack3d, scale, psf, threshold=2.2, alpha=0.7, beta=1, gamma=5, **kwargs
):
    data = stack3d.compute()
    spots = detection.detect_spots(
        images=data,
        threshold=threshold,
        voxel_size=scale,  # in nanometer (one value per dimension zyx)
        spot_radius=psf,
    )  # in nanometer (one value per dimension zyx)

    try:
        spots_post_decomposition, dense_regions, reference_spot = (
            detection.decompose_dense(
                image=data,
                spots=spots,
                voxel_size=scale,
                spot_radius=psf,
                alpha=alpha,  # alpha impacts the number of spots per candidate region
                beta=beta,  # beta impacts the number of candidate regions to decompose
                gamma=gamma,
            )
        )  # gamma the filtering step to denoise the image
    except RuntimeError:
        spots_post_decomposition, dense_regions = spots, []
    return {
        "spots": spots,
        "spots_post_decomposition": spots_post_decomposition,
        "dense_regions": dense_regions,
    }


def _add_prefix(pref, spots):
    if len(spots) == 0:
        return []
    # print("prefix, spots: ", pref, spots)
    preff = np.array([pref] * len(spots))
    # print(preff)
    return np.hstack((preff, spots))


def _loc_recursive(stack, scale, pfs, axes=[], prefix=""):
    tables = []
    # print("axes: ", axes)
    if stack.ndim > 3:
        for i, s in enumerate(tqdm(stack)):
            out = _loc_recursive(
                s, scale=scale, pfs=pfs, axes=axes + [i], prefix=prefix
            )
            if len(out["spots"]) > 0:
                tables.append(out)
        try:
            return {k: np.vstack([t[k] for t in tables]) for k in tables[0]}
        except (ValueError, IndexError):
            # print(tables)
            return {"spots": [], "spots_post_decomposition": [], "dense_regions": []}
    try:
        pref = os.path.join(prefix, f"{'_'.join(map(str, axes))}")
        spots_path = pref + "_spots.tif"
        if os.path.exists(spots_path):
            print("skip, already exists: ", spots_path)
            return {"spots": [], "spots_post_decomposition": [], "dense_regions": []}
        out = loc3d(stack, scale=scale, psf=pfs)
        data = {k: _add_prefix(axes, v) for k, v in out.items()}
        if len(d := data["spots"]):
            tf.imwrite((spots_path), d.astype("uint16"))
        if len(d := data["spots_post_decomposition"]):
            tf.imwrite(pref + "_spots_decomp.tif", d.astype("uint16"))
        if len(d := data["dense_regions"]):
            tf.imwrite(pref + "_dense_regions.tif", d.astype("uint16"))
        return data
    except ValueError as e:
        print(f"ValueError: {e}\n axes, out: ", axes, out)
        return {"spots": [], "spots_post_decomposition": [], "dense_regions": []}


def main():
    Fire(loc_recursive)
