import numpy as np
import dask.array as da
import pandas as pd
import zarr
from tqdm import tqdm
from fire import Fire


def extract_crops(
    tritc_3d_zarr,
    droplets_csv,
    output_zarr,
    crop_size=300,
    n_frames=55,
    n_droplets=500,
    n_chips=2,
    n_zplanes=25
):

    dask_array = da.from_zarr(tritc_3d_zarr)
    droplets_df = pd.read_csv(droplets_csv, index_col=0)
    # Create empty zarr array with the final dimensions
    zzz = zarr.open(
        output_zarr,
        mode='w', 
        shape=(n_chips, n_droplets, n_frames, n_zplanes, crop_size, crop_size),
        chunks=(1, 50, 1, n_zplanes, crop_size, crop_size),
        dtype='u2'
    )

    d = crop_size // 2
    assert len(dask_array) == n_frames
    for f, frame in enumerate(tqdm(dask_array, desc="frame")):
        zslices = []
        assert len(frame) == n_zplanes

        for z, zplane in enumerate(tqdm(frame, desc="z")):
            crops = [], []
            zslice = zplane.compute()
            assert len(droplets_df.values) == n_chips * n_droplets
            assert len(droplets_df.values[0]) == 3  # chip, y ,x
            for droplet in droplets_df.values:
                chip, y, x = map(int, droplet)
                crop = zslice[chip, y-d:y+d, x-d:x+d]
                crops[chip].append(crop)
            zslices.append(crops)
        data = np.array(zslices).transpose(1, 2, 0, 3, 4)
        if f == 0:
            print(data.shape)
        for c, crop in enumerate(data):
            zzz[c, :, f] = crop


def main():
    Fire(extract_crops)
