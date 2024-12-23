import dask.array as da
import pandas as pd
import numpy as np
import zarr
from dask.diagnostics import ProgressBar
from fire import Fire


def extract_crops(dask_array, droplets_df, output_zarr, crop_size=300):
    # Create empty zarr array with the final dimensions
    store = zarr.open(output_zarr, mode='w')
    output_shape = (2, 500, 55, 25, crop_size, crop_size)
    chunks = (1, 100, 55, 25, crop_size, crop_size)  # Adjust chunks as needed
    zarr_array = store.create_dataset('crops', shape=output_shape, 
                                    chunks=chunks, dtype=dask_array.dtype)
    
    half_size = crop_size // 2
    
    # Process each chip
    for chip in range(2):
        # Get coordinates for current chip
        chip_coords = droplets_df[droplets_df['chip'] == chip]
        
        if len(chip_coords) < 500:
            raise ValueError(f"Not enough coordinates for chip {chip}")
        
        # Take first 500 coordinates
        chip_coords = chip_coords.head(500)
        
        # Process crops in batches
        batch_size = 100  # Adjust based on memory constraints
        for batch_idx in range(0, 500, batch_size):
            batch_coords = chip_coords.iloc[batch_idx:batch_idx + batch_size]
            
            # Extract crops for this batch
            batch_crops = []
            for _, row in batch_coords.iterrows():
                y_center, x_center = int(row['y']), int(row['x'])
                
                # Calculate crop boundaries
                y_start = max(0, y_center - half_size)
                y_end = min(dask_array.shape[3], y_center + half_size)
                x_start = max(0, x_center - half_size)
                x_end = min(dask_array.shape[4], x_center + half_size)
                
                # Extract crop
                crop = dask_array[:, :, chip, y_start:y_end, x_start:x_end]
                
                # Pad if necessary
                if crop.shape[-2:] != (crop_size, crop_size):
                    pad_width = [
                        (0, 0),  # time
                        (0, 0),  # z
                        (max(0, half_size - y_center), 
                         max(0, y_center + half_size - dask_array.shape[3])),
                        (max(0, half_size - x_center),
                         max(0, x_center + half_size - dask_array.shape[4]))
                    ]
                    crop = da.pad(crop, pad_width, mode='constant')
                
                batch_crops.append(crop)
            
            # Stack batch crops
            batch_array = da.stack(batch_crops)
            
            # Compute batch and write to zarr
            with ProgressBar():
                computed_batch = batch_array.compute()
                
            # Write to zarr array
            slice_idx = slice(batch_idx, batch_idx + len(batch_coords))
            zarr_array[chip, slice_idx, :, :, :, :] = computed_batch
            
    return zarr_array

def main(tritc_3d_zarr, droplets_csv, crop_out_zarr, crop_size=300):
    dask_array = da.from_zarr(tritc_3d_zarr)
    droplets_df = pd.read_csv(droplets_csv, index_col=0)
    extract_crops(dask_array=dask_array, droplets_df=droplets_df, output_zarr=crop_out_zarr, crop_size=crop_size)


if __name__ == "__main__":
    Fire(main)