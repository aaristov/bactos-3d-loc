import os
import sys
from tqdm import tqdm
import tifffile as tf
import pandas as pd
from fire import Fire
import logging

level = os.environ.get("loglevel", "WARNING")
logging.basicConfig(level=level)
log = logging.getLogger(__name__)

def count(
    prefix_zarr,
    locs_folder="locs",
    n_chips=2,
    n_wells=500,
    n_frames=37,
    spots_decomp_format="{chip}_{well}_{frame}_spots_decomp.tif",
    spots_format="{chip}_{well}_{frame}_spots.tif",
    output="n_cells_spots_decomp.csv"
):
    """counts 3d localizations from tif files and saves csv"""
    path_decomp = os.path.join(prefix_zarr, locs_folder, spots_decomp_format)
    path = os.path.join(prefix_zarr, locs_folder, spots_format)
    log.debug("path format: ", path)
    save_path = os.path.join(prefix_zarr, output)
    with open(save_path, mode="w", encoding='utf8') as f:
        f.write("chip,well,frame,n_spots,n_spots_decomp\n")
        n_cells = []
        for chip in tqdm(range(n_chips), desc="chip"):
            for well in tqdm(range(n_wells), desc="well"):
                for frame in range(n_frames):
                    try:
                        locs = tf.imread(
                            ppp :=
                            path.format(frame=frame, chip=chip, well=well)
                        )
                        locs_decomp = tf.imread(
                            path_decomp.format(
                                frame=frame,
                                chip=chip,
                                well=well
                            )
                        )
                        n_cells.append(
                            {
                                "chip": chip,
                                "well": well,
                                "frame": frame,
                                "n_spots": len(locs),
                                "n_spots_decomp": len(locs_decomp),
                            }
                        )
                        ll, ld = len(locs), len(locs_decomp)
                        f.write(f"{chip},{well},{frame},{ll},{ld}\n")
                    except (FileNotFoundError, ValueError):
                        log.debug("no data: ", ppp)
                        n_cells.append(
                            {
                                "chip": chip,
                                "well": well,
                                "frame": frame,
                                "n_spots": 0,
                                "n_spots_decomp": 0,
                            }
                        )
                        f.write(f"{chip},{well},{frame},0,0\n")
    print("saved: ", save_path)
    n_cells_spots_df = pd.DataFrame(n_cells)
    n_cells_spots_df.to_csv(sp := save_path.replace(".csv", "_df.csv"))
    print("saved: ", sp)


def main():
    Fire(count)


if __name__ == "__main__":
    main()
