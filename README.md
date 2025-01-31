This pachage was created using UV

Install `uv run pip install .`

To crop droplets, run: `bactos-3d-crop TRITC-3D.zarr/0/ droplets.csv TRITC-3D-crops-300px.zarr/0/`

To localize 3d, run: `bactos-3d-loc TRITC-3D-crops-300px.zarr/0/ TRITC-3D-crops-300px.zarr/locs/`

To count localizations, run:
`bactos-3d-count TRITC-3D-crops-300px.zarr`

