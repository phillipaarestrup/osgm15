# osgm15

Read and interpolate **OSGM15** geoid-separation grids from ASCII GTX files.

OSGM15 is the Ordnance Survey's geoid model for Great Britain, Ireland, and
surrounding waters.  It provides the separation *N* between the GRS 80
ellipsoid and the local geoid so that:

```
orthometric height = ellipsoidal (GNSS) height − N
```

## Installation

```bash
pip install osgm15
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add osgm15
```

## Quick start

```python
from osgm15 import OSGM15Grid

# Load the ASCII GTX file (download from Ordnance Survey)
grid = OSGM15Grid.from_file("OSGM15_GTX_ASCII.gtx")

# Single-point lookup — returns geoid separation N in metres
N = grid.height_N(lat=52.0905, lon=-2.2160)

# Vectorised lookup (NumPy arrays in, NumPy array out)
import numpy as np
lats = np.array([52.0905, 51.5074])
lons = np.array([-2.2160, -0.1278])
N_arr = grid.height_N_array(lats, lons)  # points outside grid → NaN
```

## Obtaining the GTX file

The OSGM15 ASCII GTX file is published by Ordnance Survey and can be
downloaded from:

<https://www.ordnancesurvey.co.uk/geodesy-positioning/coordinate-transformations/resources>

## API

### `OSGM15Grid.from_file(path)`

Read an ASCII GTX file and return an `OSGM15Grid` instance.

### `OSGM15Grid.height_N(lat, lon) → float`

Bilinear interpolation of geoid separation at a single WGS 84 point.
Raises `ValueError` if the point is outside the grid.

### `OSGM15Grid.height_N_array(lat, lon) → np.ndarray`

Vectorised version — accepts array-like inputs and returns a NumPy array.
Points outside the grid are set to `NaN`.

## Development

```bash
# Clone and set up
git clone git@github.com:phillipaarestrup/osgm15.git
cd osgm15
uv sync

# Run tests
uv run pytest
```

## Example projects
TBA

## License

MIT
