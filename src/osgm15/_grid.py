"""Core grid class for OSGM15 geoid separation lookup."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class OSGM15Grid:
    """OSGM15 geoid-separation grid loaded from an ASCII GTX file.

    The grid stores geoid-ellipsoid separation values *N* on a regular
    latitude/longitude grid.  Given a WGS 84 coordinate pair the class
    returns *N* via bilinear interpolation so that::

        orthometric height = ellipsoidal (GNSS) height - N

    Parameters
    ----------
    lat0 : float
        Latitude of the south-west corner of the grid (degrees).
    lon0 : float
        Longitude of the south-west corner of the grid (degrees).
    dlat : float
        Grid spacing in latitude (degrees).
    dlon : float
        Grid spacing in longitude (degrees).
    nrows : int
        Number of rows (latitude direction).
    ncols : int
        Number of columns (longitude direction).
    vals : np.ndarray
        Geoid separation values in metres, shape ``(nrows, ncols)``,
        stored row-major starting at the south-west corner.
    """

    lat0: float
    lon0: float
    dlat: float
    dlon: float
    nrows: int
    ncols: int
    vals: np.ndarray  # (nrows, ncols), row-major from SW corner

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> OSGM15Grid:
        """Read an ASCII GTX file and return an :class:`OSGM15Grid`.

        Parameters
        ----------
        path : str or Path
            Path to the ``*.gtx`` ASCII file (e.g. ``OSGM15_GTX_ASCII.gtx``).

        Returns
        -------
        OSGM15Grid

        Raises
        ------
        ValueError
            If the header is malformed or the value count does not match
            the declared grid size.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split()
            if len(header) != 6:
                raise ValueError(f"Bad GTX header (expected 6 fields): {header}")
            lat0, lon0, dlat, dlon = map(float, header[:4])
            nrows, ncols = map(int, header[4:6])

            vals: list[float] = []
            for line in f:
                s = line.strip()
                if not s:
                    continue
                vals.append(float(s))

        expected = nrows * ncols
        if len(vals) != expected:
            raise ValueError(
                f"Bad GTX size: got {len(vals)} values, expected {expected}"
            )
        return cls(
            lat0=lat0,
            lon0=lon0,
            dlat=dlat,
            dlon=dlon,
            nrows=nrows,
            ncols=ncols,
            vals=np.array(vals, dtype=np.float64).reshape(nrows, ncols),
        )

    # ------------------------------------------------------------------
    # Scalar lookup
    # ------------------------------------------------------------------

    def _at(self, r: int, c: int) -> float:
        return float(self.vals[r, c])

    def height_N(self, lat: float, lon: float) -> float:
        """Return the geoid separation *N* (metres) at a single point.

        Uses bilinear interpolation on the grid.

        Parameters
        ----------
        lat, lon : float
            WGS 84 coordinates in degrees.

        Returns
        -------
        float
            Geoid separation in metres.

        Raises
        ------
        ValueError
            If the point falls outside the grid extent.
        """
        x = (lon - self.lon0) / self.dlon
        y = (lat - self.lat0) / self.dlat

        if not (0 <= x <= self.ncols - 1 and 0 <= y <= self.nrows - 1):
            raise ValueError("Point outside grid extent")

        c0 = int(math.floor(x))
        r0 = int(math.floor(y))
        c1 = min(c0 + 1, self.ncols - 1)
        r1 = min(r0 + 1, self.nrows - 1)

        tx = x - c0
        ty = y - r0

        v00 = self._at(r0, c0)
        v10 = self._at(r0, c1)
        v01 = self._at(r1, c0)
        v11 = self._at(r1, c1)

        return (
            v00 * (1 - tx) * (1 - ty)
            + v10 * tx * (1 - ty)
            + v01 * (1 - tx) * ty
            + v11 * tx * ty
        )

    # ------------------------------------------------------------------
    # Vectorised lookup
    # ------------------------------------------------------------------

    def height_N_array(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Return geoid separation *N* for arrays of coordinates.

        Parameters
        ----------
        lat, lon : array-like, same shape
            WGS 84 coordinates in degrees.

        Returns
        -------
        np.ndarray
            Geoid separation in metres, same shape as input.
            Points outside the grid extent are set to ``NaN``.
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)

        xf = (lon - self.lon0) / self.dlon
        yf = (lat - self.lat0) / self.dlat

        inside = (
            (xf >= 0)
            & (xf <= self.ncols - 1)
            & (yf >= 0)
            & (yf <= self.nrows - 1)
        )

        xv = xf[inside]
        yv = yf[inside]

        c0 = np.floor(xv).astype(int)
        r0 = np.floor(yv).astype(int)
        c1 = np.minimum(c0 + 1, self.ncols - 1)
        r1 = np.minimum(r0 + 1, self.nrows - 1)

        tx = xv - c0
        ty = yv - r0

        v00 = self.vals[r0, c0]
        v10 = self.vals[r0, c1]
        v01 = self.vals[r1, c0]
        v11 = self.vals[r1, c1]

        interp = (
            v00 * (1 - tx) * (1 - ty)
            + v10 * tx * (1 - ty)
            + v01 * (1 - tx) * ty
            + v11 * tx * ty
        )

        out = np.full(lat.shape, np.nan, dtype=np.float64)
        out[inside] = interp
        return out
