"""Tests for OSGM15Grid using a small synthetic GTX file."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from osgm15 import OSGM15Grid


@pytest.fixture()
def tiny_gtx(tmp_path: Path) -> Path:
    """Write a minimal 2x3 ASCII GTX file and return its path.

    Grid corners:
        lat0=50.0  lon0=-3.0  dlat=1.0  dlon=1.0
        nrows=2    ncols=3

    Values (row-major, SW corner first):
        row 0 (lat=50): 10.0  20.0  30.0
        row 1 (lat=51): 40.0  50.0  60.0
    """
    p = tmp_path / "test.gtx"
    p.write_text(
        textwrap.dedent("""\
            50.0 -3.0 1.0 1.0 2 3
            10.0
            20.0
            30.0
            40.0
            50.0
            60.0
        """)
    )
    return p


class TestFromFile:
    def test_loads_header(self, tiny_gtx: Path) -> None:
        g = OSGM15Grid.from_file(tiny_gtx)
        assert g.lat0 == 50.0
        assert g.lon0 == -3.0
        assert g.dlat == 1.0
        assert g.dlon == 1.0
        assert g.nrows == 2
        assert g.ncols == 3

    def test_loads_values(self, tiny_gtx: Path) -> None:
        g = OSGM15Grid.from_file(tiny_gtx)
        expected = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        np.testing.assert_array_equal(g.vals, expected)

    def test_bad_header(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.gtx"
        p.write_text("1.0 2.0 3.0\n")
        with pytest.raises(ValueError, match="Bad GTX header"):
            OSGM15Grid.from_file(p)

    def test_bad_value_count(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.gtx"
        p.write_text("50.0 -3.0 1.0 1.0 2 3\n10.0\n20.0\n")
        with pytest.raises(ValueError, match="Bad GTX size"):
            OSGM15Grid.from_file(p)


class TestHeightN:
    def test_corner_sw(self, tiny_gtx: Path) -> None:
        g = OSGM15Grid.from_file(tiny_gtx)
        assert g.height_N(50.0, -3.0) == pytest.approx(10.0)

    def test_corner_ne(self, tiny_gtx: Path) -> None:
        g = OSGM15Grid.from_file(tiny_gtx)
        assert g.height_N(51.0, -1.0) == pytest.approx(60.0)

    def test_midpoint_bilinear(self, tiny_gtx: Path) -> None:
        g = OSGM15Grid.from_file(tiny_gtx)
        # Centre of SW cell: lat=50.5, lon=-2.5
        # corners: 10, 20, 40, 50 → mean = 30
        assert g.height_N(50.5, -2.5) == pytest.approx(30.0)

    def test_outside_raises(self, tiny_gtx: Path) -> None:
        g = OSGM15Grid.from_file(tiny_gtx)
        with pytest.raises(ValueError, match="outside grid"):
            g.height_N(49.0, -3.0)


class TestHeightNArray:
    def test_vectorised_matches_scalar(self, tiny_gtx: Path) -> None:
        g = OSGM15Grid.from_file(tiny_gtx)
        lats = np.array([50.0, 50.5, 51.0])
        lons = np.array([-3.0, -2.5, -1.0])
        result = g.height_N_array(lats, lons)
        expected = np.array([
            g.height_N(50.0, -3.0),
            g.height_N(50.5, -2.5),
            g.height_N(51.0, -1.0),
        ])
        np.testing.assert_allclose(result, expected)

    def test_outside_returns_nan(self, tiny_gtx: Path) -> None:
        g = OSGM15Grid.from_file(tiny_gtx)
        result = g.height_N_array(np.array([0.0]), np.array([0.0]))
        assert np.isnan(result[0])

    def test_mixed_inside_outside(self, tiny_gtx: Path) -> None:
        g = OSGM15Grid.from_file(tiny_gtx)
        lats = np.array([50.0, 0.0, 51.0])
        lons = np.array([-3.0, 0.0, -1.0])
        result = g.height_N_array(lats, lons)
        assert result[0] == pytest.approx(10.0)
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(60.0)
