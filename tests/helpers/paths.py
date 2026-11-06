"""Directories of the tests, from this file: tests run from any directory."""

from pathlib import Path

TESTS = Path(__file__).resolve().parent.parent
DATA = TESTS / "data"
# what the layers of the original codes received and returned
LAYER_DATA = DATA / "layers"
CIF = TESTS / "cif"
