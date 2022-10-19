# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 19.10.2022
from pathlib import Path

import numpy as np
import numba
import scipy.interpolate

# Local constants are not very cool but we are lazy, yeah?
NUM_SEGMENTS = 16
A_SEGMENTS = 4
SPRING_CONSTANT = 3 * (NUM_SEGMENTS - 1) / 2
DATA_DIR = Path(__file__).parent.parent.parent / "data"

_pos, _phobic, _philic = np.loadtxt(DATA_DIR / "omega.dat", delimiter=" ", skiprows=1, dtype=np.float64).T
_field_A = scipy.interpolate.interp1d(_pos, _phobic, fill_value=_phobic[-1], bounds_error=False)
_field_B = scipy.interpolate.interp1d(_pos, _philic, fill_value=_philic[-1], bounds_error=False)

@numba.njit
def total_bond_energy(polymer):
    """Calculate the energy of a polymer stored in it's bonds."""
    diffs = polymer[..., 1:, :] - polymer[..., :-1, :]
    summed = np.sum(diffs * diffs)
    return summed * SPRING_CONSTANT

@numba.njit
def polymer_pos_to_rad(polymer):
    xs, ys, zs = polymer[..., 0], polymer[..., 1], polymer[..., 2]
    return np.sqrt(xs * xs + ys * ys + zs * zs)

def energy_with_external_fields(polymer):
    """Calculate the energy of a polymer in the external field."""
    bond_energy = total_bond_energy(polymer)
    polymer_rad = polymer_pos_to_rad(polymer)
    aas, bbs = polymer_rad[:A_SEGMENTS], polymer_rad[A_SEGMENTS:]
    field_energy = np.sum(_field_A(aas)) + np.sum(_field_B(bbs))
    return bond_energy + field_energy / NUM_SEGMENTS
