# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 19.10.2022

import numpy as np
import numba

# Local constants are not very cool but we are lazy, yeah?
NUM_SEGMENTS = 16
SPRING_CONSTANT = 3 * (NUM_SEGMENTS - 1) / 2

@numba.njit
def total_bond_energy(polymer):
    """Calculate the energy of a polymer stored in it's bonds."""
    diffs = polymer[..., 1:, :] - polymer[..., :-1, :]
    summed = np.sum(diffs * diffs)
    return summed * SPRING_CONSTANT
