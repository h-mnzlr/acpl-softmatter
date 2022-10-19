# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 19.10.2022

import numpy as np
import numba

# Again: local constants are sad!
NUM_SEGMENTS = 16

@numba.njit
def polymer_step(polymer):
    """Propose a step on a polymer with a single segment displacement."""
    selected_index = np.random.choice(NUM_SEGMENTS)

    step_length = 1 / np.sqrt(NUM_SEGMENTS - 1)
    uniform_step = np.random.rand(3) * 2 - 1
    displacement = uniform_step * step_length

    new_polymer = polymer.copy()
    new_polymer[selected_index, :] += displacement

    return new_polymer
