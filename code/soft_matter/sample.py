# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 18.10.2022

import numpy as np
from numpy.typing import NDArray

def sample_ball_radius() -> NDArray[np.float64]:
    """Generate a sample from a ball uniformly distributed along radius."""
    raw_sample = np.random.uniform(low=-1, high=1, size=3)  # 3 dimensions
    normed = raw_sample / np.linalg.norm(raw_sample)
    radius_sample = np.random.uniform(low=0, high=1)
    return normed * radius_sample

def sample_initial(num_segments: int) -> NDArray[np.float64]:
    """Create an independently sampled initial configuration."""
    polymer = np.zeros((num_segments, 3))
    for seg in range(1, num_segments):
        bond_length = sample_ball_radius()
        polymer[seg, :] += polymer[seg - 1, :] + bond_length

    return polymer
