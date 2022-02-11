import numpy as np
from scipy.stats import truncnorm


def truncated_z_sample(batch_size, z_dim, truncation=0.5, seed=None):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim), random_state=state)
    return truncation * values
