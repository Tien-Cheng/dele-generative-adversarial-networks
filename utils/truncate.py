import numpy as np
from scipy.stats import truncnorm


def truncated_z_sample(batch_size, z_dim, truncation=0.5, seed=None):
    """
    Apply truncation to noise vector. Implementation from https://github.com/ajbrock/BigGAN-PyTorch/blob/7b65e82d058bfe035fc4e299f322a1f83993e04c/TFHub/biggan_v1.py#L16 
    """
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim), random_state=state)
    return truncation * values
