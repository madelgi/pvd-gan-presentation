import numpy as np


def sample_data(num_pts=10000, scale=100):
    """Generate sample data for our GAN.
    """
    xdata = 100 * (np.random.random_sample(num_pts) - 0.5)
    return np.array([[x, 10 + x * x] for x in xdata])
