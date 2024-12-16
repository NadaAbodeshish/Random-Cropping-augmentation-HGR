import numpy as np

def generate_random_basis(num_bps_points, n_dims=3, radius=1., random_seed=None):
    """
    Generate a random basis point set.
    
    Args:
        num_bps_points (int): Number of basis points to generate.
        n_dims (int): Dimensionality of the points (default is 3 for 3D).
        radius (float): Radius of the uniform distribution.
        random_seed (int): Random seed for reproducibility.
    
    Returns:
        np.ndarray: Generated basis points of shape (num_bps_points, n_dims).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    return np.random.uniform(-radius, radius, (num_bps_points, n_dims))
