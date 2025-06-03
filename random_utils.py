import numpy as np

class RandomGenerator:
    """
    Provides utilities for random number generation, 
    primarily standard normal variables, i.e. N(0,1).
    """
    def __init__(self, seed=None):
        """
        Initializes the random number generator.
        Args:
            seed(int, optional): Seed for the random number
            generator for reproducibility.
            Defaults to None (random seed).
        """
        self.rng = np.random.default_rng(seed)
    

    def get_normal(self, size=None):
        """
        Generates standard normal random variate(s) (mean 0, variance 1).
        Args:
            size (int or tuple, optional): Output shape. If None, a single scalar is returned.
        Returns:
            float or np.ndarray: A single standard normal variate or an array of them.
        """
        return self.rng.normal(loc=0.0, scale=1.0, size=size)
    
    def get_uniform(self, low=0.0, high=1.0, size=None):
        """
        Generates uniform random variate(s) in [low, high).
        Args:
            low (float): Lower bound (inclusive).
            high (float): Upper bound (exclusive).
            size (int or tuple, optional): Output shape. If None, a single scalar is returned.
        Returns:
            float or np.ndarray: A single uniform variate or an array of them.
        """
        return self.rng.uniform(low=low, high=high, size=size)

# Example usage (optional, for testing this file directly)
# if __name__ == '__main__':
#     rng_seeded = RandomGenerator(seed=42)
#     print("Seeded Normal:", [rng_seeded.get_normal() for _ in range(3)])
    
#     rng_unseeded = RandomGenerator()
#     print("Unseeded Normal:", [rng_unseeded.get_normal() for _ in range(3)])
#     print("Normal Array:", rng_unseeded.get_normal(size=(2,2)))