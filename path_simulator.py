import numpy as np
from random_utils import RandomGenerator


class PathSimulator:
    """
    Simulates asset price path using GBM.
    """
    def __init__(self, initial_price: float, risk_free_rate: float, dividend_yield: float,
         volatility: float, time_to_expiry: float, num_steps: float, random_generator: RandomGenerator):
        """
        Args:
            initial_price (S0)  : Initial asset price.
            risk_free_rate (r)  : Risk-free interest rate (annualized).
            dividend_yield (q)  : Continuous dividend yield (annualized).
            volatility (sigma)  : Asset price volatility (annualized).
            time_to_expiry (T)  : Time to option expiry in years.
            num_steps           : Number of time steps in the path.
            random_generator    : An instance of RandomGenerator.
        """
        if num_steps <= 0:
            raise ValueError("Number of steps must be greater than 0.")

        self.S0         = initial_price
        self.r          = risk_free_rate
        self.q          = dividend_yield
        self.sigma      = volatility
        self.T          = time_to_expiry
        self.num_steps  = num_steps
        self.dt         = self.T/self.num_steps
        self.rng        = random_generator

    def generate_path(self) -> np.ndarray:
        """
        Generates a single asset price path as an array.
        The path includes S0 up to S_T (num_steps + 1 points).
        """
        path = np.zeros(self.num_steps + 1)
        path[0] = self.S0

        # GBM discrete time evolution: S_{t+dt} = S_t * exp((r - q - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z)
        drift = (self.r - self.q - 0.5 * self.sigma**2) * self.dt
        diffusion_factor = self.sigma * np.sqrt(self.dt)

        # Generate all random numbers at once for efficiency
        Zs = self.rng.get_normal(size=self.num_steps)

        for i in range(1, self.num_steps + 1):
            path[i] = path[i-1] * np.exp(drift + diffusion_factor * Zs[i-1])
        
        return path
    
    def generate_paths_matrix(self, num_paths: float) -> np.ndarray:
        """
        Generates a matrix of asset price paths.
        Each row is a path, each column is a time step.
        Shape: (num_paths, num_steps + 1)
        This is often more efficient for Monte Carlo if payoffs can be vectorized.
        """
        paths_matrix = np.zeros((num_paths, self.num_steps + 1))
        paths_matrix[:, 0] = self.S0

        drift = (self.r - self.q - 0.5 * self.sigma**2) * self.dt
        diffusion_factor = self.sigma * np.sqrt(self.dt)

        # Generate all random numbers for all paths and steps at once
        Zs_matrix = self.rng.get_normal(size=(num_paths, self.num_steps))

        for i in range(1, self.num_steps + 1):
            paths_matrix[:, i] = paths_matrix[:, i-1] * np.exp(drift + diffusion_factor * Zs_matrix[:, i-1])
        
        return paths_matrix

# Example usage (optional, for testing this file directly)
# if __name__ == '__main__':
#     rng = RandomGenerator(seed=42)
#     simulator = PathSimulator(initial_price=100.0, risk_free_rate=0.05, dividend_yield=0.01,
#                               volatility=0.2, time_to_expiry=1.0, num_steps=252,
#                               random_generator=rng)
    
#     single_path = simulator.generate_path()
#     print("Single Path (first 5 steps):", single_path[:5])
#     print("Single Path length:", len(single_path))

#     paths_mat = simulator.generate_paths_matrix(num_paths=3)
#     print("\nPaths Matrix (first 5 steps for 3 paths):\n", paths_mat[:, :5])
#     print("Paths Matrix shape:", paths_mat.shape)
