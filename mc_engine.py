import numpy as np
from options_models import Option
from path_simulator import PathSimulator

class MonteCarloEngine:
    """
    Orchestrates Monte Carlo simulations for option pricing.
    """
    def __init__(self, option: Option, simulator: PathSimulator, risk_free_rate: float):
        """
        Args:
            option: An instance of a class derived from Option.
            simulator: An instance of PathSimulator, configured for the underlying.
            risk_free_rate: The risk-free rate for discounting.
        """
        if not isinstance(option, Option):
            raise TypeError("option must be an instance of Option or its subclass.")
        if not isinstance(simulator, PathSimulator):
            raise TypeError("simulator must be an instance of PathSimulator.")

        self.option     = option
        self.simulator  = simulator
        self.r          = risk_free_rate

    def run_simulations(self, num_simulations: int, use_vectorized_paths=False) -> tuple[float, float]:
        """
        Runs the Monte Carlo simulation.
        Args:
            num_simulations: The number of simulation paths to generate.
            use_vectorized_paths: If True, use the simulator's ability to generate
                                  a matrix of paths at once. This requires the option's
                                  payoff function to be designed to handle a matrix if possible,
                                  or we loop through the matrix here.
                                  For simplicity, we'll loop here if payoff is not vectorized.
        Returns:
            A tuple containing:
                - estimated_price (float): The Monte Carlo estimate of the option price.
                - standard_error (float): The standard error of the estimate.
        """
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be positive.")

        payoffs = np.zeros(num_simulations)

        if use_vectorized_paths and hasattr(self.simulator, 'generate_paths_matrix'):
            # Generate all paths at once
            all_paths = self.simulator.generate_paths_matrix(num_simulations)
            for i in range(num_simulations):
                payoffs[i] = self.option.payoff(all_paths[i, :])
            # Note: If option.payoff could take all_paths and return an array of payoffs,
            # this loop could be avoided, further improving performance.
        else:
            # Generate paths one by one
            for i in range(num_simulations):
                path = self.simulator.generate_path()
                payoffs[i] = self.option.payoff(path)
        
        average_payoff = np.mean(payoffs)
        option_price = average_payoff * np.exp(-self.r * self.option.time_to_expiry)
        
        # Calculate standard error of the mean
        # SE = std(discounted_payoffs) / sqrt(num_simulations)
        # Or, equivalently, std(payoffs) * exp(-rT) / sqrt(num_simulations)
        std_dev_payoffs = np.std(payoffs, ddof=1) # ddof=1 for sample standard deviation
        standard_error = (std_dev_payoffs / np.sqrt(num_simulations)) * np.exp(-self.r * self.option.time_to_expiry)
        
        return option_price, standard_error

# Example usage (optional, for testing this file directly)
# if __name__ == '__main__':
#     from random_utils import RandomGenerator
#     from option_models import EuropeanOption, OptionSide

#     rng_test = RandomGenerator(seed=123)
#     sim_test = PathSimulator(100, 0.05, 0.01, 0.2, 1.0, 252, rng_test)
#     opt_test = EuropeanOption(strike=100, time_to_expiry=1.0, side=OptionSide.CALL)
    
#     engine_test = MonteCarloEngine(opt_test, sim_test, 0.05)
    
#     price, se = engine_test.run_simulations(1000)
#     print(f"Price (path-by-path): {price:.5f}, SE: {se:.5f}")

#     price_vec, se_vec = engine_test.run_simulations(1000, use_vectorized_paths=True)
#     print(f"Price (vectorized paths): {price_vec:.5f}, SE: {se_vec:.5f}")
