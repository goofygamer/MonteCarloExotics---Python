import numpy as np
import time
from random_utils import RandomGenerator
from path_simulator import PathSimulator
from options_models import EuropeanOption, OptionSide # Using European for initial test
from mc_engine import MonteCarloEngine

def main():
    # --- Simulation Parameters ---
    S0 = 100.0       # Initial stock price
    K = 105.0        # Strike price
    r = 0.05         # Risk-free rate (5%)
    q = 0.01         # Dividend yield (1%)
    sigma = 0.20     # Volatility (20%)
    T = 1.0          # Time to expiry (1 year)
    num_steps = 252  # Number of time steps (e.g., daily for a year)
    num_simulations = 250_000 # Number of Monte Carlo simulations
    seed = 3142001        # Seed for reproducibility

    print("--- Python Monte Carlo Pricer (Core Engine Test) ---")
    print("Parameters:")
    print(f"  S0: {S0}, K: {K}, r: {r}, q: {q}, sigma: {sigma}, T: {T}")
    print(f"  Time Steps: {num_steps}, Simulations: {num_simulations}, Seed: {seed}")
    print("-------------------------------------------------------")

    # --- Setup ---
    # 1. Random Number Generator
    rng = RandomGenerator(seed=seed)

    # 2. Path Simulator
    path_sim = PathSimulator(S0, r, q, sigma, T, num_steps, rng)

    # 3. Option (Using European Call for this test)
    european_call = EuropeanOption(strike=K, time_to_expiry=T, side=OptionSide.CALL)
    
    # 4. Monte Carlo Engine
    mc_engine_call = MonteCarloEngine(european_call, path_sim, r)

    # --- Run Simulation (Call Option) ---
    print(f"\nPricing {european_call.option_type()}...")
    start_time = time.time()
    # For a large number of simulations, use_vectorized_paths might be faster
    # if the payoff calculation is simple.
    # For very complex path-dependent payoffs, path-by-path might not be much slower
    # relative to the payoff calc itself. Test both for your specific exotics.
    price_call, se_call = mc_engine_call.run_simulations(num_simulations, use_vectorized_paths=True)
    end_time = time.time()

    # --- Output Results (Call Option) ---
    print(f"  Estimated Price: {price_call:.5f}")
    print(f"  Standard Error:  {se_call:.5f}")
    print(f"  Confidence Interval (95%): [{price_call - 1.96*se_call:.5f}, {price_call + 1.96*se_call:.5f}]")
    print(f"  Time taken: {end_time - start_time:.2f} seconds")
    print("-------------------------------------------------------")

    # --- Test with a Put Option ---
    european_put = EuropeanOption(strike=K, time_to_expiry=T, side=OptionSide.PUT)
    mc_engine_put = MonteCarloEngine(european_put, path_sim, r) # Re-use path_sim

    print(f"\nPricing {european_put.option_type()}...")
    start_time = time.time()
    price_put, se_put = mc_engine_put.run_simulations(num_simulations, use_vectorized_paths=True)
    end_time = time.time()

    print(f"  Estimated Price: {price_put:.5f}")
    print(f"  Standard Error:  {se_put:.5f}")
    print(f"  Confidence Interval (95%): [{price_put - 1.96*se_put:.5f}, {price_put + 1.96*se_put:.5f}]")
    print(f"  Time taken: {end_time - start_time:.2f} seconds")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    main()