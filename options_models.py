from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

class OptionSide(Enum):
    CALL = "CALL"
    PUT  = "PUT"

class Option(ABC):
    """
    Abstract base class for financial options.
    """
    def __init__(self, time_to_expiry: float):
        if time_to_expiry <= 0:
            raise ValueError("Time to expiry must be positive.")
        self._time_to_expiry = time_to_expiry
    
    @abstractmethod
    def payoff(self, asset_path: np.ndarray) -> float:
        """
        Calculates the payoff of the option given an asset price path.
        Args:
            asset_path: A NumPy array representing the asset's price path (S0 to ST).
        Returns:
            The option's payoff.
        """
        pass

    @property
    def time_to_expiry(self) -> float:
        return self._time_to_expiry

    @abstractmethod
    def option_type(self) -> str:
        """
        Returns a string identifying the type of the option.
        """
        pass

class EuropeanOption(Option):
    """
    Represents a European-style option.
    """
    def __init__(self, strike: float, time_to_expiry: float, side: OptionSide):
        super().__init__(time_to_expiry)
        if strike <= 0:
            raise ValueError("Strike price must be positive.")
        self.strike = strike
        self.side = side
    
    def payoff(self, asset_path: np.ndarray) -> float:
        if not isinstance(asset_path, np.ndarray) or asset_path.ndim != 1 or len(asset_path) == 0:
            raise ValueError("asset_path must be a non-empty 1D NumPy array (speed performance preference that is in-built).")
        terminal_price = asset_path[-1] # S_T

        if self.side == OptionSide.CALL:
            return np.maximum(0.0, terminal_price - self.strike)
        elif self.side == OptionSide.PUT:
            return np.maximum(0.0, self.strike - terminal_price)
        else:
            raise ValueError("Invalid option side.") # Should not happen with Enum

    def option_type(self) -> str:
        return f"European {self.side.value}"

# Example usage (optional, for testing this file directly)
# if __name__ == '__main__':
#     dummy_path = np.array([100, 102, 105, 103, 107])
    
#     call_option = EuropeanOption(strike=105, time_to_expiry=1.0, side=OptionSide.CALL)
#     print(f"{call_option.option_type()} Payoff: {call_option.payoff(dummy_path)}")
#     print(f"Time to Expiry: {call_option.time_to_expiry}")

#     put_option = EuropeanOption(strike=105, time_to_expiry=1.0, side=OptionSide.PUT)
#     print(f"{put_option.option_type()} Payoff: {put_option.payoff(dummy_path)}")