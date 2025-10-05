"""
gradient.py

A simple script to demonstrate the calculation of a function's gradient.
This script defines a function f(x, y) and computes its gradient at a specific point.
"""

import numpy as np
from numpy.typing import NDArray

def f(x: float, y: float) -> float:
    """Calculates the value of the function f(x, y) = x^2 + y^3."""
    return x**2 + y**3

def calculate_gradient(x: float, y: float) -> NDArray[np.float64]:
    """
    Calculates the gradient of the function f(x, y).

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        NDArray[np.float64]: A numpy array representing the gradient [df/dx, df/dy].
    """
    df_dx = 2 * x
    df_dy = 3 * y**2
    return np.array([df_dx, df_dy])

def main() -> None:
    """Main function to execute the script."""
    point_to_evaluate = (2.0, 3.0)
    
    value_at_point = f(point_to_evaluate[0], point_to_evaluate[1])
    gradient_at_point = calculate_gradient(point_to_evaluate[0], point_to_evaluate[1])

    print(f"Function value at point {point_to_evaluate} is: {value_at_point}")
    print(f"Gradient at point {point_to_evaluate} is: {gradient_at_point}")


if __name__ == "__main__":
    main()
