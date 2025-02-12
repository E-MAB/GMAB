BOUNDS_2D = [(-5, 10), (-5, 10)]
RESULT_2D = [1, 1]


def objective(number: list):
    """Computes the value of the multidimensional rosenbrock function."""
    return sum(
        [
            100 * (number[i + 1] - number[i] ** 2) ** 2 + (1 - number[i]) ** 2
            for i in range(len(number) - 1)
        ]
    )
