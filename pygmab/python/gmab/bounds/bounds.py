class Bounds:
    """The Bounds module manages the user-configured boundaries for gmab.

    This object provides interfaces to set the bounds that will be suggested
    for the optimization.
    """

    def __init__(self) -> None:
        self.low = []
        self.high = []
        self.step = []
        self.n_steps = []

    @property
    def internal(self) -> list[tuple]:
        """Creates and returns the internal bounds for gmab.

        Returns:
            list[tuple]: A list of tuples, containaing a (low, high) pair for each bound.
        """
        as_tuple = [(0, self.n_steps[idx]) for idx in range(len(self.n_steps))]
        return as_tuple

    def map_to_external_repr(self, action_vector: list[int]) -> list[int]:
        """Map the internal action vector to the external value representation

        Args:
            action_vector (list[int]): An internal action_vector.

        Returns:
            list[int]: The external represenation of the action_vector, based on mapping
            the action_vector using the bounds.
        """
        values = [
            self.low[idx] + self.step[idx] * action_vector[idx]
            for idx in range(len(action_vector))
        ]
        return values

    def suggest_int(self, low: int, high: int, size: int = 1, step: int = 1) -> None:
        """Adds an integer decision parameter to the bounds.

        Args:
            low (int): The lower bound of the parameter.
            high (int): The upper bound of the parameter.
            size (int, optional): Length of the decision vector, defaults to 1.
            step (int, optional): Stepsize for the decision vector, defaults to 1.

        Raises:
            TypeError: If low, high, or size are not integers.
            ValueError: If high <= low or size non-positive.
        """
        if not all(isinstance(var, int) for var in [low, high, size, step]):
            raise TypeError("low, high, size must be integers when suggesting an int.")
        if high <= low:
            raise ValueError("high must be larger than low when suggesting an int.")
        if size < 1:
            raise ValueError("size must be positive when suggesting an int.")
        if step < 1:
            raise ValueError("step must be positive when suggeting an int.")

        n_steps = (high - low) // step
        if n_steps < 1:
            raise ValueError("step must be smaller than the difference between low and high.")

        for _ in range(size):
            self.low.append(low)
            self.high.append(high)
            self.step.append(step)
            self.n_steps.append(n_steps)
