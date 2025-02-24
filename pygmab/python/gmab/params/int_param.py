class IntParam:
    """
    A class representing an integer parameter.
    """

    def __init__(self, low: int, high: int, size: int = 1, step: int = 1):
        """
        Creates an IntParam that will suggest integer values during the optimization.

        The parameter can be either an integer, or a list of integers, depending on the specified
        size. The values sampled by the optimization will be limited to the specified granularity,
        lower and upper bounds.

        Args:
            low (int): The lower bound of the suggested values.
            high (int): The upper bound of the suggested values.
            size (int): The size if the parameter shall be a list of integers. Default is 1.
            step (int): The step size between the suggested values. Default is 1.

        Returns:
            IntParam: An instance of the parameter with the specified properties.

        Raises:
            TypeError: If any of the arguments are not integers.
            ValueError: If high is not greater than low, or if size or step is not positive.

        Example:
        >>> param = IntParam(low=1, high=10, size=3, step=2)
        >>> print(param)
        IntParam(low=1, high=10, size=3, step=2)

        Notes:
            The step size determines the granularity of the sampled values. For example, a step
            of 2 means that only every second integer within the range will be considered.

        """
        if not all(isinstance(args, int) for args in [low, high, size, step]):
            raise TypeError("low, high, size and step must be int for IntParams")
        if high <= low:
            raise ValueError("high must be larger than low for IntParams.")
        if size < 1:
            raise ValueError("size must be positive for IntParams.")
        if step < 1:
            raise ValueError("step must be positive for IntParams")

        self.low: int = low
        self.high: int = high
        self.size: int = size
        self.step: int = step
        self._bounds: list[tuple] | None = None

    def __repr__(self):
        return f"IntParam(low={self.low}, high={self.high}, size={self.size}, step={self.step})"

    @property
    def bounds(self) -> list[tuple]:
        """
        Calculate and return the parameter's internal bounds for the optimization.

        The bounds will be used as constraints for the internal representation (or actions)
        of the optimization algorithm about the parameter's value.

        Returns:
            list[tuple]: A list of tuples representing the bounds.

        """
        if not self._bounds:
            if self.step == 1:
                upper_bound = self.high
            else:
                upper_bound = (self.high - self.low) // self.step
                upper_bound += 1 if (self.high - self.low) % self.step != 0 else 0
                upper_bound += self.low
            self._bounds = [(self.low, upper_bound)] * self.size
        return self._bounds

    def map_to_value(self, actions: list[int]) -> int | list[int]:
        """
        Maps an action by the optimization problem to the value of the parameter.

        Args:
            actions (list[int]): A list of integers to map.

        Returns:
            int | list[int]: The resulting value.
        """
        if self.step > 1:
            actions = [min(self.low + x * self.step, self.high) for x in actions]

        if self.size == 1:
            return actions[0]
        return actions
