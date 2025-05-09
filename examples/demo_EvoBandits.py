# Copyright 2025 EvoBandits
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from evobandits import EvoBandits


def rosenbrock_function(number: list):
    return sum(
        [
            100 * (number[i + 1] - number[i] ** 2) ** 2 + (1 - number[i]) ** 2
            for i in range(len(number) - 1)
        ]
    )


if __name__ == "__main__":
    bounds = [(-5, 10), (-5, 10)]
    evaluation_budget = 10000
    n_best = 1
    evobandits = EvoBandits()
    results = evobandits.optimize(
        rosenbrock_function, bounds, evaluation_budget, n_best
    )

    print(len(results))  # matches n_best
    print(results[0].to_dict)  # action_vector, mean_reward, and num_pull for best arm
