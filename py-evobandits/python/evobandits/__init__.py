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

import importlib.util

from evobandits import logging
from evobandits.evobandits import Arm, EvoBandits
from evobandits.params import CategoricalParam, FloatParam, IntParam
from evobandits.study import ALGORITHM_DEFAULT, Study

__all__ = [
    "Arm",
    "ALGORITHM_DEFAULT",
    "EvoBandits",
    "logging",
    "Study",
    "CategoricalParam",
    "FloatParam",
    "IntParam",
]

if importlib.util.find_spec("sklearn") is not None:
    # Only import and expose EvoBanditsSearchCV if sklearn is available
    from evobandits.search import EvoBanditsSearchCV  # noqa

    __all__.append("EvoBanditsSearchCV")
