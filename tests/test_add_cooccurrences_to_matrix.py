# Copyright 2023 AICONSLab
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

import numpy as np

from snftools.snf import add_cooccurrences_to_matrix


class TestAddCooccurrencesToMatrix(unittest.TestCase):
    def test_zero_matrix(self):
        matrix = np.zeros((3, 3))
        cooccurrences = np.array([0, 1])
        expected = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        add_cooccurrences_to_matrix(matrix, cooccurrences)
        np.testing.assert_equal(expected, matrix)

    def test_nonzero_matrix(self):
        matrix = np.array([[0, 4, 7], [4, 0, 3], [7, 3, 0]])
        cooccurrences = np.array([1, 2])
        expected = np.array([[0, 4, 7], [4, 0, 4], [7, 4, 0]])
        add_cooccurrences_to_matrix(matrix, cooccurrences)
        np.testing.assert_equal(expected, matrix)


if __name__ == "__main__":
    unittest.main()
