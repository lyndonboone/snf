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
from snftools.algo import Solution


def dummy_fn(array, c):
    return array.sum() * c  # no relation to clustering, just wanted to use c


class TestSolution(unittest.TestCase):
    def test_len(self):
        solution = Solution(4, [1, 2, 3, 4])
        self.assertEqual(len(solution), 4)

    def test_update_fitness(self):
        solution = Solution(2, [0, 2])
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        solution.update_fitness(data, dummy_fn)
        self.assertEqual(solution.fitness, 32)

    def test_fuse_wrong_c(self):
        a = Solution(2, [0, 2])
        b = Solution(3, [3])
        with self.assertRaises(AssertionError):
            a.fuse(b)

    def test_fuse(self):
        a = Solution(2, [0, 2])
        b = Solution(2, [3])
        c = a.fuse(b)
        self.assertEqual(c.c, 2)
        self.assertListEqual(c.features, [0, 2, 3])


if __name__ == "__main__":
    unittest.main()
