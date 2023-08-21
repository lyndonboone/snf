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


# from copy import deepcopy
# import unittest

# import numpy as np
# from snftools.algo import Solutions


# def dummy_fn(array, c):
#     return array.sum() * c  # no relation to clustering, just wanted to use c


# class TestSolutions(unittest.TestCase):
#     def test_init(self):
#         cmin = 2
#         cmax = 3
#         features = [1, 2, 4]
#         solutions = Solutions(cmin, cmax, features)
#         for i, c in enumerate(range(cmin, cmax + 1)):
#             for j, f in enumerate(features):
#                 idx = i * len(features) + j
#                 self.assertEqual(solutions[idx].c, c)
#                 self.assertListEqual(solutions[idx].features, [f])

#     def test_dummy(self):
#         self.assertEqual(2, 2)

#     def test_update_fitness(self):
#         cmin = 2
#         cmax = 3
#         features = [0, 2]
#         solutions = Solutions(cmin, cmax, features)
#         data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
#         solutions.update_fitness(data, dummy_fn)
#         expected = [12, 20, 18, 30]
#         fitnesses = [solution.fitness for solution in solutions]
#         self.assertListEqual(fitnesses, expected)

#     def test_sort(self):
#         cmin = 2
#         cmax = 3
#         features = [1, 4]
#         solutions = Solutions(cmin, cmax, features)
#         init_fitnesses = [0.5, 0.9, 0.2, 0.7]
#         for i, fitness in enumerate(init_fitnesses):
#             solutions[i].fitness = fitness
#         solutions.sort()
#         cs = [solution.c for solution in solutions]
#         fs = [solution.features[0] for solution in solutions]
#         fitnesses = [solution.fitness for solution in solutions]
#         self.assertListEqual(cs, [2, 3, 2, 3])
#         self.assertListEqual(fs, [4, 4, 1, 1])
#         self.assertListEqual(fitnesses, [0.9, 0.7, 0.5, 0.2])

#     def test_trim(self):
#         cmin = 2
#         cmax = 3
#         features = [1, 4]
#         solutions = Solutions(cmin, cmax, features)
#         init_fitnesses = [0.5, 0.9, 0.2, 0.7]
#         for i, fitness in enumerate(init_fitnesses):
#             solutions[i].fitness = fitness
#         solutions.sort()
#         solutions.trim(2)
#         cs = [solution.c for solution in solutions]
#         fs = [solution.features[0] for solution in solutions]
#         fitnesses = [solution.fitness for solution in solutions]
#         self.assertListEqual(cs, [2, 3])
#         self.assertListEqual(fs, [4, 4])
#         self.assertListEqual(fitnesses, [0.9, 0.7])

#     def test_fuse(self):
#         cmin = 2
#         cmax = 3
#         features = [1, 4]
#         solutions = Solutions(cmin, cmax, features)
#         solutions.trim(2)
#         bank = deepcopy(solutions)
#         solutions[0].fused = True
#         solutions.fuse(bank)
#         cs = [solution.c for solution in solutions]
#         fs = [ele for solution in solutions for ele in solution.features]
#         fitnesses = [solution.fitness for solution in solutions]
#         fused = [solution.fused for solution in solutions]
#         self.assertListEqual(cs, [2, 2, 2])
#         self.assertListEqual(fs, [1, 4, 4, 1])
#         self.assertListEqual(fitnesses, [None, None, None])
#         self.assertListEqual(fused, [True, True, False])

#     def test_max_fitness(self):
#         cmin = 2
#         cmax = 3
#         features = [1, 4]
#         solutions = Solutions(cmin, cmax, features)
#         init_fitnesses = [0.5, 0.9, 0.2, 0.7]
#         for i, fitness in enumerate(init_fitnesses):
#             solutions[i].fitness = fitness
#         self.assertEqual(solutions.max_fitness(), 0.9)


# if __name__ == "__main__":
#     unittest.main()
