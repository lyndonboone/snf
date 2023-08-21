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


from pathlib import Path
import unittest

import numpy as np
from pandas import read_csv
from snf import datasets
from snf import make_affinity


digits = datasets.load_digits()
testing_dir = Path(__file__).resolve().parent / "_make_affinity_matrices"


class TestMakeAffinityDigits(unittest.TestCase):
    def do(self, K, mu):
        affinity_networks = make_affinity(digits.data, metric="sqeuclidean", K=K, mu=mu)
        for i, net in enumerate(affinity_networks):
            df = read_csv(testing_dir / f"digits/K-{K}_mu-{mu:.1f}/{i}.csv")
            np.testing.assert_allclose(net, df.values)

    def test_sqeuclidean_K20_mu05(self):
        self.do(K=20, mu=0.5)

    def test_sqeuclidean_K10_mu03(self):
        self.do(K=10, mu=0.3)

    def test_sqeuclidean_K20_mu08(self):
        self.do(K=20, mu=0.8)


if __name__ == "__main__":
    unittest.main()
