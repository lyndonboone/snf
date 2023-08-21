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

from snftools.snf import SNF


digits = datasets.load_digits()
testing_dir = Path(__file__).resolve().parent / "_snf"


class TestSNFDigits(unittest.TestCase):
    def do(self, K, mu, t, rtol):
        affinity_networks = make_affinity(digits.data, metric="sqeuclidean", K=K, mu=mu)
        fused_network = SNF(affinity_networks, K=K, t=t)
        df = read_csv(testing_dir / f"digits/K-{K}_mu-{mu:.1f}_t-{t}.csv")
        # pay attention to rtol below. Trying to work it down
        np.testing.assert_allclose(fused_network, df.values, rtol=rtol)

    def test_sqeuclidean_K10_mu05_t10(self):
        self.do(K=10, mu=0.5, t=10, rtol=1e-7)

    def test_sqeuclidean_K10_mu05_t20(self):
        self.do(K=10, mu=0.5, t=20, rtol=1e-7)

    def test_sqeuclidean_K20_mu05_t10(self):
        self.do(K=20, mu=0.5, t=10, rtol=1e-7)

    def test_sqeuclidean_K20_mu05_t20(self):
        self.do(K=20, mu=0.5, t=20, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
