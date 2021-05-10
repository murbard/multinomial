import unittest
from collections import Counter
from math import sqrt

import scipy.stats
import numpy as np

from multinomial import binomial, int_sqrt, sample_binomial, sample_binomial_p, sample_multinomial_p


class TestMultinomial(unittest.TestCase):

    def test_binomial(self):
        self.assertEqual(binomial(10, 5), 252)
        self.assertEqual(binomial(10, 0), 1)
        self.assertEqual(binomial(10, 10), 1)
        self.assertEqual(sum(binomial(20, i) for i in range(0, 21)), 2 ** 20)

    def test_sqrt(self):
        for n in range(1, 2000, 17):
            s = int_sqrt(n)
            self.assertLess(sqrt(n) - 1, s)
            self.assertLess(s, sqrt(n) + 1)

    def sample_binomial_tester(self, n, N):
        k = [sample_binomial(n) for _ in range(0, N)]
        if n == 0:
            self.assertTrue(np.all(np.array(k) == 0))
            return
        pmf = scipy.stats.binom(n=n, p=0.5).pmf
        c = Counter(k)
        test = scipy.stats.chisquare([c[i] / N for i in range(0, n + 1)], [pmf(i) for i in range(0, n + 1)])
        self.assertGreater(test.pvalue, 0.001)  # yes this is not very sound

    def test_sample_binomial(self):
        self.sample_binomial_tester(20, 1000)
        self.sample_binomial_tester(0, 1000)
        self.sample_binomial_tester(1, 2000)
        self.sample_binomial_tester(13, 2000)
        pass

    def sample_binomial_p_tester(self, n, N, p, q):
        k = [sample_binomial_p(n, p, q) for _ in range(0, N)]
        pmf = scipy.stats.binom(n=n, p=p / q).pmf
        c = Counter(k)
        test = scipy.stats.chisquare([c[i] / N for i in range(0, n + 1)], [pmf(i) for i in range(0, n + 1)])
        self.assertGreater(test.pvalue, 0.001)  # yes this is not very sound

    def test_sample_binomial_p(self):
        self.sample_binomial_p_tester(20, 1000, 716221, 1000000)
        self.sample_binomial_p_tester(103, 2000, 1, 100)
        self.sample_binomial_p_tester(103, 2000, 99, 100)

    def sample_multinomial_p_tester(self, n, N, rs):
        k = [tuple(sample_multinomial_p(n, rs)) for i in range(0, N)]
        s = sum(rs)
        p = [r / s for r in rs]
        pmf = scipy.stats.multinomial(n, p).pmf
        c = Counter(k)
        events = list(np.ndindex(tuple([n + 1 for _ in range(len(rs))])))  # all possible draws
        empirical = np.array([c[e] / N for e in events])
        theoretical = np.array([pmf(e) for e in events])
        positive = np.where(empirical * theoretical > 0)
        empirical = empirical[positive]
        theoretical = theoretical[positive]
        test = scipy.stats.chisquare(empirical, theoretical)
        self.assertGreater(test.pvalue, 0.001)  # yes this is not very sound

    def test_sample_multinomial_p(self):
        self.sample_multinomial_p_tester(5, 2000, [14, 5, 7])


if __name__ == '__main__':
    unittest.main()
