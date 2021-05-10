import unittest
from collections import Counter
from math import sqrt
import scipy.stats
from multinomial import binomial, int_sqrt, sample_binomial, sample_binomial_p, sample_multinomial_p


def get_tuples(length, total):
    """
    Computes all possible multinomial draws, the support of the multinomial distribution
    :param length: number of categories
    :param total: sum of the multinomial
    :return: list with all possible multinomial draws
    """
    if length == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for t in get_tuples(length - 1, total - i):
            yield (i,) + t


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

            m = int_sqrt(2 * n) + 1
            self.assertGreaterEqual(m, sqrt(2 * n))
            self.assertLessEqual(m, sqrt(2 * n) + 3)

    def sample_binomial_tester(self, n, N):
        """
        Helper function to test sample_binomial using the chi-square test
        :param n: parameter n of the binomial
        :param N: number of draws from the binomial distribution B(n, 1/2)
        :return: nothing, just makes assertions
        """
        k = [sample_binomial(n) for _ in range(0, N)]
        c = Counter(k)
        if n == 0:
            self.assertEqual(c, Counter({0: N}))
            return
        pmf = scipy.stats.binom(n=n, p=0.5).pmf
        test = scipy.stats.chisquare([c[i] / N for i in range(0, n + 1)], [pmf(i) for i in range(0, n + 1)])
        self.assertGreater(test.pvalue, 0.001)  # yes this is not very sound

    def test_sample_binomial(self):
        self.sample_binomial_tester(20, 1000)
        self.sample_binomial_tester(0, 1000)
        self.sample_binomial_tester(1, 2000)
        self.sample_binomial_tester(13, 2000)
        pass

    def sample_binomial_p_tester(self, n, N, p, q):
        """
        Helper function to test sample_binomial_p using the chi-square test
        :param n: parameter n of sample_binomial_p
        :param N: number of draws from the binomial distribution B(n, p/q)
        :param p: parameter p of sample_binomial_p
        :param q: parameter q of sample_binomial_p
        """
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
        """
        Helper function to test sample_multinomial_p using the chi-square test
        :param n: parameter n of sample_multinomial_p
        :param N: number of draws from the multinomial distribution M(n, [r/sum(rs) for r in rs])
        :param rs: parameter n of sample_multinomial_p
        """
        k = [tuple(sample_multinomial_p(n, rs)) for i in range(0, N)]
        s = sum(rs)
        p = [r / s for r in rs]
        pmf = scipy.stats.multinomial(n, p).pmf
        c = Counter(k)
        events = list(get_tuples(len(rs), n))
        test = scipy.stats.chisquare([c[e] / N for e in events], [pmf(e) for e in events])
        self.assertGreater(test.pvalue, 0.001)  # yes this is not very sound

    def test_sample_multinomial_p(self):
        self.sample_multinomial_p_tester(10, 2000, [14, 5, 7])


if __name__ == '__main__':
    unittest.main()
