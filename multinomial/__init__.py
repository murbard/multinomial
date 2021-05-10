from random import randint


# Return n chooses k, can (and should be) cached
def binomial(n, k):
    if 2 * k > n:
        return binomial(n, n - k)
    if k < 0 or k > n:
        return 0
    r = 1
    for i in range(1, k + 1):
        r = (r * (n - i + 1)) // i
    return r


def int_sqrt(n):
    """
    Computes the closest integer to the square root of n
    :param n: integer whose square root is being taken
    :return: the cloest integer to the square root of n
    """
    s0 = 1
    s1 = n
    while s0 != s1:
        s0, s1 = s1, round((s1 * s1 + n) / (2 * s1))
    return s0


def sample_bernoulli():
    """
    Returns a bernoulli trial
    :return: 0 with probability 1/2 and 1 with probability 1/2
    """
    return randint(0, 1)


def sample_binomial(n):
    """
    Samples from the binomial distribution with probability 1/2 and n elements.
    Equivalent to the sum of n bernoulli trials.
    Rejection sampling algorithm adapted from https://people.mpi-inf.mpg.de/~kbringma/paper/2014ICALP.pdf section A.2
    :param n:
    :return:
    """
    if n < 0:
        print('wut?')
    if n == 0:
        return 0
    if n % 2 == 1:  # if n is odd, reduce to the even case
        return sample_binomial(n - 1) + sample_bernoulli()
    n //= 2
    # Sample from "2n" using rejection sampling
    m = int_sqrt(2 * n) + 1
    while True:
        sign = 2 * sample_bernoulli() - 1
        k = 0
        while sample_bernoulli() > 0:
            k += 1
        if sign < 0:
            k = - k - 1
        i = k * m + randint(0, m - 1)
        u = randint(0, 4 ** (n + 1))
        if u < binomial(2 * n, n + i) * (m * 2 ** (max(k, -k - 1))):
            return n + i



def sample_binomial_p(n, p, q):
    """
    Samples from a binomial distribution with a probability parameter p/q.
    Uses the trick described in https://link.springer.com/article/10.1007/s00453-015-0077-8
    :param n: number of Bernoulli trials
    :param p: numerator of a fraction indicating the probability
    :param q: denominator of a fraction indicating the probability
    :return: sum of n Bernoulli trials with probability p/q
    """
    if n == 0:
        return 0
    b = sample_binomial(n)
    if 2 * p >= q:
        return b + sample_binomial_p(n - b, 2 * p - q, q)
    else:
        return sample_binomial_p(b, 2 * p, q)


def sample_multinomial_p(n, rs):
    """
    Samples from a multinomial distribution, equivalent to drawing n samples from a categorical distribution
    with probability weighted by the integer list rs.
    :param n: Number of trials
    :param rs: list of integers representing weights
    :return: number of trials that fell in each of |rs| buckets
    """
    total = sum(rs)
    result = []
    for r in rs: # Draw the multinomial by factoring the distribution into binomials
        b = sample_binomial_p(n, r, total)
        result.append(b)
        n -= b
        total -= r
    return result