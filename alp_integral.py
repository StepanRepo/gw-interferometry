#! /bin/python
import numpy as np
from scipy.special import loggamma as lgm
from py3nj import wigner3j as w3j
from sympy.physics.wigner import wigner_3j
from scipy.special import logsumexp


def logI2(l1, m1, l2, m2):
    """
    Calculate the overlap integral of two associated Legendre polynomials P_{l1}^{m1} and P_{l2}^{m2}.

    Parameters:
    l1, m1 : int
        Degree and order of the first associated Legendre polynomial.
    l2, m2 : int
        Degree and order of the second associated Legendre polynomial.

    Returns:
    float
        The value of the overlap integral I(l1, m1; l2, m2).
    """
    # Calculate the phase delta (equation 10)
    delta = min(m1, m2)
    mabs = np.abs(m1-m2)

    if np.abs(m1) > l1 or np.abs(m2) > l2:
        return -np.inf, 0.0

    # Special case m1=m2
    if m1 == m2:
        if l1 == l2:
            return np.log(2/(2*l1 + 1)) + lgm(l1 + m1 + 1) - lgm(l1 - m1 + 1), 1.0
        else:
            return -np.inf, 0.0

    # Calculate C(l1, m1; l2, m2) (equation 8)
    sq = lgm(l1 + m1 + 1) + lgm(l2 + m2 + 1) -\
            lgm(l1 - m1 + 1) - lgm(l2 - m2 + 1)
    sq = sq/2

    log_C = np.log(mabs) + np.log(2) * (mabs - 2) + sq
    sign_C = (-1)**delta

    # Determine the range of l values (|l1 - l2| <= l <= l1 + l2, l >= |m2 - m1|, and l + l1 + l2 even)
    l_min = max(abs(l1 - l2), abs(m2 - m1))
    l_max = l1 + l2

    # Create array of all possible l values
    l_values = np.arange(l_min, l_max + 1)
    # Filter for l values where l + l1 + l2 is even
    mask = (l_values + l1 + l2) % 2 == 0
    l_values = l_values[mask]

    mask = np.logical_and(
            np.abs(l1 - l2) <= l_values,
            l_values <= l1 + l2)
    l_values = l_values[mask]

    if len(l_values) == 0:
        return -np.inf, 0.0

    # Vectorized calculation of term1
    term1 = 1 + (-1)**(l_values + mabs)
    valid_mask = term1 != 0

    # Apply mask to l_values and term1
    l_values = l_values[valid_mask]
    term1 = term1[valid_mask]

    if len(l_values) == 0:
        return -np.inf, 0.0

    # Vectorized calculation of term2
    term2 = (lgm(l_values - mabs + 1) - lgm(l_values + mabs + 1))/2

    # Vectorized calculation of term3
    term3 = lgm(l_values/2) + lgm((l_values + mabs + 1)/2) -\
            lgm((l_values - mabs)/2 + 1) - lgm((l_values + 3)/2)

    # Vectorized calculation of 3-j symbols
    three_j_1 = w3j(2*l1, 2*l2, 2*l_values, 0, 0, 0)
    three_j_2 = w3j(2*l1, 2*l2, 2*l_values, -2*m1, 2*m2, 2*(m1 - m2))

    js = np.log(np.abs(three_j_1)) + np.log(np.abs(three_j_2))

    # Vectorized calculation of log_sum and signs
    log_sum = term2 + term3 + js + np.log(term1 * (2*l_values + 1))
    signs = np.sign(three_j_1) * np.sign(three_j_2)

    summ, sign = logsumexp(log_sum, b=signs, return_sign=True)

    summ = summ + log_C 
    sign = sign * sign_C

    return summ, sign



if __name__ == "__main__":
    a = logI2(3, 2, 3, -2)
    print(np.exp(a[0]) * a[1])
