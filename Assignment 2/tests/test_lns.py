import math
import random

from lns import LNS8, LNS16, add, multiply, mac, encode, decode


def close(a, b, tolerance):
    return abs(a - b) / max(abs(a), 1e-30) < tolerance


def test_conversion_and_sign():
    for fmt, tolerance in [(LNS8, .1), (LNS16, .01)]:
        for value in [0, 1, -1, .5, -.5, 10, -10]:
            assert close(value, decode(encode(value, fmt)), tolerance)


def test_zero_and_addition():
    for fmt in [LNS8, LNS16]:
        assert decode(add(0, 2, fmt)) == decode(encode(2, fmt))
        assert decode(add(2, -2, fmt)) == 0


def test_multiply_and_mac():
    for fmt, tolerance in [(LNS8, .2), (LNS16, .02)]:
        assert close(6, decode(multiply(2, 3, fmt)), tolerance)
        assert close(11, decode(mac(2, 3, 5, fmt)), tolerance)


def test_underflow_and_overflow_are_finite():
    for fmt in [LNS8, LNS16]:
        small = decode(encode(2 ** (fmt.log_min - 2), fmt))
        large = decode(encode(2 ** (fmt.log_max + 2), fmt))
        assert small == 0
        assert math.isfinite(large)


def test_random_values():
    random.seed(1)
    for fmt in [LNS8, LNS16]:
        for _ in range(100):
            value = random.choice([-1, 1]) * 2 ** random.uniform(fmt.log_min, fmt.log_max)
            assert math.isfinite(decode(encode(value, fmt)))
