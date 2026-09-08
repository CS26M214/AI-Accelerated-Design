import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LNSFormat:
    name: str
    bits: int
    fractional_bits: int
    log_min: float

    @property
    def magnitude_bits(self):
        return self.bits - 1

    @property
    def scale(self):
        return 2 ** self.fractional_bits

    @property
    def zero_code(self):
        return (2 ** self.magnitude_bits) - 1

    @property
    def log_max(self):
        return self.log_min + (self.zero_code - 1) / self.scale

    @property
    def min_value(self):
        return 2 ** self.log_min

    @property
    def max_value(self):
        return 2 ** self.log_max


LNS16 = LNSFormat("LNS16", 16, 8, -64)
LNS8 = LNSFormat("LNS8", 8, 3, -8)


class LNSNumber:
    def __init__(self, bits, fmt):
        self.bits = int(bits)
        self.fmt = fmt

    @property
    def is_zero(self):
        return (self.bits & (self.fmt.zero_code)) == self.fmt.zero_code

    @property
    def negative(self):
        return bool(self.bits >> self.fmt.magnitude_bits)

    @property
    def log_value(self):
        if self.is_zero:
            return -math.inf
        code = self.bits & self.fmt.zero_code
        return self.fmt.log_min + code / self.fmt.scale

    def to_float(self):
        if self.is_zero:
            return 0.0
        value = 2 ** self.log_value
        return -value if self.negative else value

    def __repr__(self):
        return f"{self.fmt.name}({self.bits:#x}, {self.to_float()})"


def _format(fmt):
    if isinstance(fmt, LNSFormat):
        return fmt
    if fmt == 8:
        return LNS8
    if fmt == 16:
        return LNS16
    raise ValueError("format must be LNS8, LNS16, 8, or 16")


def encode(value, fmt=LNS16, input_dtype="fp32"):
    fmt = _format(fmt)
    if input_dtype == "fp16":
        import numpy as np
        value = float(np.float16(value))
    else:
        value = float(value)

    if math.isnan(value):
        raise ValueError("NaN is not supported")
    if value == 0:
        return LNSNumber(fmt.zero_code, fmt)

    negative = value < 0
    log_value = math.log2(abs(value))
    return _from_log(log_value, negative, fmt)


def decode(value):
    if not isinstance(value, LNSNumber):
        raise TypeError("decode expects an LNSNumber")
    return value.to_float()


def _from_log(log_value, negative, fmt):
    if log_value == -math.inf or log_value < fmt.log_min:
        return LNSNumber(fmt.zero_code, fmt)
    if log_value == math.inf or log_value > fmt.log_max:
        code = fmt.zero_code - 1
    else:
        code = int(math.floor((log_value - fmt.log_min) * fmt.scale + 0.5))
        code = max(0, min(code, fmt.zero_code - 1))
    sign = 1 if negative else 0
    return LNSNumber((sign << fmt.magnitude_bits) | code, fmt)


def _number(value, fmt):
    if isinstance(value, LNSNumber):
        if value.fmt != fmt:
            raise ValueError("both operands must use the same LNS format")
        return value
    return encode(value, fmt)


def add(a, b, fmt=LNS16):
    fmt = _format(fmt)
    a = _number(a, fmt)
    b = _number(b, fmt)
    if a.is_zero:
        return b
    if b.is_zero:
        return a

    la, lb = a.log_value, b.log_value
    if la < lb:
        a, b = b, a
        la, lb = lb, la

    ratio = 2 ** (lb - la)
    if a.negative == b.negative:
        new_log = la + math.log2(1 + ratio)
        negative = a.negative
    else:
        if la == lb:
            return LNSNumber(fmt.zero_code, fmt)
        new_log = la + math.log2(1 - ratio)
        negative = a.negative
    return _from_log(new_log, negative, fmt)


def multiply(a, b, fmt=LNS16):
    fmt = _format(fmt)
    a = _number(a, fmt)
    b = _number(b, fmt)
    if a.is_zero or b.is_zero:
        return LNSNumber(fmt.zero_code, fmt)
    return _from_log(a.log_value + b.log_value, a.negative != b.negative, fmt)


def mac(a, b, c, fmt=LNS16):
    fmt = _format(fmt)
    product = multiply(a, b, fmt)
    return add(product, c, fmt)
