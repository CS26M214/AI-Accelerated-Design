# Assignment 2: Logarithmic Number System Library

This assignment implements basic arithmetic in the Logarithmic Number System (LNS).

## Files

- `lns/core.py`: LNS formats, conversion, addition, multiplication, and MAC.
- `lns/__init__.py`: makes the library importable using `from lns import ...`.
- `tests/test_lns.py`: basic tests for signs, zero, random values, underflow, and overflow.
- `experiments/run_experiments.py`: compares LNS results with FP32 and FP16 reference values.
- `results/errors.csv`: mean and maximum errors for each operation and format.
- `results/summary.txt`: text version of the experiment results.
- `pyproject.toml`: package configuration.
- `requirements.txt`: packages needed to run the tests and experiments.

## LNS format used

Each number has one sign bit and the remaining bits store a quantized base-2 logarithm. The largest logarithm code is reserved for zero.

- LNS8: 1 sign bit, 7 logarithm bits, 3 fractional bits. The logarithm range is approximately -8 to 7.75.
- LNS16: 1 sign bit, 15 logarithm bits, 8 fractional bits. The logarithm range is approximately -64 to 63.996.

The logarithm is rounded to the nearest available code. Values below the minimum representable magnitude underflow to zero. Values above the maximum representable magnitude saturate to the maximum finite value.

## Arithmetic

The operations use logarithm identities directly:

- Multiplication: `log2(|a*b|) = log2(|a|) + log2(|b|)`
- Addition: `log2(a+b) = log2(a) + log2(1 + 2^(log2(b)-log2(a)))` for equal signs
- MAC: multiplication followed by LNS addition

Operands are not converted back to floating point before these operations. Only the final result is decoded for comparison with FP32 or FP16.

## Running

From this directory:

```bash
uv pip install -r requirements.txt
pytest
python experiments/run_experiments.py
```

The experiment tests positive, negative, zero, small, large, and randomly generated values. It reports relative error using `|reference - LNS| / |reference|`. For a zero reference, absolute error is used.
