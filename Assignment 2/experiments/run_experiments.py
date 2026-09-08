import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lns import LNS8, LNS16, add, multiply, mac, decode, encode


def error(reference, result):
    if reference == 0:
        return abs(reference - result)
    return abs(reference - result) / abs(reference)


def values(dtype):
    if dtype == np.float16:
        fixed = [0, 1, -1, .001, -.001, 100, -100, 60000, -60000]
        random_values = np.random.default_rng(1).uniform(-100, 100, 1000)
    else:
        fixed = [0, 1, -1, 1e-8, -1e-8, 100, -100, 1e-30, 1e30]
        random_values = np.random.default_rng(1).uniform(-1e5, 1e5, 1000)
    return [dtype(x) for x in fixed + list(random_values)]


def reference(kind, a, b, c, dtype):
    a, b, c = dtype(a), dtype(b), dtype(c)
    if kind == "conversion":
        return a
    if kind == "addition":
        return dtype(a + b)
    if kind == "multiplication":
        return dtype(a * b)
    return dtype(dtype(a * b) + c)


def lns_result(kind, a, b, c, fmt, dtype):
    a, b, c = encode(a, fmt, "fp16" if dtype == np.float16 else "fp32"), encode(b, fmt, "fp16" if dtype == np.float16 else "fp32"), encode(c, fmt, "fp16" if dtype == np.float16 else "fp32")
    if kind == "conversion":
        return decode(a)
    if kind == "addition":
        return decode(add(a, b, fmt))
    if kind == "multiplication":
        return decode(multiply(a, b, fmt))
    return decode(mac(a, b, c, fmt))


def run():
    rows = []
    for dtype in [np.float32, np.float16]:
        data = values(dtype)
        for kind in ["conversion", "addition", "multiplication", "mac"]:
            for fmt in [LNS8, LNS16]:
                errors = []
                for i, a in enumerate(data):
                    b = data[(i + 1) % len(data)]
                    c = data[(i + 2) % len(data)]
                    ref = reference(kind, a, b, c, dtype)
                    result = lns_result(kind, a, b, c, fmt, dtype)
                    if np.isfinite(ref):
                        errors.append(error(float(ref), result))
                rows.append({"input": dtype.__name__, "operation": kind,
                             "format": fmt.name, "mean_error": np.mean(errors),
                             "maximum_error": np.max(errors), "samples": len(errors)})

    os.makedirs("results", exist_ok=True)
    with open("results/errors.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with open("results/summary.txt", "w") as file:
        for row in rows:
            file.write(str(row) + "\n")
    print("Saved results/errors.csv and results/summary.txt")


if __name__ == "__main__":
    run()
