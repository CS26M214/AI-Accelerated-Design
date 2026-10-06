"""Manual BCHW <-> B x (CHW) conversion and reproducible experiments.

The two conversion functions below use explicit element indexing and loops.
NumPy is used only as tensor storage and for element access; no library shape
conversion is used by the required algorithms.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import load_sample_image


def flatten_bchw(tensor: np.ndarray) -> np.ndarray:
    """Map a BCHW tensor to B x (C*H*W) in C, then H, then W order."""
    if tensor.ndim != 4:
        raise ValueError(f"Expected a 4-D BCHW tensor, received {tensor.ndim}-D")

    batch_size, channels, height, width = tensor.shape
    flat_width = channels * height * width
    flattened = np.empty((batch_size, flat_width), dtype=tensor.dtype)

    for b in range(batch_size):
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    j = c * height * width + h * width + w
                    flattened[b, j] = tensor[b, c, h, w]

    return flattened


def reconstruct_bchw(
    flattened: np.ndarray, channels: int, height: int, width: int
) -> np.ndarray:
    """Reconstruct BCHW by deriving each coordinate from its flat index."""
    if flattened.ndim != 2:
        raise ValueError(f"Expected a 2-D flattened tensor, received {flattened.ndim}-D")
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError("C, H, and W must all be positive")

    batch_size, flat_width = flattened.shape
    expected_width = channels * height * width
    if flat_width != expected_width:
        raise ValueError(
            f"Flat width is {flat_width}; expected C*H*W = {expected_width}"
        )

    reconstructed = np.empty(
        (batch_size, channels, height, width), dtype=flattened.dtype
    )
    spatial_size = height * width

    for b in range(batch_size):
        for j in range(flat_width):
            c = j // spatial_size
            remainder = j % spatial_size
            h = remainder // width
            w = remainder % width
            reconstructed[b, c, h, w] = flattened[b, j]

    return reconstructed


def _sample_image_tensors() -> tuple[np.ndarray, np.ndarray]:
    """Prepare a small grayscale and RGB crop from scikit-learn's sample photo."""
    image = load_sample_image("china.jpg")
    height = min(64, image.shape[0])
    width = min(96, image.shape[1])

    rgb_bchw = np.empty((1, 3, height, width), dtype=image.dtype)
    grayscale_bchw = np.empty((1, 1, height, width), dtype=np.float32)

    for h in range(height):
        for w in range(width):
            red = float(image[h, w, 0])
            green = float(image[h, w, 1])
            blue = float(image[h, w, 2])
            grayscale_bchw[0, 0, h, w] = (
                0.299 * red + 0.587 * green + 0.114 * blue
            )
            for c in range(3):
                rgb_bchw[0, c, h, w] = image[h, w, c]

    return grayscale_bchw, rgb_bchw


def _measure_case(name: str, tensor: np.ndarray) -> dict[str, Any]:
    batch_size, channels, height, width = tensor.shape
    flattened = flatten_bchw(tensor)
    reconstructed = reconstruct_bchw(flattened, channels, height, width)

    # Cast before subtraction so unsigned image values cannot wrap around.
    difference = np.abs(
        tensor.astype(np.float64) - reconstructed.astype(np.float64)
    )
    return {
        "name": name,
        "B": batch_size,
        "C": channels,
        "H": height,
        "W": width,
        "Emax": float(difference.max()),
        "MAE": float(difference.mean()),
    }


def run_experiments() -> list[dict[str, Any]]:
    """Run two bundled image cases and seven seeded feature-map cases."""
    grayscale, rgb = _sample_image_tensors()
    cases: list[tuple[str, np.ndarray]] = [
        ("Sample photo, grayscale crop", grayscale),
        ("Sample photo, RGB crop", rgb),
    ]

    generator = np.random.default_rng(20261005)
    batch_size, height, width = 2, 8, 8
    for channels in (8, 16, 32, 64, 128, 256, 500):
        feature_map = generator.uniform(
            low=-1.0,
            high=1.0,
            size=(batch_size, channels, height, width),
        ).astype(np.float32)
        cases.append((f"Synthetic feature map C={channels}", feature_map))

    return [_measure_case(name, tensor) for name, tensor in cases]


def print_results(results: list[dict[str, Any]]) -> None:
    """Print the measured dimensions and reconstruction errors."""
    headings = ("Input", "B", "C", "H", "W", "Emax", "MAE")
    rows = [
        (
            row["name"],
            str(row["B"]),
            str(row["C"]),
            str(row["H"]),
            str(row["W"]),
            f'{row["Emax"]:.8g}',
            f'{row["MAE"]:.8g}',
        )
        for row in results
    ]
    widths = [max(len(headings[i]), *(len(row[i]) for row in rows)) for i in range(7)]
    print(" | ".join(headings[i].ljust(widths[i]) for i in range(7)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(7)))


if __name__ == "__main__":
    print_results(run_experiments())
