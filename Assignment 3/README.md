# AI Accelerator Design - Assignment 3

This directory contains the manual BCHW tensor flattening implementation and
the concise report with experiment results.

## Run

Install the dependencies and execute from this directory:

```sh
python3 -m pip install -r requirements.txt
python3 tensor_flat.py
python3 build_report.py
```

`tensor_flat.py` prints the measured dimensions, maximum absolute error, and
mean absolute error. `build_report.py` reruns the same deterministic experiments
and writes `AIAD_Assignment3_Report.pdf` beside the source files.

The input image cases use the bundled scikit-learn `china.jpg` sample. The
feature maps use a fixed random seed so results can be reproduced. Conversion
uses explicit loops and element indexing; no library shape-conversion function
is used in either required algorithm.
