# DNN Numerical Range and Precision Analysis

## Repository contents

This repository contains only the files needed to run and review the assignment:

- `analyze_dnn.py`: FP32 training and inference analysis script
- `requirements.txt`: Python package requirements
- `results/`: generated CSV summaries and distribution plots

The `.venv` folder and downloaded datasets are not uploaded to the repository. They are created locally or in Google Colab when the code is run.

## Assignment

This project studies the numerical values used by deep neural networks during training and inference. All experiments are performed using FP32. The observed values are then compared with FP16, FP8, INT16, and INT8 ranges.

## Training analysis

The training experiment uses:

- Model: ResNet-18
- Dataset: CIFAR-10
- Data type: FP32

The program records the following values separately for each layer:

- Inputs
- Weights
- Biases
- Activations
- Gradients
- Weight updates

Values are collected at the beginning, middle, and end of training. Convolution, batch-normalization, activation, pooling, and linear layers are included.

## Inference analysis

The following model and dataset combinations can be run:

- ResNet-18 with CIFAR-10
- MobileNetV2 with CIFAR-100
- LeNet-5 with MNIST

During inference, the program records inputs, weights, biases, activations, and output logits. The complete test dataset is used for inputs, activations, and logits.

## Measurements

For every recorded tensor, the program calculates:

- Minimum value
- Maximum value
- Smallest nonzero magnitude
- Spacing between observed values
- FP16 and FP8 range checks
- INT8 and INT16 scale and reconstruction error

It also creates a distribution plot showing the full numerical range and a magnified view around zero.

## Running in Google Colab

Upload `analyze_dnn.py` and `requirements.txt` to Google Drive. Mount Drive in Colab and move to the folder containing the files:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd "/content/drive/MyDrive/Colab Notebooks"
```

Install the required packages:

```python
!pip install uv
!uv pip install --system -r requirements.txt
```

Select a GPU using `Runtime > Change runtime type > T4 GPU` and run:

```python
!python analyze_dnn.py train \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 20 \
    --batch 512 \
    --device cuda \
    --data /content/data \
    --out "/content/drive/MyDrive/Colab Notebooks/results"
```

Inference examples:

```python
!python analyze_dnn.py infer --model resnet18 --dataset cifar10 --device cuda
!python analyze_dnn.py infer --model mobilenetv2 --dataset cifar100 --device cuda
!python analyze_dnn.py infer --model lenet5 --dataset mnist --device cuda
```

## Output files

The program creates a `results` folder containing:

- `summary.csv`: numerical results for every data type and layer
- `plots/`: distribution plots

The datasets are downloaded automatically if they are not already present.

The reduced-precision formats are not used for training or inference. They are only studied using the FP32 values collected during the experiments.
