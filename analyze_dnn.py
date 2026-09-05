import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, mobilenet_v2
import matplotlib.pyplot as plt


class LeNet5(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.AvgPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 4 * 4, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, nclasses))

    def forward(self, x):
        return self.classifier(self.features(x))


def get_model(name, nclasses):
    if name == 'resnet18':
        model = resnet18(weights=None, num_classes=nclasses)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
        return model
    if name == 'mobilenetv2':
        model = mobilenet_v2(weights=None, num_classes=nclasses)
        model.features[0][0] = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        return model
    return LeNet5(nclasses)


def get_data(name, folder):
    if name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.4914, .4822, .4465), (.247, .243, .261))])
        train = torchvision.datasets.CIFAR10(folder, True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10(folder, False, download=True, transform=transform)
        return train, test, 10
    if name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.507, .487, .441), (.267, .256, .276))])
        train = torchvision.datasets.CIFAR100(folder, True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR100(folder, False, download=True, transform=transform)
        return train, test, 100
    transform = transforms.ToTensor()
    train = torchvision.datasets.MNIST(folder, True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(folder, False, download=True, transform=transform)
    return train, test, 10


class Values:
    def __init__(self):
        self.parts = []
        self.count = 0
        self.minimum = None
        self.maximum = None
        self.smallest = None

    def add(self, tensor):
        values = tensor.detach().cpu().numpy().astype('float32').flatten()
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return
        self.count += len(values)
        self.minimum = values.min() if self.minimum is None else min(self.minimum, values.min())
        self.maximum = values.max() if self.maximum is None else max(self.maximum, values.max())
        nonzero = np.abs(values[values != 0])
        if len(nonzero):
            self.smallest = nonzero.min() if self.smallest is None else min(self.smallest, nonzero.min())
        current = sum(len(x) for x in self.parts)
        if current < 30000:
            self.parts.append(values[:30000 - current])

    def sample(self):
        return np.concatenate(self.parts) if self.parts else np.array([])


def plot_values(values, filename, title):
    x = values.sample()
    if len(x) == 0:
        return
    limit = max(abs(x.min()), abs(x.max()))
    small_limit = max(limit * .02, 1e-7)
    small = x[np.abs(x) <= small_limit]
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.hist(x, bins=100, density=True)
    plt.title('Overall range')
    plt.xlabel('value')
    plt.ylabel('probability density')
    plt.subplot(1, 2, 2)
    if len(small):
        plt.hist(small, bins=50, density=True)
    plt.xlim(-small_limit, small_limit)
    plt.title('Magnified around zero')
    plt.xlabel('value')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def add_parameter_values(all_values, model, stage):
    for name, parameter in model.named_parameters():
        layer = name.rsplit('.', 1)[0]
        kind = 'bias' if name.endswith('bias') else 'weights'
        key = (kind, stage, layer)
        if key not in all_values:
            all_values[key] = Values()
        all_values[key].add(parameter)


def save_results(all_values, folder):
    os.makedirs(folder + '/plots', exist_ok=True)
    rows = []
    for (kind, stage, layer), values in all_values.items():
        x = values.sample()
        unique = np.unique(x)
        spacing = np.min(np.diff(unique)) if len(unique) > 1 else 0
        maximum = max(abs(values.minimum), abs(values.maximum))
        scale8 = maximum / 127
        scale16 = maximum / 32767
        q8 = np.clip(np.round(x / scale8), -127, 127) * scale8 if scale8 else x
        q16 = np.clip(np.round(x / scale16), -32767, 32767) * scale16 if scale16 else x
        row = {
            'kind': kind, 'stage': stage, 'layer': layer,
            'count': values.count, 'minimum': values.minimum,
            'maximum': values.maximum, 'smallest_nonzero': values.smallest,
            'spacing_in_sample': spacing, 'int8_scale': scale8,
            'int16_scale': scale16,
            'int8_relative_rmse': np.sqrt(np.mean((x - q8) ** 2)) / (np.sqrt(np.mean(x ** 2)) + 1e-12),
            'int16_relative_rmse': np.sqrt(np.mean((x - q16) ** 2)) / (np.sqrt(np.mean(x ** 2)) + 1e-12),
            'fp16_out_of_range_percent': np.mean(abs(x) > 65504) * 100,
            'fp8_e4m3_out_of_range_percent': np.mean(abs(x) > 448) * 100,
            'fp8_e5m2_out_of_range_percent': np.mean(abs(x) > 57344) * 100}
        filename = folder + '/plots/' + kind + '_' + stage + '_' + layer.replace('.', '_') + '.png'
        plot_values(values, filename, kind + ' ' + stage + ' ' + layer)
        row['plot'] = filename
        rows.append(row)
    pd.DataFrame(rows).to_csv(folder + '/summary.csv', index=False)


def add_activation_hooks(model, all_values, stage):
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            def save(module, inputs, output, name=name):
                if torch.is_tensor(output):
                    key = ('activations', stage, name)
                    if key not in all_values:
                        all_values[key] = Values()
                    all_values[key].add(output)
            hooks.append(module.register_forward_hook(save))
    return hooks


def train(args):
    train_data, _, nclasses = get_data(args.dataset, args.data)
    model = get_model(args.model, nclasses).to(args.device)
    loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
    loss_function = nn.CrossEntropyLoss()
    all_values = {}
    add_parameter_values(all_values, model, 'beginning')
    stages = {0: 'beginning', max(0, args.epochs // 2 - 1): 'middle', args.epochs - 1: 'end'}
    for epoch in range(args.epochs):
        stage = stages.get(epoch, 'other')
        model.train()
        hooks = add_activation_hooks(model, all_values, stage)
        for inputs, labels in loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            key = ('inputs', stage, 'model_input')
            if key not in all_values:
                all_values[key] = Values()
            all_values[key].add(inputs)
            old_weights = {name: p.detach().clone() for name, p in model.named_parameters()}
            optimizer.zero_grad()
            output = model(inputs)
            loss_function(output, labels).backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    key = ('gradients', stage, name.rsplit('.', 1)[0])
                    if key not in all_values:
                        all_values[key] = Values()
                    all_values[key].add(p.grad)
            optimizer.step()
            for name, p in model.named_parameters():
                if name.endswith('weight'):
                    key = ('weight_updates', stage, name.rsplit('.', 1)[0])
                    if key not in all_values:
                        all_values[key] = Values()
                    all_values[key].add(p - old_weights[name])
        for hook in hooks:
            hook.remove()
        if epoch > 0 or stage != 'beginning':
            add_parameter_values(all_values, model, stage)
    save_results(all_values, args.out + '/training')


def inference(args):
    _, test_data, nclasses = get_data(args.dataset, args.data)
    model = get_model(args.model, nclasses).to(args.device).eval()
    loader = DataLoader(test_data, batch_size=args.batch, shuffle=False)
    all_values = {}
    hooks = add_activation_hooks(model, all_values, 'test')
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(args.device)
            key = ('inputs', 'test', 'model_input')
            if key not in all_values:
                all_values[key] = Values()
            all_values[key].add(inputs)
            output = model(inputs)
            key = ('output_logits', 'test', 'model_output')
            if key not in all_values:
                all_values[key] = Values()
            all_values[key].add(output)
    for hook in hooks:
        hook.remove()
    add_parameter_values(all_values, model, 'test')
    save_results(all_values, args.out + '/inference')


parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['train', 'infer'])
parser.add_argument('--model', required=True, choices=['resnet18', 'mobilenetv2', 'lenet5'])
parser.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100', 'mnist'])
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--data', default='./data')
parser.add_argument('--out', default='./results')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

if args.mode == 'train':
    train(args)
else:
    inference(args)
